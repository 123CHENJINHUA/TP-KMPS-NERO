"""Pick-and-lift trajectory, recording schema and checks (no Isaac Sim imports)."""

from dataclasses import asdict, dataclass
import json

import numpy as np
from scipy.spatial.transform import Rotation

from nero_bridge.data import poses


TASK = "cube_pick_lift_v1"
# gripper_base -> finger contact center, derived from the supplied Nero meshes.
GRASP_OFFSET = np.array([0., 0., .125])
# A top-down seed within the Nero URDF limits, near TCP [0.30, 0, 0.35].
GRASP_SEED = np.array([2.11862610, .28014185, 1.26384501, 1.67105050,
                       -.21988699, .27441669, 1.38342634])
STAGES = ("open", "descend", "close", "lift", "hold")
# Explicit force-drive units: N/m, N*s/m and N (per prismatic finger).
FINGER_STIFFNESS = 600.
FINGER_DAMPING = 10.
FINGER_MAX_FORCE = 10.
GRASP_X_LIMITS = (.28, .34)
GRASP_Y_LIMITS = (-.05, .05)


def grasp_position_bounds(spread=.012, x_range=None, y_range=None):
    """Supported sampling area, within the table and the checked top-down workspace."""
    if not np.isfinite(spread) or not 0 <= spread <= .02:
        raise ValueError("grasp-spread must be in [0, 0.02]")
    bounds = np.asarray([(.3 - spread, .3 + spread) if x_range is None else x_range,
                         (-spread, spread) if y_range is None else y_range], dtype=float)
    limits = np.asarray([GRASP_X_LIMITS, GRASP_Y_LIMITS])
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1])
            or np.any(bounds[:, 0] < limits[:, 0] - 1e-12)
            or np.any(bounds[:, 1] > limits[:, 1] + 1e-12)):
        raise ValueError("Use ordered X bounds within [0.28, 0.34] and Y bounds within [-0.05, 0.05] meters")
    return bounds


def sample_grasp_positions(count, seed, bounds, sampling="random"):
    """Latin-hypercube sampling covers every interval on each axis exactly once."""
    if count < 1 or seed < 0 or sampling not in {"random", "stratified"}:
        raise ValueError("Use a positive count, nonnegative seed and random/stratified sampling")
    bounds = grasp_position_bounds(x_range=bounds[0], y_range=bounds[1])
    rng = np.random.default_rng(seed)
    if sampling == "random":
        unit = rng.random((count, 2))
    else:
        unit = np.column_stack([(rng.permutation(count) + rng.random(count)) / count for _ in range(2)])
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


@dataclass(frozen=True)
class GraspConfig:
    cube_size: float = .03
    cube_mass: float = .03
    table_height: float = .15
    lift_height: float = .06
    approach_height: float = .06
    open_width: float = .08
    object_name: str = "cube"
    object_origin_height: float = 0.

    def __post_init__(self):
        numeric = [value for key, value in asdict(self).items() if key != "object_name"]
        if not np.isfinite(numeric).all():
            raise ValueError("Grasp parameters must be finite")
        if not (.025 <= self.cube_size <= .04 and 0 < self.cube_mass <= .1
                and .05 <= self.table_height <= .25 and .04 <= self.lift_height <= .1
                and .04 <= self.approach_height <= .1
                and self.cube_size + .02 <= self.open_width <= .1):
            raise ValueError("Grasp configuration is outside the supported small-object range")
        if (not isinstance(self.object_name, str) or not self.object_name.isidentifier()
                or self.object_origin_height < 0
                or (self.object_name != "cube" and not 0 < self.object_origin_height <= .1)):
            raise ValueError("Invalid grasp object name or origin height")

    @property
    def support_height(self):
        return self.cube_size / 2 if self.object_name == "cube" else self.object_origin_height

    @classmethod
    def for_object(cls, name):
        from sim_bridge.objects import load_object_profile
        profile = load_object_profile(name)
        if profile is None:
            return cls()
        return cls(object_name=name, cube_size=profile["grip_width_m"],
                   cube_mass=profile["mass_kg"], object_origin_height=profile["origin_height_m"])

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, value):
        return cls(**json.loads(str(value)))


def grasp_pose(target):
    """Top-down pose for an upright cube; respect its yaw, not roll or pitch."""
    target = poses(target)
    rotation = Rotation.from_quat(target[[4, 5, 6, 3]])
    if np.dot(rotation.apply([0., 0., 1.]), [0., 0., 1.]) < .99:
        raise ValueError("Pick-and-lift requires an upright cube")
    yaw = np.arctan2(rotation.as_matrix()[1, 0], rotation.as_matrix()[0, 0])
    orientation = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    return np.r_[target[:3] - orientation.apply(GRASP_OFFSET), orientation.as_quat()[[3, 0, 1, 2]]]


def grasp_waypoint(target, phase, config):
    """Fixed stages keep demonstrations aligned for TP-KMP phase resampling."""
    if not np.isfinite(phase) or not 0 <= phase <= 1:
        raise ValueError("Grasp phase must be in [0, 1]")
    wanted = grasp_pose(target)
    opening = config.open_width
    if phase < .1:
        height, stage = config.approach_height, 0
    elif phase < .4:
        s = (phase - .1) / .3
        height, stage = config.approach_height * (1 - s * s * (3 - 2 * s)), 1
    elif phase < .65:
        height, opening, stage = 0., 0., 2
    elif phase < .9:
        s = (phase - .65) / .25
        height, opening, stage = config.lift_height * s * s * (3 - 2 * s), 0., 3
    else:
        height, opening, stage = config.lift_height, 0., 4
    wanted[2] += height
    return wanted, opening, stage


def assess_grasp(times, tcp, objects, fingers, target, config, duration):
    """Require sustained lift, an object between the fingers and little relative slip."""
    times = np.asarray(times, dtype=float)
    tcp, objects, target = poses(tcp), poses(objects), poses(target)
    fingers = np.asarray(fingers, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or tcp.shape != (len(times), 7)
            or objects.shape != tcp.shape or fingers.shape != (len(times), 2)
            or not np.isfinite(times).all() or not np.isfinite(fingers).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid synchronized grasp samples")
    hold = times >= duration * .9
    if hold.sum() < 3 or times[-1] < duration - .05:
        return dict(passed=False, reason="Incomplete final hold")
    obj = objects[hold, :3]
    rotations = Rotation.from_quat(tcp[hold][:, [4, 5, 6, 3]])
    relative = rotations.inv().apply(obj - tcp[hold, :3])
    center_error = np.linalg.norm(relative - GRASP_OFFSET, axis=1)
    slip = np.linalg.norm(relative - relative[0], axis=1)
    gap = fingers[hold, 0] - fingers[hold, 1]
    lift = obj[:, 2] - target[2]
    hold_time = float(times[hold][-1] - times[hold][0])
    checks = dict(
        lifted=bool(lift.min() >= config.lift_height * .65),
        between_fingers=bool(center_error.max() <= .018),
        fingers_blocked=bool(np.all((gap >= config.cube_size * .5) & (gap <= config.cube_size * 1.6))),
        stable=bool(slip.max() <= .008),
        held_long_enough=bool(hold_time >= .5),
    )
    before_lift = (times >= duration * .60) & (times < duration * .65)
    gap_before_lift = float(np.median(fingers[before_lift, 0] - fingers[before_lift, 1])) if before_lift.any() else None
    initially_blocked = (gap_before_lift is not None
                         and config.cube_size * .5 <= gap_before_lift <= config.cube_size * 1.6)
    passed = all(checks.values())
    if passed:
        reason = "Object lifted and retained between the fingers"
    elif initially_blocked and not all(checks[key] for key in ("lifted", "between_fingers", "fingers_blocked")):
        reason = "Likely slip during lift: fingers initially blocked at cube width, but cube was not retained"
    elif not initially_blocked:
        reason = "Fingers did not close around the cube before lift; check alignment and collision geometry"
    else:
        reason = "Final hold failed: " + ", ".join(key for key, passed_check in checks.items() if not passed_check)
    return dict(passed=passed, reason=reason, checks=checks,
                pre_lift_finger_gap_m=gap_before_lift,
                max_cube_lift_m=float(np.max(objects[:, 2] - target[2])),
                hold_seconds=hold_time, min_lift_m=float(lift.min()),
                final_lift_m=float(lift[-1]), max_center_error_m=float(center_error.max()),
                max_relative_slip_m=float(slip.max()), final_finger_gap_m=float(gap[-1]))


def grasp_recording_fields(rows, fingers, commands, objects, stages, config, duration, success):
    count = len(rows)
    fingers = np.asarray(fingers, dtype=float)
    commands = np.asarray(commands, dtype=float)
    objects = poses(objects)
    stages = np.asarray(stages)
    if (fingers.shape != (count, 2) or commands.shape != (count,)
            or objects.shape != (count, 7) or stages.shape != (count,)
            or not np.isfinite(fingers).all() or not np.isfinite(commands).all()
            or np.any((commands < 0) | (commands > .1))
            or not np.isin(stages, np.arange(len(STAGES))).all()):
        raise ValueError("Invalid gripper/object recording")
    return dict(task=TASK, grasp_config=config.to_json(), grasp_duration=duration,
                grasp_success=bool(success), gripper_joint_positions=fingers,
                gripper_commands=commands, object_poses=objects, grasp_stages=stages)


def held_commands(source_phase, commands, phase):
    """Keep discrete close events discrete instead of interpolating them early."""
    source_phase, commands = np.asarray(source_phase), np.asarray(commands)
    if (source_phase.ndim != 1 or len(source_phase) < 2 or commands.shape != source_phase.shape
            or not np.isfinite(source_phase).all() or not np.isfinite(commands).all()
            or np.any(np.diff(source_phase) <= 0) or not np.isclose(source_phase[0], 0)
            or not np.isclose(source_phase[-1], 1) or np.any((commands < 0) | (commands > .1))):
        raise ValueError("Invalid gripper phase/command profile")
    indices = np.clip(np.searchsorted(source_phase, phase, side="right") - 1, 0, len(commands) - 1)
    return commands[indices]


def grasp_skill_fields(recordings, samples):
    """Carry the demonstrated gripper schedule alongside the learned TCP model."""
    tasks = [str(data.get("task", "reach")) for data in recordings]
    if all(task == "reach" for task in tasks):
        return {}
    if any(task != TASK for task in tasks):
        raise ValueError("Do not mix grasp and reach demonstrations")
    config = GraspConfig.from_json(recordings[0]["grasp_config"])
    duration = float(recordings[0]["grasp_duration"])
    if not np.isfinite(duration) or duration < 8:
        raise ValueError("Grasp duration must be at least 8 seconds")
    profiles = []
    phase = np.linspace(0, 1, samples)
    for data in recordings:
        if not bool(data["grasp_success"]):
            raise ValueError("Failed grasps cannot be used as successful demonstrations")
        if (GraspConfig.from_json(data["grasp_config"]) != config
                or not np.isclose(float(data["grasp_duration"]), duration)):
            raise ValueError("Grasp demonstrations must share geometry and stage timing")
        time = data["demo"][:, 0]
        source = (time - time[0]) / (time[-1] - time[0])
        profiles.append(held_commands(source, data["gripper_commands"], phase))
    return dict(task=TASK, grasp_config=config.to_json(), grasp_duration=duration,
                gripper_phase=phase, gripper_commands=np.median(profiles, axis=0))


def grasp_prediction_fields(skill):
    if str(skill.get("task", "reach")) == "reach":
        return {}
    if str(skill["task"]) != TASK:
        raise ValueError("Unknown skill task")
    config = GraspConfig.from_json(skill["grasp_config"])
    duration = float(skill["grasp_duration"])
    if not np.isfinite(duration) or duration < 8:
        raise ValueError("Grasp duration must be at least 8 seconds")
    held_commands(skill["gripper_phase"], skill["gripper_commands"], 0.)
    return dict(task=TASK, grasp_config=config.to_json(), grasp_duration=duration,
                gripper_phase=skill["gripper_phase"], gripper_commands=skill["gripper_commands"])
