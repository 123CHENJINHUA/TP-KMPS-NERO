"""Isaac Sim 5.1: interactive collection, automatic collection and TP-KMP replay."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME = np.array([0, -.4, 0, 1., 0, .2, 0])


class GraspCollectionFailed(RuntimeError):
    """A completed collection with rejected grasps; diagnostic files are available."""


def gripper_joint_targets(opening):
    """Map total finger travel (meters) to the two URDF prismatic joints."""
    if not np.isfinite(opening):
        raise ValueError("Gripper opening must be finite")
    half = np.clip(opening, 0., .1) / 2
    return np.array([half, -half])


def resolve_scene_usd(value):
    """Accept a saved USD scene only when it lives inside this project."""
    if not value:
        return None
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Saved scene USD must be inside this project")
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Saved scene must be a .usd, .usda or .usdc file")
    if not path.is_file():
        raise ValueError(f"Saved scene does not exist: {path}")
    return path


def keyboard_key_name(value):
    """Accept both string key names and Carb KeyboardInput enum values."""
    name = value if isinstance(value, str) else getattr(value, "name", "")
    name = name.upper()
    if name in {"1", "2", "3", "4", "5", "6"}:
        return "KEY_" + name
    return name


def keyboard_pose(pose, key, translation_step, rotation_step):
    """Increment a TCP pose along/about world axes; quaternion order is wxyz."""
    from scipy.spatial.transform import Rotation

    bindings = {"I": (0, 1), "K": (0, -1), "J": (1, 1), "L": (1, -1),
                "U": (2, 1), "O": (2, -1),
                "KEY_1": (3, 1), "KEY_2": (3, -1),
                "KEY_3": (4, 1), "KEY_4": (4, -1),
                "KEY_5": (5, 1), "KEY_6": (5, -1)}
    result = np.asarray(pose, dtype=float).copy()
    axis, sign = bindings[key]
    if axis < 3:
        result[axis] += sign * translation_step
    else:
        delta = np.zeros(3)
        delta[axis - 3] = sign * rotation_step
        rotation = Rotation.from_rotvec(delta) * Rotation.from_quat(result[[4, 5, 6, 3]])
        result[3:] = rotation.as_quat()[[3, 0, 1, 2]]
    return result


def apply_ik_target(robot, action, max_step):
    # Limit the command's slew rate, not its error relative to the sagging measured joint.
    previous = robot.get_articulation_controller().get_applied_action().joint_positions[action.joint_indices]
    action.joint_positions = previous + np.clip(action.joint_positions - previous, -max_step, max_step)
    robot.apply_action(action)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["interactive", "dual", "teach-sync", "collect", "grasp", "replay", "smoke"], default="interactive")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default="recordings/session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--trajectory")
    parser.add_argument("--object", default="cube", help="Grasp model: cube or a directory name under assets/objects")
    parser.add_argument("--demos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for cube placement in grasp mode")
    parser.add_argument("--grasp-spread", type=float, default=.012, help="Cube XY variation about [0.30, 0], in meters (0..0.02)")
    parser.add_argument("--grasp-x-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Cube X interval in meters, within [0.28, 0.34]; overrides X spread")
    parser.add_argument("--grasp-y-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Cube Y interval in meters, within [-0.05, 0.05]; overrides Y spread")
    parser.add_argument("--grasp-sampling", choices=["random", "stratified"], default="random",
                        help="Stratified sampling covers every interval on each axis")
    parser.add_argument("--grasp-groups", type=int, default=1,
                        help="Collect this many groups in one simulation; --demos is attempts per group")
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--record-hz", type=int, default=30)
    parser.add_argument("--joint-speed", type=float, default=.8, help="Maximum change of joint targets in rad/s")
    parser.add_argument("--control", choices=["xbox", "keyboard"], default="xbox",
                        help="Interactive input; Xbox is the default")
    parser.add_argument("--gamepad-device", default="auto", help="auto or /dev/input/jsN")
    parser.add_argument("--dual-spacing", type=float, default=.75, help="Distance between fixed bases in meters")
    parser.add_argument("--dual-mount-height", type=float, default=1.0, help="Mount height above ground in meters")
    parser.add_argument("--scene-usd", help="Environment-only USD inside this project, for dual/teach-sync modes")
    parser.add_argument("--wrist-camera-config", default="assets/cameras/wrist_cameras.json",
                        help="JSON with wrist cameras and optional extra_cameras")
    parser.add_argument("--no-wrist-cameras", action="store_true",
                        help="Disable all configured cameras in dual/teach-sync modes")
    parser.add_argument("--no-camera-views", action="store_true",
                        help="Create configured cameras without opening their viewport windows")
    parser.add_argument("--teach-left-can", default="can0", help="CAN channel for Nero 1 in teach-sync mode")
    parser.add_argument("--teach-right-can", default="can1", help="CAN channel for Nero 2 in teach-sync mode")
    parser.add_argument("--teach-stale", type=float, default=.25,
                        help="Hold a simulated arm when CAN data is older than this many seconds")
    parser.add_argument("--teach-start-timeout", type=float, default=5.,
                        help="Seconds to wait for both complete teaching streams")
    parser.add_argument("--gamepad-speed", type=float, default=.03, help="TCP translation speed in m/s")
    parser.add_argument("--gamepad-angle-speed", type=float, default=15., help="TCP rotation speed in deg/s")
    parser.add_argument("--gamepad-deadzone", type=float, default=.12)
    parser.add_argument("--keyboard-step", type=float, default=.001, help="TCP translation per key event in meters")
    parser.add_argument("--keyboard-angle", type=float, default=1., help="TCP rotation per key event in degrees")
    parser.add_argument("--kp", type=float, default=6000.)
    parser.add_argument("--kd", type=float, default=300.)
    parser.add_argument("--position-tolerance", type=float, default=.01)
    parser.add_argument("--angle-tolerance", type=float, default=.1)
    parser.add_argument("--steps", type=int, default=0, help="Optional GUI test frame limit")
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.joint_speed, args.kp, args.kd,
                         args.position_tolerance, args.angle_tolerance, args.keyboard_step, args.keyboard_angle]).all()
            or min(args.duration, args.joint_speed, args.kp, args.kd, args.position_tolerance,
                   args.angle_tolerance, args.keyboard_step, args.keyboard_angle) <= 0):
        parser.error("Duration, gains, joint speed, tolerances and keyboard increments must be finite and positive")
    if (not np.isfinite([args.gamepad_speed, args.gamepad_angle_speed, args.gamepad_deadzone]).all()
            or args.gamepad_speed <= 0 or args.gamepad_angle_speed <= 0
            or not 0 <= args.gamepad_deadzone < 1):
        parser.error("Gamepad speeds must be positive; deadzone must be in [0, 1)")
    if (not np.isfinite([args.dual_spacing, args.dual_mount_height]).all()
            or not .6 <= args.dual_spacing <= 1.5 or not .3 <= args.dual_mount_height <= 1.):
        parser.error("Use dual-spacing in [0.6, 1.5] and dual-mount-height in [0.3, 1.0] meters")
    if args.mode == "dual" and args.control != "xbox":
        parser.error("Dual-arm mode requires --control xbox")
    if args.scene_usd:
        if args.mode not in {"dual", "teach-sync"}:
            parser.error("--scene-usd is supported only in dual and teach-sync modes")
        try:
            args.scene_usd = str(resolve_scene_usd(args.scene_usd))
        except ValueError as exc:
            parser.error(str(exc))
    if args.mode in {"dual", "teach-sync"} and not args.no_wrist_cameras:
        from sim_bridge.wrist_cameras import load_camera_config
        try:
            load_camera_config(args.wrist_camera_config)
        except ValueError as exc:
            parser.error(str(exc))
    if (args.teach_left_can == args.teach_right_can
            or not np.isfinite([args.teach_stale, args.teach_start_timeout]).all()
            or not .05 <= args.teach_stale <= 2.
            or not 1. <= args.teach_start_timeout <= 30.):
        parser.error("Teach sync requires distinct CAN channels, stale in [0.05, 2] and start timeout in [1, 30]")
    if args.physics_hz < 30 or not 1 <= args.record_hz <= args.physics_hz or args.physics_hz % args.record_hz:
        parser.error("physics-hz must be >= 30 and an integer multiple of record-hz")
    if args.demos < 2 or args.steps < 0:
        parser.error("Use at least two demonstrations and nonnegative steps")
    if not np.isfinite(args.grasp_spread) or not 0 <= args.grasp_spread <= .02 or args.seed < 0:
        parser.error("Use grasp-spread in [0, 0.02] and a nonnegative seed")
    if args.grasp_groups < 1:
        parser.error("grasp-groups must be positive")
    from sim_bridge.grasp import grasp_position_bounds, GraspConfig
    try:
        grasp_position_bounds(args.grasp_spread, args.grasp_x_range, args.grasp_y_range)
        GraspConfig.for_object(args.object)
    except ValueError as exc:
        parser.error(str(exc))
    if args.object != "cube" and args.mode not in {"grasp", "replay"}:
        parser.error("--object is supported only in grasp and replay modes")
    if args.mode == "grasp" and args.duration < 8:
        parser.error("Grasp collection needs duration >= 8 seconds for closing and holding")
    if args.mode in {"interactive", "dual", "teach-sync"} and args.headless and not args.steps:
        parser.error("Interactive collection needs the GUI; use --mode collect for headless operation")
    if args.mode == "replay" and not args.trajectory:
        parser.error("Replay requires --trajectory")
    return args


def run(app, args):
    if args.mode in {"dual", "teach-sync"}:
        from sim_bridge.dual import run_dual
        return run_dual(app, args)
    # Omniverse modules must be imported after SimulationApp starts.
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane, VisualCuboid
    from isaacsim.core.prims import Articulation, SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from scipy.spatial.transform import Rotation, Slerp
    from pxr import UsdLux

    from nero_bridge.data import load_npz, poses, save_npz
    from sim_bridge.data import demo_arrays, metrics, require_sim, save_demo, save_json
    from sim_bridge.grasp import (TASK, GRASP_SEED, STAGES, FINGER_STIFFNESS, FINGER_DAMPING,
                                  FINGER_MAX_FORCE, GraspConfig, assess_grasp,
                                  grasp_recording_fields, grasp_waypoint, held_commands,
                                  grasp_position_bounds, sample_grasp_positions)

    replay_data = require_sim(load_npz(args.trajectory)) if args.mode == "replay" else None
    physical_grasp = args.mode == "grasp" or (replay_data is not None and str(replay_data.get("task", "")) == TASK)
    grasp_config = GraspConfig.from_json(replay_data["grasp_config"]) if physical_grasp and replay_data is not None else GraspConfig.for_object(args.object)
    cube_center = np.array([.3, 0., grasp_config.table_height + grasp_config.support_height])
    if physical_grasp and replay_data is not None:
        if args.object != "cube" and args.object != grasp_config.object_name:
            raise ValueError("Replay object must match the skill object; train a separate skill for keel")
        cube_center = poses(replay_data["frames"])[0, :3]
        if args.duration < float(replay_data["grasp_duration"]):
            raise ValueError("Grasp replay duration must be at least the recorded grasp_duration")
        held_commands(replay_data["gripper_phase"], replay_data["gripper_commands"], 0.)
    if physical_grasp:
        output = Path(args.output_dir).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use a new or empty output directory for physical grasp trials")

    enable_extension("isaacsim.asset.importer.urdf")
    enable_extension("isaacsim.robot_motion.motion_generation")
    app.update()
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    from isaacsim.asset.importer.urdf import _urdf

    world = World(stage_units_in_meters=1., physics_dt=1 / args.physics_hz, rendering_dt=1 / 60, device="cpu")
    world.scene.add(GroundPlane("/World/Ground", z_position=-.03, size=4.))
    UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/Light").CreateIntensityAttr().Set(700.)
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDF importer configuration failed")
    config.merge_fixed_joints = False
    config.fix_base = True
    config.import_inertia_tensor = True
    config.distance_scale = 1.
    config.self_collision = False
    config.default_drive_strength = args.kp
    config.default_position_drive_damping = args.kd
    config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    status, articulation_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(ROOT / "assets/nero/nero.urdf"),
        import_config=config, get_articulation_root=True,
    )
    if not status:
        raise RuntimeError("Nero URDF import failed")
    robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name="nero"))
    target = world.scene.add(VisualCuboid("/World/Target", name="target", size=.035, color=np.array([.1, .8, .2])))
    goal = world.scene.add(VisualCuboid("/World/TCPGoal", name="tcp_goal", size=.018, color=np.array([.9, .15, .1])))
    cube = None
    if physical_grasp:
        from sim_bridge.grasp_scene import create_grasp_scene
        cube, physics_settings = create_grasp_scene(world, grasp_config, cube_center)
        save_json(output / "physics.json", physics_settings)
        target.set_visibility(False)  # The orange dynamic cube is the visible task object.
    set_camera_view(eye=np.array([1.25, 1.25, .95]), target=np.array([0, 0, .35]))
    world.reset()
    robot.set_solver_position_iteration_count(64 if physical_grasp else 32)
    if physical_grasp:
        robot.set_solver_velocity_iteration_count(8)
    dynamics = Articulation(prim_paths_expr=articulation_path, name="nero_dynamics")
    dynamics.initialize()
    names = robot.dof_names
    if not all(name in names for name in JOINTS):
        raise RuntimeError(f"Imported articulation is missing Nero joints: {names}")
    indices = np.asarray([names.index(name) for name in JOINTS])
    home = np.zeros(robot.num_dof)
    home[indices] = HOME
    gripper_indices = np.asarray([names.index("gripper_joint1"), names.index("gripper_joint2")])
    gripper_opening = .05
    gripper_command = gripper_joint_targets(gripper_opening)
    home[gripper_indices] = gripper_command
    robot.set_joints_default_state(positions=home)
    controller = robot.get_articulation_controller()
    kps, kds = np.full(robot.num_dof, args.kp), np.full(robot.num_dof, args.kd)
    if physical_grasp:
        kps[gripper_indices], kds[gripper_indices] = FINGER_STIFFNESS, FINGER_DAMPING
    controller.set_gains(kps=kps, kds=kds)
    if physical_grasp:
        controller.set_max_efforts(np.full(2, FINGER_MAX_FORCE), joint_indices=gripper_indices)
    solver = LulaKinematicsSolver(str(ROOT / "assets/lula.yaml"), str(ROOT / "assets/nero/nero.urdf"))
    solver.set_robot_base_pose(np.zeros(3), np.array([1., 0, 0, 0]))
    kinematics = ArticulationKinematicsSolver(robot, solver, "gripper_base")
    stride = args.physics_hz // args.record_hz
    output = Path(args.output_dir).resolve()

    def actual_pose():
        # FK uses measured PhysX joint positions, never the commanded IK solution.
        position, matrix = kinematics.compute_end_effector_pose()
        quaternion = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
        return poses(np.r_[position, quaternion])

    def object_pose(obj):
        p, q = obj.get_world_pose()
        return poses(np.r_[p, q])

    def step():
        nonlocal gripper_command
        if not app.is_running():
            raise InterruptedError("Simulation closed")
        gravity = dynamics.get_generalized_gravity_forces()[0]
        if not np.isfinite(gravity).all():
            raise RuntimeError("Invalid gravity compensation from PhysX")
        robot.set_joint_efforts(gravity)
        # Each finger advances at at most 20 mm/s, independently of arm IK.
        gripper_command = gripper_command + np.clip(
            gripper_joint_targets(gripper_opening) - gripper_command,
            -.02 / args.physics_hz, .02 / args.physics_hz,
        )
        robot.apply_action(ArticulationAction(joint_positions=gripper_command.copy(),
                                            joint_indices=gripper_indices))
        world.step(render=not args.headless)

    def reset():
        nonlocal gripper_opening, gripper_command
        gripper_opening = .05
        gripper_command = gripper_joint_targets(gripper_opening)
        robot.set_joint_positions(home)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        robot.apply_action(ArticulationAction(joint_positions=home))
        for _ in range(args.physics_hz):
            step()
        pose = actual_pose()
        goal.set_world_pose(pose[:3], pose[3:])
        return pose

    def follow(pose):
        action, success = kinematics.compute_inverse_kinematics(
            pose[:3], pose[3:], position_tolerance=.0005, orientation_tolerance=.005,
        )
        if not success or not np.isfinite(action.joint_positions).all():
            return False
        apply_ik_target(robot, action, args.joint_speed / args.physics_hz)
        return True

    def settle(pose, timeout=12.):
        goal.set_world_pose(pose[:3], pose[3:])
        for _ in range(int(timeout * args.physics_hz)):
            if not follow(pose):
                raise RuntimeError("IK failed while approaching the trajectory start")
            step()
            current = actual_pose()
            error = metrics(np.tile(pose, (2, 1)), np.tile(current, (2, 1)), .002, .02)
            if error["passed"]:
                return
        raise RuntimeError(f"Could not settle: {error}; actual_joints={robot.get_joint_positions()}; "
                           f"command={controller.get_applied_action().joint_positions}")

    initial = reset()
    marker = initial.copy()
    marker[:3] += [.025, 0, -.06]
    marker[3:] = [1, 0, 0, 0]
    target.set_world_pose(marker[:3], marker[3:])
    print(f"NERO_READY articulation={articulation_path} joints={names} tcp={initial.tolist()}", flush=True)

    def prepare_grasp(frame, waypoints, opening):
        """Preflight IK, then reset the trial outside the recorded interval."""
        nonlocal gripper_opening, gripper_command
        seed = GRASP_SEED.copy()
        first_joints = None
        for wanted in waypoints:
            joints, success = solver.compute_inverse_kinematics(
                "gripper_base", wanted[:3], wanted[3:], warm_start=seed,
                position_tolerance=.0005, orientation_tolerance=.005,
            )
            if not success or not np.isfinite(joints).all():
                raise RuntimeError("Grasp trajectory failed IK preflight")
            if first_joints is not None and np.max(np.abs(joints - seed)) > .4:
                raise RuntimeError("Grasp trajectory has an IK branch jump")
            seed = joints
            if first_joints is None:
                first_joints = joints.copy()
        trial_home = home.copy()
        trial_home[indices] = first_joints
        gripper_opening = float(opening)
        gripper_command = gripper_joint_targets(gripper_opening)
        trial_home[gripper_indices] = gripper_command
        # Only episode initialization uses set_joint_positions/set_world_pose.
        # During the recorded grasp the cube moves solely through PhysX contact.
        robot.set_joint_positions(trial_home)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        robot.apply_action(ArticulationAction(joint_positions=trial_home))
        cube.set_world_pose(frame[:3], frame[3:])
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        goal.set_world_pose(waypoints[0][:3], waypoints[0][3:])
        for _ in range(args.physics_hz):
            if not follow(waypoints[0]):
                raise RuntimeError("IK failed during grasp initialization")
            step()
        if not metrics(np.tile(waypoints[0], (2, 1)), np.tile(actual_pose(), (2, 1)), .003, .025)["passed"]:
            raise RuntimeError("Robot did not settle at pregrasp pose")
        settled = object_pose(cube)
        if np.linalg.norm(settled[:3] - frame[:3]) > .004 or np.linalg.norm(cube.get_linear_velocity()) > .01:
            raise RuntimeError("Object did not settle on the table")
        return settled

    def execute_grasp(frame, waypoint_at):
        nonlocal gripper_opening
        log = dict(rows=[], joints=[], fingers=[], commands=[], objects=[], stages=[], desired=[])
        previous_stage = None
        error = None
        steps = round(args.duration * args.physics_hz)
        for tick in range(steps + 1):
            phase = tick / steps
            wanted, opening, stage = waypoint_at(phase)
            if not world.is_playing():
                error = "Timeline paused/stopped during grasp"
                break
            if stage != previous_stage:
                print(f"GRASP_STAGE {STAGES[stage]} phase={phase:.3f}", flush=True)
                # Do not squeeze before the measured TCP actually reaches the cube.
                if stage == 2 and not metrics(np.tile(wanted, (2, 1)), np.tile(actual_pose(), (2, 1)), .004, .04)["passed"]:
                    error = "TCP did not reach grasp pose before closing"
                    break
                previous_stage = stage
            gripper_opening = float(opening)
            goal.set_world_pose(wanted[:3], wanted[3:])
            if not follow(wanted):
                error = f"IK failed during {STAGES[stage]}"
                break
            step()
            if tick % stride == 0 or tick == steps:
                log["rows"].append([tick / args.physics_hz, *actual_pose()])
                log["joints"].append(robot.get_joint_positions(joint_indices=indices).copy())
                log["fingers"].append(robot.get_joint_positions(joint_indices=gripper_indices).copy())
                log["commands"].append(gripper_opening)
                log["objects"].append(object_pose(cube))
                log["stages"].append(stage)
                log["desired"].append(wanted.copy())
        if error:
            report = dict(passed=False, reason=error)
        else:
            rows = np.asarray(log["rows"])
            report = assess_grasp(rows[:, 0], rows[:, 1:], log["objects"], log["fingers"], frame,
                                  grasp_config, args.duration)
        return log, report

    def save_grasp_log(path, frame, log, report):
        if len(log["rows"]) < 3:
            return  # An initialization failure has a JSON report, not a training file.
        save_npz(path, **demo_arrays(log["rows"], log["joints"], frame), desired=log["desired"],
                 **grasp_recording_fields(log["rows"], log["fingers"], log["commands"], log["objects"],
                                          log["stages"], grasp_config, args.duration, report["passed"]))

    if args.mode == "grasp":
        bounds = grasp_position_bounds(args.grasp_spread, args.grasp_x_range, args.grasp_y_range)
        groups, total_successful = [], 0
        for group_index in range(args.grasp_groups):
            group_output = output if args.grasp_groups == 1 else output / f"group_{group_index + 1:03d}"
            group_seed = args.seed + group_index
            positions = sample_grasp_positions(args.demos, group_seed, bounds, args.grasp_sampling)
            planned_frames = np.tile(np.r_[cube_center, 1., 0., 0., 0.], (args.demos, 1))
            planned_frames[:, :2] = positions
            save_json(group_output / "positions.json", dict(seed=group_seed, sampling=args.grasp_sampling,
                      xy_bounds_m=bounds.tolist(), target_poses=planned_frames.tolist()))
            print(f"GRASP_GROUP {group_index + 1}/{args.grasp_groups} seed={group_seed} output={group_output}", flush=True)
            reports = []
            for index, planned_frame in enumerate(planned_frames):
                frame = planned_frame.copy()
                log = None
                try:
                    waypoints = [grasp_waypoint(frame, phase, grasp_config)[0] for phase in np.linspace(0, 1, 101)]
                    frame = prepare_grasp(frame, waypoints, grasp_config.open_width)
                    target.set_world_pose(frame[:3], frame[3:])
                    log, report = execute_grasp(frame, lambda phase: grasp_waypoint(frame, phase, grasp_config))
                except (ValueError, RuntimeError) as exc:
                    report = dict(passed=False, reason=str(exc))
                prefix = "demo" if report["passed"] else "failed"
                path = group_output / f"{prefix}_{index + 1:03d}.npz"
                if log is not None:
                    save_grasp_log(path, frame, log, report)
                report.update(attempt=index + 1, planned_target=planned_frame.tolist(), target=frame.tolist(),
                              recording=str(path) if path.exists() else None)
                save_json(group_output / f"attempt_{index + 1:03d}.json", report)
                reports.append(report)
                print(f"GRASP_RESULT {report}", flush=True)
                if not world.is_playing():
                    break
            successful = sum(report["passed"] for report in reports)
            total_successful += successful
            save_json(group_output / "summary.json", dict(task=TASK, requested=args.demos, succeeded=successful,
                      attempts=reports, seed=group_seed, sampling=args.grasp_sampling,
                      xy_bounds_m=bounds.tolist(), config=grasp_config.to_json()))
            groups.append(dict(group=group_index + 1, seed=group_seed, requested=args.demos,
                               succeeded=successful, summary=str(group_output / "summary.json")))
            if not world.is_playing():
                break
        requested = args.demos * args.grasp_groups
        if args.grasp_groups > 1:
            save_json(output / "summary.json", dict(task=TASK, requested=requested, succeeded=total_successful,
                      groups=groups, seed=args.seed, sampling=args.grasp_sampling,
                      xy_bounds_m=bounds.tolist(), config=grasp_config.to_json()))
        if total_successful != requested:
            raise GraspCollectionFailed(f"Only {total_successful}/{requested} grasps passed. See {output / 'summary.json'}")
        print(f"GRASP_COLLECTION_COMPLETE successful={total_successful} output={output}", flush=True)
        return

    if args.mode == "smoke":
        raised = initial.copy()
        raised[2] += .02
        settle(raised)
        report = metrics(np.tile(raised, (2, 1)), np.tile(actual_pose(), (2, 1)), .002, .02)
        save_json(output / "smoke.json", dict(articulation=articulation_path, joint_names=names,
                                              initial_tcp=initial.tolist(), raised_tcp=actual_pose().tolist(), **report))
        print(f"LIFT_CHECK {report}", flush=True)
        return

    if args.mode == "collect":
        # Small, reachable demonstrations around HOME; recorded samples still come from physics.
        for index, shift in enumerate(np.linspace(-.015, .015, args.demos)):
            reset()
            start = initial.copy()
            start[1] += shift
            settle(start)
            marker = start.copy()
            marker[:3] += [.025, 0, -.06]
            marker[3:] = [1, 0, 0, 0]
            target.set_world_pose(marker[:3], marker[3:])
            rows, joints = [], []
            steps = max(2, round(args.duration * args.physics_hz))
            for tick in range(steps + 1):
                phase = tick / steps
                blend = phase * phase * (3 - 2 * phase)
                wanted = start.copy()
                wanted[:3] += [.025 * blend, .008 * np.sin(np.pi * blend), -.025 * blend]
                goal.set_world_pose(wanted[:3], wanted[3:])
                if not follow(wanted):
                    raise RuntimeError(f"IK failed in demonstration {index + 1} at phase {phase}")
                step()
                if tick % stride == 0 or tick == steps:
                    rows.append([tick / args.physics_hz, *actual_pose()])
                    joints.append(robot.get_joint_positions(joint_indices=indices).copy())
            path = output / f"demo_{index + 1:03d}.npz"
            save_demo(path, rows, joints, marker)
            print(f"DEMO_SAVED {path} samples={len(rows)}", flush=True)
        return

    if args.mode == "replay":
        data = replay_data
        trajectory = poses(data["trajectory"])
        phase = np.asarray(data["phase"], dtype=float)
        if (trajectory.ndim != 2 or len(trajectory) < 2 or phase.shape != (len(trajectory),)
                or not np.isfinite(phase).all() or np.any(np.diff(phase) <= 0)
                or not np.isclose(phase[0], 0) or not np.isclose(phase[-1], 1)):
            raise ValueError("Invalid prediction trajectory or phase")
        if data["frame_names"].tolist() != ["target"] or data["frames"].shape != (1, 7):
            raise ValueError("Expected one task frame named target")
        frame = poses(data["frames"])[0]
        target.set_world_pose(frame[:3], frame[3:])
        rotations = Slerp(phase, Rotation.from_quat(trajectory[:, [4, 5, 6, 3]]))
        if physical_grasp:
            def replay_waypoint(s):
                position = [np.interp(s, phase, trajectory[:, axis]) for axis in range(3)]
                wanted = np.r_[position, rotations(s).as_quat()[[3, 0, 1, 2]]]
                opening = held_commands(data["gripper_phase"], data["gripper_commands"], s)
                stage = grasp_waypoint(frame, s, grasp_config)[2]
                return wanted, opening, stage

            # Denser preflight prevents sparse prediction files hiding a branch jump.
            waypoints = [replay_waypoint(s)[0] for s in np.linspace(0, 1, 101)]
            settled_frame = prepare_grasp(frame, waypoints, replay_waypoint(0.)[1])
            log, report = execute_grasp(settled_frame, replay_waypoint)
            save_grasp_log(output / "grasp_tracking.npz", settled_frame, log, report)
            save_json(output / "grasp_result.json", report)
            if len(log["rows"]) >= 3:
                rows = np.asarray(log["rows"])
                tracking = metrics(log["desired"], rows[:, 1:], args.position_tolerance, args.angle_tolerance)
                save_npz(output / "tracking.npz", time=rows[:, 0], desired=log["desired"], actual=rows[:, 1:])
                save_json(output / "metrics.json", tracking)
            else:
                tracking = dict(passed=False)
            print(f"GRASP_REPLAY_RESULT {report}", flush=True)
            if not report["passed"] or not tracking["passed"]:
                raise RuntimeError("Grasp replay failed grasp or tracking checks; see output reports")
            return
        settle(trajectory[0])
        desired, actual, times = [], [], []
        steps = max(2, round(args.duration * args.physics_hz))
        for tick in range(steps + 1):
            s = tick / steps
            position = [np.interp(s, phase, trajectory[:, axis]) for axis in range(3)]
            wanted = np.r_[position, rotations(s).as_quat()[[3, 0, 1, 2]]]
            goal.set_world_pose(wanted[:3], wanted[3:])
            if not follow(wanted):
                raise RuntimeError(f"Replay IK failed at phase {s}")
            step()
            if tick % stride == 0 or tick == steps:
                desired.append(wanted)
                actual.append(actual_pose())
                times.append(tick / args.physics_hz)
        report = metrics(desired, actual, args.position_tolerance, args.angle_tolerance)
        save_npz(output / "tracking.npz", time=times, desired=desired, actual=actual)
        save_json(output / "metrics.json", report)
        print(f"REPLAY_RESULT {report}", flush=True)
        if not report["passed"]:
            raise RuntimeError("Replay exceeded tracking tolerances; see metrics.json")
        return

    import omni.ui as ui
    import omni.appwindow
    import carb.input

    recording = None
    reset_pending = False
    from sim_bridge.gamepad import XboxJoystick, XboxControl, gamepad_pose, START
    import time

    use_xbox = args.control == "xbox"
    gamepad = XboxJoystick(args.gamepad_device) if use_xbox else None
    xbox_control = XboxControl(args.gamepad_deadzone)
    gamepad_enabled = ui.SimpleBoolModel(True)
    gamepad_status = ui.SimpleStringModel("Connect Xbox; release sticks, triggers and buttons to enable.")
    keyboard_enabled = ui.SimpleBoolModel(False)
    key_queue = []
    status_model = ui.SimpleStringModel("Xbox control: connect controller and return controls to neutral." if use_xbox
                                       else "Enable Keyboard control to move Nero with keys, or drag TCPGoal.")

    def set_gripper(opening):
        nonlocal gripper_opening
        if not world.is_playing() or reset_pending:
            return
        gripper_opening = float(np.clip(opening, 0., .1))

    def on_keyboard(event):
        if (not keyboard_enabled.get_value_as_bool() or not world.is_playing() or reset_pending):
            return True
        key = keyboard_key_name(event.input)
        if key == "ESCAPE":
            keyboard_enabled.set_value(False)
            key_queue.clear()
            return False
        if key in {"Z", "X"}:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                set_gripper(.1 if key == "Z" else 0.)
            return False
        if key not in {"I", "K", "J", "L", "U", "O", "KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6"}:
            return True
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            key_queue.append(key)
        return False

    keyboard_enabled.add_value_changed_fn(lambda model: key_queue.clear())

    def record_start():
        nonlocal recording
        if recording is not None:
            status_model.set_value("Already recording; Save or Discard first.")
            return
        recording = dict(rows=[], joints=[], target=object_pose(target), start=world.current_time, tick=0)
        status_model.set_value("RECORDING actual TCP. Keep green target fixed; move TCP with the selected controller.")

    def discard():
        nonlocal recording
        recording = None
        status_model.set_value("Recording discarded.")

    def save():
        nonlocal recording
        if recording is None:
            status_model.set_value("No recording to save.")
            return
        path = output / ("demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npz")
        try:
            save_demo(path, recording["rows"], recording["joints"], recording["target"])
        except (ValueError, OSError) as exc:
            status_model.set_value(str(exc))
            return
        recording = None
        status_model.set_value("SAVED " + str(path))
        print(f"DEMO_SAVED {path}", flush=True)

    def reset_request():
        nonlocal reset_pending
        if recording is not None:
            status_model.set_value("Save or Discard before reset.")
            return
        reset_pending = True
        key_queue.clear()

    window = ui.Window("Nero TP-KMP Collection", width=660, height=450)
    with window.frame:
        with ui.VStack(spacing=8):
            if use_xbox:
                with ui.HStack(height=24):
                    ui.CheckBox(model=gamepad_enabled, width=24)
                    ui.Label("Xbox control (Menu: enable / pause; return controls to neutral)")
                ui.Label("TCP axes: left stick XY; LT/RT +Z/-Z; right stick Rx/Ry; A/B +Rz/-Rz")
                ui.Label("D-pad: left/right close/open by 3%; up close; down 35% opening")
                ui.Label(f"Speed: {args.gamepad_speed * 1000:g} mm/s; {args.gamepad_angle_speed:g} deg/s")
                ui.StringField(model=gamepad_status, read_only=True, height=45, multiline=True)
            else:
                ui.Label("Select TCPGoal (red), W: translate / E: rotate. Target (green) is the task frame.")
                with ui.HStack(height=24):
                    ui.CheckBox(model=keyboard_enabled, width=24)
                    ui.Label("Keyboard control (Esc disables; disable before editing other panels)")
                ui.Label("World axes: I/K: +X/-X, J/L: +Y/-Y, U/O: +Z/-Z")
                ui.Label("Rotation: 1/2: +X/-X, 3/4: +Y/-Y, 5/6: +Z/-Z")
                ui.Label(f"Per press: {args.keyboard_step * 1000:g} mm / {args.keyboard_angle:g} deg")
            with ui.HStack(height=35):
                ui.Button("Open Gripper" if use_xbox else "Open Gripper (Z)", clicked_fn=lambda: set_gripper(.1))
                ui.Button("Close Gripper" if use_xbox else "Close Gripper (X)", clicked_fn=lambda: set_gripper(0.))
                ui.Button("Half Open", clicked_fn=lambda: set_gripper(.05))
            ui.Label("Gripper motion is not recorded or learned by TP-KMP in this version.")
            with ui.HStack(height=35):
                ui.Button("Record", clicked_fn=record_start)
                ui.Button("Save", clicked_fn=save)
                ui.Button("Discard", clicked_fn=discard)
                ui.Button("Reset Home", clicked_fn=reset_request)
            ui.StringField(model=status_model, read_only=True, height=65, multiline=True)
            ui.Label("Output: " + str(output), word_wrap=True)
    omni.usd.get_context().get_selection().set_selected_prim_paths(["/World/TCPGoal"], False)
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_subscription = None if use_xbox else input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard)
    tick = 0
    try:
        while app.is_running() and (not args.steps or tick < args.steps):
            if reset_pending:
                xbox_control.reset()
                reset()
                reset_pending = False
                status_model.set_value("Home restored. Target position unchanged.")
            connected = gamepad.poll() if gamepad is not None else False
            controls = gamepad.controls() if connected else {}
            if gamepad is not None and START in gamepad.pressed:
                gamepad_enabled.set_value(not gamepad_enabled.get_value_as_bool())
                xbox_control.reset()
            if not world.is_playing():
                xbox_control.reset()
                key_queue.clear()
                if recording is not None:
                    discard()
                    status_model.set_value("Timeline paused/stopped: recording discarded. Use this panel's Reset Home.")
                app.update()
                continue
            wanted = object_pose(goal)
            keyboard_move = bool(key_queue)
            for key in key_queue:
                wanted = keyboard_pose(wanted, key, args.keyboard_step, np.deg2rad(args.keyboard_angle))
            key_queue.clear()
            xbox_move = False
            if gamepad is not None:
                active, opening = xbox_control.update(connected, gamepad_enabled.get_value_as_bool(),
                                                       controls, gamepad.buttons, gripper_opening, time.monotonic())
                gamepad_status.set_value(gamepad.message + (" | active" if active else " | paused / waiting for neutral"))
                if active:
                    wanted = gamepad_pose(wanted, controls, gamepad.buttons, 1. / args.physics_hz,
                                          args.gamepad_speed, np.deg2rad(args.gamepad_angle_speed), args.gamepad_deadzone)
                    set_gripper(opening)
                    xbox_move = True
            success = follow(wanted)
            if (keyboard_move or xbox_move) and success:
                goal.set_world_pose(wanted[:3], wanted[3:])
            step()
            tick += 1
            if not success:
                if recording is not None:
                    discard()
                status_model.set_value("IK failed: controller move rejected; move TCPGoal closer. Recording discarded.")
            if recording is not None:
                if not np.allclose(object_pose(target), recording["target"], atol=1e-5):
                    discard()
                    status_model.set_value("Target moved during recording; discarded. Record again with target fixed.")
                    continue
                if recording["tick"] % stride == 0:
                    recording["rows"].append([world.current_time - recording["start"], *actual_pose()])
                    recording["joints"].append(robot.get_joint_positions(joint_indices=indices).copy())
                recording["tick"] += 1
    finally:
        if keyboard_subscription is not None:
            input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        if gamepad is not None:
            gamepad.close()
    if recording is not None:
        print("Unsaved recording discarded. Use Save before closing the window.", flush=True)


if __name__ == "__main__":
    args = arguments()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720,
                         "renderer": "RaytracedLighting", "anti_aliasing": 0})
    try:
        run(app, args)
    except GraspCollectionFailed as exc:
        print(f"GRASP_COLLECTION_FAILED: {exc}", flush=True)
        app.app.post_quit(2)
        raise SystemExit(2) from None
    except BaseException:
        import traceback
        traceback.print_exc()
        app.app.post_quit(1)
        raise
    finally:
        app.close()
