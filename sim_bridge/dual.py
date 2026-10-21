"""Two fixed-base Nero arms controlled by one Xbox joystick."""

import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from sim_bridge.app import HOME, JOINTS, ROOT, apply_ik_target, gripper_joint_targets
from sim_bridge.gamepad import START, Y, XboxControl, XboxJoystick, gamepad_pose


def mount_poses(spacing, height):
    """Rotate both bases 90 degrees from the former inward-facing poses.

    Left: +90° -> 0° (clockwise); right: -90° -> 0° (counterclockwise).
    Quaternions use Isaac's wxyz order.
    """
    return [
        (np.array([0., -spacing / 2, height]), np.array([1., 0., 0., 0.])),
        (np.array([0., spacing / 2, height]), np.array([1., 0., 0., 0.])),
    ]


def selected_after_y(current, pressed, connected, enabled, playing):
    """Switch one arm per physical Y press, only while control is available."""
    return 1 - current if connected and enabled and playing and Y in pressed else current


def urdf_joint_limits(urdf):
    """Return the seven arm joint limits in JOINTS order."""
    root = ET.parse(urdf).getroot()
    by_name = {joint.get("name"): joint.find("limit") for joint in root.findall("joint")}
    return np.asarray([[float(by_name[name].get("lower")), float(by_name[name].get("upper"))]
                       for name in JOINTS], dtype=float)


def clip_teach_joints(joints, limits):
    values = np.asarray(joints, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Teaching arm must provide seven finite joint angles")
    clipped = np.clip(values, limits[:, 0], limits[:, 1])
    limited = tuple(JOINTS[i] for i in range(7) if abs(values[i] - clipped[i]) > 1e-5)
    return clipped, limited


def teach_gripper_opening(raw_width, arm_index):
    """Map either teaching gripper directly to the same simulated opening."""
    return float(np.clip(raw_width, 0., .1))


def gripper_command_step(previous, opening, physics_hz, live_teaching):
    """Use the newest teaching width without the Xbox/grasp command ramp."""
    target = gripper_joint_targets(opening)
    if live_teaching:
        return target
    return previous + np.clip(target - previous,
                              -.02 / physics_hz, .02 / physics_hz)


def physics_view_ready(view):
    """Disposed Isaac 5.1 views can have no _physics_view attribute at all."""
    if view is None or not view.is_valid():
        return False
    try:
        return bool(view.is_physics_handle_valid())
    except AttributeError:
        return False


def robot_handles_ready(robot):
    if not robot.is_valid():
        return False
    try:
        return bool(robot.handles_initialized)
    except AttributeError:
        return False


def run_dual(app, args):
    import omni.kit.commands
    import omni.usd
    import omni.ui as ui
    from isaacsim.asset.importer.urdf import _urdf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCuboid
    from isaacsim.core.prims import Articulation, SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, Usd, UsdGeom, UsdLux

    enable_extension("isaacsim.asset.importer.urdf")
    enable_extension("isaacsim.robot_motion.motion_generation")
    app.update()
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

    world = World(stage_units_in_meters=1., physics_dt=1 / args.physics_hz,
                  rendering_dt=1 / 60, device="cpu")
    world.scene.add(GroundPlane("/World/Ground", z_position=-.03, size=4.))
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        raise RuntimeError("Simulation stage has no /World prim")
    stage.SetDefaultPrim(world_prim)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr().Set(700.)
    if args.scene_usd:
        source = Usd.Stage.Open(args.scene_usd)
        if source is None or not source.GetDefaultPrim().IsValid():
            raise RuntimeError("Saved scene USD must have a default prim; create a /World root in Isaac Sim")
        generated = {"Ground", "Light", "Stand1", "Stand2", "StandCrossbar",
                     "nero", "nero2", "TCPGoal1", "TCPGoal2", "DualGraspMaterial"}
        duplicated = sorted(generated & {child.GetName() for child in source.GetDefaultPrim().GetChildren()})
        if duplicated:
            raise RuntimeError("Saved environment contains generated simulation prims: "
                               + ", ".join(duplicated)
                               + "; save only extra scenery and objects in the USD")
        scene_prim = add_reference_to_stage(args.scene_usd, "/World/UserScene")
        if not scene_prim.IsValid():
            raise RuntimeError(f"Could not reference saved scene: {args.scene_usd}")
        print(f"USER_SCENE_LOADED path={args.scene_usd} prim=/World/UserScene", flush=True)
    if not stage.GetPrimAtPath("/World/UserScene").IsValid():
        stage.DefinePrim("/World/UserScene", "Xform")
    if not stage.GetPrimAtPath("/World/UserScene/Environment").IsValid():
        stage.DefinePrim("/World/UserScene/Environment", "Xform")
    mounts = mount_poses(args.dual_spacing, args.dual_mount_height)
    for index, (position, _) in enumerate(mounts, 1):
        world.scene.add(FixedCuboid(
            prim_path=f"/World/Stand{index}", name=f"stand{index}",
            position=np.array([position[0], position[1], args.dual_mount_height / 2]),
            scale=np.array([.24, .24, args.dual_mount_height]),
            color=np.array([.22, .27, .32]),
        ))
    world.scene.add(FixedCuboid(
        prim_path="/World/StandCrossbar", name="stand_crossbar",
        position=np.array([0., 0., args.dual_mount_height - .065]),
        scale=np.array([.10, args.dual_spacing + .1, .10]),
        color=np.array([.22, .27, .32]),
    ))

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
    urdfs = (ROOT / "assets/nero/nero.urdf", ROOT / "assets/nero/nero_dual_2.urdf")
    robots = []
    for index, (urdf, (position, orientation)) in enumerate(zip(urdfs, mounts), 1):
        if not urdf.is_file():
            raise FileNotFoundError(f"Missing robot model: {urdf}; regenerate assets")
        stage.SetDefaultPrim(world_prim)
        status, articulation_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(urdf), import_config=config,
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError(f"Nero {index} URDF import failed")
        # The URDF importer creates a world-rooted robot. Move the entire fixed-base
        # hierarchy before physics initialization, including its collision meshes.
        root_path = articulation_path.rsplit("/", 1)[0]
        if any(existing["root"] == root_path or root_path.startswith(existing["root"] + "/")
               or existing["root"].startswith(root_path + "/") for existing in robots):
            raise RuntimeError(f"Nero {index} was nested inside another arm: {root_path}")
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Imported Nero {index} has no root prim at {root_path}")
        transform = UsdGeom.Xformable(root_prim)
        ops = {op.GetOpType(): op for op in transform.GetOrderedXformOps()}
        translate = ops.get(UsdGeom.XformOp.TypeTranslate) or transform.AddTranslateOp()
        orient = ops.get(UsdGeom.XformOp.TypeOrient) or transform.AddOrientOp()
        translate.Set(Gf.Vec3d(*position) if str(translate.GetAttr().GetTypeName()) == "double3"
                      else Gf.Vec3f(*position))
        quaternion = (Gf.Quatf(*orientation) if str(orient.GetAttr().GetTypeName()) == "quatf"
                      else Gf.Quatd(*orientation))
        orient.Set(quaternion)
        robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name=f"nero{index}"))
        goal = world.scene.add(VisualCuboid(
            f"/World/TCPGoal{index}", name=f"tcp_goal{index}", size=.022,
            color=np.array([.9, .18, .12] if index == 1 else [.12, .45, .95]),
        ))
        robots.append(dict(robot=robot, root=root_path, path=articulation_path,
                           position=position, orientation=orientation, goal=goal,
                           opening=.05, command=gripper_joint_targets(.05)))
    if not args.no_wrist_cameras:
        from sim_bridge.wrist_cameras import create_wrist_cameras, load_camera_config
        camera_config = load_camera_config(args.wrist_camera_config)
        cameras = create_wrist_cameras(
            stage, [arm["root"] for arm in robots], camera_config,
            show_views=not args.headless and not args.no_camera_views)
        for arm, camera in zip(robots, cameras):
            arm["wrist_camera"] = camera
    set_camera_view(eye=np.array([1.8, 1.7, 1.7]), target=np.array([0., 0., .85]))
    from sim_bridge.grasp_scene import configure_dual_grasp_contacts
    from sim_bridge.grasp import FINGER_STIFFNESS, FINGER_DAMPING, FINGER_MAX_FORCE
    configure_dual_grasp_contacts(world, [arm["root"] for arm in robots])

    def set_arm_gains(arm):
        robot = arm["robot"]
        controller = robot.get_articulation_controller()
        kps, kds = np.full(robot.num_dof, args.kp), np.full(robot.num_dof, args.kd)
        kps[arm["gripper_indices"]] = FINGER_STIFFNESS
        kds[arm["gripper_indices"]] = FINGER_DAMPING
        controller.set_gains(kps=kps, kds=kds)
        controller.set_max_efforts(np.full(2, FINGER_MAX_FORCE),
                                   joint_indices=arm["gripper_indices"])
        robot.set_solver_position_iteration_count(32)

    world.reset()
    for index, arm in enumerate(robots, 1):
        robot = arm["robot"]
        dynamics = Articulation(prim_paths_expr=arm["path"], name=f"nero{index}_dynamics")
        dynamics.initialize()
        arm["dynamics"] = dynamics
        names = robot.dof_names
        if not all(name in names for name in JOINTS + ["gripper_joint1", "gripper_joint2"]):
            raise RuntimeError(f"Nero {index} articulation joints missing: {names}")
        arm["indices"] = np.array([names.index(name) for name in JOINTS])
        arm["gripper_indices"] = np.array([names.index(name) for name in ("gripper_joint1", "gripper_joint2")])
        home = np.zeros(robot.num_dof)
        home[arm["indices"]] = HOME
        home[arm["gripper_indices"]] = arm["command"]
        arm["home"] = home
        robot.set_joints_default_state(positions=home)
        set_arm_gains(arm)
        solver = LulaKinematicsSolver(str(ROOT / "assets/lula.yaml"), str(urdfs[index - 1]))
        solver.set_robot_base_pose(arm["position"], arm["orientation"])
        arm["solver"] = solver
        arm["kinematics"] = ArticulationKinematicsSolver(robot, solver, "gripper_base")
        robot.set_joint_positions(home)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        robot.apply_action(ArticulationAction(joint_positions=home))

    def actual_pose(arm):
        position, matrix = arm["kinematics"].compute_end_effector_pose()
        quaternion = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
        return np.r_[position, quaternion]

    def goal_pose(arm):
        position, quaternion = arm["goal"].get_world_pose()
        return np.r_[position, quaternion]

    def ensure_physics_ready():
        # Run before any controller/IK/gravity access, not just before world.step.
        if not world.is_playing():
            return False
        if all(robot_handles_ready(arm["robot"]) and physics_view_ready(arm["dynamics"])
               for arm in robots):
            return True
        for index, arm in enumerate(robots, 1):
            if not stage.GetPrimAtPath(arm["path"]).IsValid():
                raise RuntimeError(f"Nero {index} was removed from the stage; restart the simulation")
            if not arm["robot"].is_valid():
                name = arm["robot"].name
                world.scene.remove_object(name, registry_only=True)
                arm["robot"] = world.scene.add(SingleArticulation(
                    prim_path=arm["path"], name=name, reset_xform_properties=False))
                arm["robot"].set_joints_default_state(positions=arm["home"])
                arm["kinematics"] = ArticulationKinematicsSolver(
                    arm["robot"], arm["solver"], "gripper_base")
        world.initialize_physics()
        for index, arm in enumerate(robots, 1):
            robot = arm["robot"]
            robot.initialize()
            dynamics = arm["dynamics"]
            if not physics_view_ready(dynamics):
                # initialize() itself raises on a disposed view in Isaac 5.1.
                # Replace it using the cached USD path instead of repairing internals.
                if not dynamics.is_valid() or not hasattr(dynamics, "_physics_view"):
                    dynamics = Articulation(prim_paths_expr=arm["path"],
                                            name=f"nero{index}_dynamics",
                                            reset_xform_properties=False)
                    arm["dynamics"] = dynamics
                dynamics.initialize()
            if not robot_handles_ready(robot) or not physics_view_ready(dynamics):
                raise RuntimeError(f"Nero {index} physics handles could not be restored")
            set_arm_gains(arm)
            positions = robot.get_joint_positions()
            if positions is None or not np.isfinite(positions).all():
                raise RuntimeError(f"Nero {index} has invalid joint state after scene update")
            # Hold the actual pose until fresh input arrives. Do not teleport home.
            robot.apply_action(ArticulationAction(joint_positions=positions.copy()))
            if "teach_target" in arm:
                arm["teach_target"] = positions[arm["indices"]].copy()
            pose = actual_pose(arm)
            arm["goal"].set_world_pose(pose[:3], pose[3:])
        print("DUAL_PHYSICS_RESTORED arms=2", flush=True)
        return True

    def physics_step():
        if not ensure_physics_ready():
            return False
        for arm in robots:
            robot = arm["robot"]
            gravity = arm["dynamics"].get_generalized_gravity_forces()[0]
            if not np.isfinite(gravity).all():
                raise RuntimeError("Invalid gravity compensation from PhysX")
            robot.set_joint_efforts(gravity)
            arm["command"] = gripper_command_step(
                arm["command"], arm["opening"], args.physics_hz,
                live_teaching=args.mode == "teach-sync",
            )
            robot.apply_action(ArticulationAction(
                joint_positions=arm["command"].copy(), joint_indices=arm["gripper_indices"],
            ))
        world.step(render=not args.headless)
        # Kit processes scene/timeline events during the step. Restore before
        # the teaching status UI or FK reads any handles in this same iteration.
        return ensure_physics_ready()

    def reset_both():
        if not ensure_physics_ready():
            return False
        for arm in robots:
            robot = arm["robot"]
            arm["opening"] = .05
            arm["command"] = gripper_joint_targets(.05)
            robot.set_joint_positions(arm["home"])
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.apply_action(ArticulationAction(joint_positions=arm["home"]))
        for _ in range(args.physics_hz):
            if not physics_step():
                return False
        for arm in robots:
            pose = actual_pose(arm)
            arm["goal"].set_world_pose(pose[:3], pose[3:])
        return True

    reset_both()
    for index, arm in enumerate(robots, 1):
        pose = actual_pose(arm)
        pose[:3] += Rotation.from_quat(pose[[4, 5, 6, 3]]).apply([0., .001, 0.])
        action, reachable = arm["kinematics"].compute_inverse_kinematics(
            pose[:3], pose[3:], position_tolerance=.0005, orientation_tolerance=.005,
        )
        if not reachable or not np.isfinite(action.joint_positions).all():
            raise RuntimeError(f"Nero {index} mounted IK preflight failed")
        print(f"DUAL_READY arm={index} root={arm['root']} articulation={arm['path']} "
              f"base={arm['position'].tolist()} tcp={actual_pose(arm).tolist()}", flush=True)

    from sim_bridge.scene_edit import next_scene_snapshot_path, save_user_scene
    from sim_bridge.ui_layout import build_layout_controls

    def save_environment(model):
        try:
            destination = save_user_scene(stage, next_scene_snapshot_path())
        except Exception as exc:
            model.set_value(f"Save failed: {exc}")
            print(f"USER_SCENE_SAVE_FAILED {exc}", flush=True)
        else:
            model.set_value(f"Saved environment: {destination}")
            print(f"USER_SCENE_SAVED path={destination}", flush=True)

    if args.mode == "teach-sync":
        from sim_bridge.teach_can import TeachCanReader

        channels = (args.teach_left_can, args.teach_right_can)
        limits = [urdf_joint_limits(urdf) for urdf in urdfs]
        reader = TeachCanReader(channels, stale_after=args.teach_stale)
        try:
            started = time.monotonic()
            snapshots = None
            while app.is_running() and time.monotonic() - started < args.teach_start_timeout:
                reader.poll()
                snapshots = [reader.snapshot(ch) for ch in channels]
                if all(state["joints"] is not None and state["opening"] is not None
                       for state in snapshots):
                    break
                app.update()
                time.sleep(.005)
            else:
                snapshots = snapshots or [{"joints": None, "opening": None} for _ in channels]
                detail = [f"{ch}: joints={state['joints'] is not None}, "
                          f"width_gripper={state['opening'] is not None}"
                          for ch, state in zip(channels, snapshots)]
                raise RuntimeError("Teaching CAN startup timed out: " + "; ".join(detail))

            while app.is_running() and not world.is_playing():
                reader.poll()
                app.update()
                time.sleep(.005)
            if not app.is_running():
                return
            ensure_physics_ready()
            for index, (arm, sample, bounds) in enumerate(zip(robots, snapshots, limits), 1):
                joints, limited = clip_teach_joints(sample["joints"], bounds)
                opening = teach_gripper_opening(sample["opening"], index - 1)
                initial = arm["home"].copy()
                initial[arm["indices"]] = joints
                initial[arm["gripper_indices"]] = gripper_joint_targets(opening)
                arm["robot"].set_joint_positions(initial)
                arm["robot"].set_joint_velocities(np.zeros(arm["robot"].num_dof))
                arm["robot"].apply_action(ArticulationAction(joint_positions=initial))
                arm["teach_target"] = joints
                arm["opening"] = opening
                arm["command"] = gripper_joint_targets(opening)
                print(f"TEACH_SYNC_READY arm={index} channel={channels[index - 1]} "
                      f"joints={joints.tolist()} gripper_width={opening:.4f} "
                      f"clamped={list(limited)}", flush=True)
            sync_enabled = ui.SimpleBoolModel(True)
            status_model = ui.SimpleStringModel("Receiving both teaching arms.")
            scene_model = ui.SimpleStringModel("Add objects under /World/UserScene/Environment.")
            window = ui.Window("Dual Nero Teaching Sync", width=650, height=345)
            with window.frame:
                with ui.VStack(spacing=8):
                    ui.Label(f"{channels[0]} / {channels[1]} teaching arms -> simulated Nero 1 / Nero 2")
                    with ui.HStack(height=25):
                        ui.CheckBox(model=sync_enabled, width=25)
                        ui.Label("Sync enabled; uncheck to hold both simulated arms")
                    ui.StringField(model=status_model, read_only=True, height=60, multiline=True)
                    ui.Button("Save Environment USD", clicked_fn=lambda: save_environment(scene_model),
                              height=30)
                    ui.StringField(model=scene_model, read_only=True, height=45, multiline=True)
                    layout_status = build_layout_controls()
            tick = 0
            last_ui = 0.
            last_limits = [(), ()]
            while app.is_running() and (not args.steps or tick < args.steps):
                reader.poll()
                if not world.is_playing():
                    app.update()
                    continue
                if not ensure_physics_ready():
                    app.update()
                    continue
                statuses = []
                for index, (arm, channel, bounds) in enumerate(zip(robots, channels, limits)):
                    sample = reader.snapshot(channel)
                    if not sync_enabled.get_value_as_bool():
                        statuses.append(f"{channel}: paused")
                        continue
                    if sample["joints"] is None:
                        statuses.append(f"{channel}: stale joints; holding")
                        continue
                    desired, limited = clip_teach_joints(sample["joints"], bounds)
                    if limited != last_limits[index]:
                        if limited:
                            print(f"TEACH_SYNC_LIMIT arm={index + 1} joints={list(limited)}", flush=True)
                        last_limits[index] = limited
                    previous = arm["teach_target"]
                    step = args.joint_speed / args.physics_hz
                    target = previous + np.clip(desired - previous, -step, step)
                    arm["teach_target"] = target
                    arm["robot"].apply_action(ArticulationAction(
                        joint_positions=target.copy(), joint_indices=arm["indices"]))
                    opening = sample["opening"]
                    if opening is not None:
                        arm["opening"] = teach_gripper_opening(opening, index)
                    status = f"{channel}: live"
                    if opening is None:
                        status += ", gripper stale/angle mode"
                    if limited:
                        status += ", limit " + ",".join(limited)
                    statuses.append(status)
                if not physics_step():
                    continue
                tick += 1
                if tick % 12 == 0:
                    for arm in robots:
                        pose = actual_pose(arm)
                        arm["goal"].set_world_pose(pose[:3], pose[3:])
                if time.monotonic() - last_ui >= .2:
                    lines = []
                    for arm, status in zip(robots, statuses):
                        fingers = arm["robot"].get_joint_positions(
                            joint_indices=arm["gripper_indices"])
                        actual_width = float(fingers[0] - fingers[1])
                        lines.append(f"{status}; gripper actual/target "
                                     f"{actual_width * 1000:.1f}/{arm['opening'] * 1000:.1f} mm")
                    status_model.set_value("\n".join(lines))
                    last_ui = time.monotonic()
        finally:
            reader.close()
        return

    gamepad = XboxJoystick(args.gamepad_device)
    control = XboxControl(args.gamepad_deadzone)
    enabled = ui.SimpleBoolModel(True)
    selected = 0
    reset_pending = False
    selected_model = ui.SimpleStringModel("Controlling Nero 1 (red)")
    status_model = ui.SimpleStringModel("Connect Xbox; release controls to enable.")
    scene_model = ui.SimpleStringModel("Add objects under /World/UserScene/Environment.")

    def request_reset():
        nonlocal reset_pending
        reset_pending = True
        control.reset()

    window = ui.Window("Dual Nero Xbox Control", width=600, height=455)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Two fixed-base Nero arms; Y switches active arm.")
            ui.StringField(model=selected_model, read_only=True, height=25)
            with ui.HStack(height=25):
                ui.CheckBox(model=enabled, width=25)
                ui.Label("Xbox enabled; Menu toggles, release controls after switching")
            ui.Label("Left stick: TCP XY; LT/RT: TCP Z; right stick/A/B: TCP rotation")
            ui.Label("D-pad: gripper; each arm retains its own goal and opening")
            ui.StringField(model=status_model, read_only=True, height=50, multiline=True)
            ui.Button("Reset Both Arms", clicked_fn=request_reset, height=35)
            ui.Button("Save Environment USD", clicked_fn=lambda: save_environment(scene_model),
                      height=30)
            ui.StringField(model=scene_model, read_only=True, height=45, multiline=True)
            layout_status = build_layout_controls()
    try:
        tick = 0
        while app.is_running() and (not args.steps or tick < args.steps):
            connected = gamepad.poll()
            controls = gamepad.controls() if connected else {}
            if connected and START in gamepad.pressed:
                enabled.set_value(not enabled.get_value_as_bool())
                control.reset()
            next_selected = selected_after_y(selected, gamepad.pressed, connected,
                                             enabled.get_value_as_bool(), world.is_playing())
            if next_selected != selected:
                selected = next_selected
                control.reset()
                selected_model.set_value(f"Controlling Nero {selected + 1} ({'red' if selected == 0 else 'blue'})")
                print(f"DUAL_SELECTED arm={selected + 1}", flush=True)
            if not world.is_playing():
                control.reset()
                app.update()
                continue
            if not ensure_physics_ready():
                app.update()
                continue
            if reset_pending:
                if not reset_both():
                    continue
                reset_pending = False
                status_model.set_value("Both arms reset to home.")
            arm = robots[selected]
            active, opening = control.update(connected, enabled.get_value_as_bool(),
                                             controls, gamepad.buttons, arm["opening"], time.monotonic())
            status_model.set_value(gamepad.message + (" | active" if active else " | paused / waiting for neutral"))
            if active:
                wanted = gamepad_pose(goal_pose(arm), controls, gamepad.buttons,
                                      1. / args.physics_hz, args.gamepad_speed,
                                      np.deg2rad(args.gamepad_angle_speed), args.gamepad_deadzone)
                action, success = arm["kinematics"].compute_inverse_kinematics(
                    wanted[:3], wanted[3:], position_tolerance=.0005, orientation_tolerance=.005,
                )
                if success and np.isfinite(action.joint_positions).all():
                    apply_ik_target(arm["robot"], action, args.joint_speed / args.physics_hz)
                    arm["goal"].set_world_pose(wanted[:3], wanted[3:])
                    arm["opening"] = opening
                else:
                    status_model.set_value(f"Nero {selected + 1}: IK rejected this move")
            physics_step()
            tick += 1
    finally:
        gamepad.close()
