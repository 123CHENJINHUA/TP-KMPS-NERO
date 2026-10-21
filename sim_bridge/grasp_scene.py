"""Physical cube and table creation. Import only after SimulationApp starts."""

import numpy as np

from sim_bridge.grasp import FINGER_DAMPING, FINGER_MAX_FORCE, FINGER_STIFFNESS


def configure_finger_drives(stage, robot_roots=None):
    """Configure force drives for every selected robot, leaving arm drives unchanged."""
    from pxr import UsdPhysics

    finger_drives = []
    for prim in stage.Traverse():
        if robot_roots is not None and not any(
                str(prim.GetPath()).startswith(root + "/") for root in robot_roots):
            continue
        if prim.GetName() not in {"gripper_joint1", "gripper_joint2"} or not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        previous_type = drive.GetTypeAttr().Get() if drive else None
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(FINGER_STIFFNESS)
        drive.CreateDampingAttr().Set(FINGER_DAMPING)
        drive.CreateMaxForceAttr().Set(FINGER_MAX_FORCE)
        finger_drives.append(dict(path=str(prim.GetPath()), previous_type=previous_type,
                                  type=drive.GetTypeAttr().Get()))
    expected = 2 if robot_roots is None else 2 * len(robot_roots)
    if len(finger_drives) != expected:
        raise RuntimeError(f"Expected {expected} Nero finger drives, found {len(finger_drives)}")
    return finger_drives


def create_grasp_scene(world, config, center):
    import omni.usd
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from sim_bridge.objects import load_object_profile
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    # Position targets alone do not select force vs. acceleration drive semantics.
    # Explicit force drives make the finger gains meaningful in N/m, not 1/s^2.
    world.get_physics_context().set_solve_articulation_contact_last(True)
    material = PhysicsMaterial("/World/GraspMaterial", static_friction=1.2, dynamic_friction=1., restitution=0.)
    profile = load_object_profile(config.object_name)
    table_size = [.18, .18] if profile is None else profile["table_size_xy_m"]
    if profile is not None and not np.allclose(
            [config.cube_size, config.support_height], [profile["grip_width_m"], profile["origin_height_m"]]):
        raise ValueError("Object grasp profile differs from the recorded skill geometry")
    table_top = center[2] - config.support_height
    world.scene.add(FixedCuboid(
        "/World/GraspTable", name="grasp_table", size=1.,
        scale=np.array([*table_size, .02]), position=np.array([*center[:2], table_top - .01]),
        color=np.array([.45, .45, .5]), physics_material=material,
    ))
    if profile is None:
        cube = world.scene.add(DynamicCuboid(
            "/World/GraspCube", name="grasp_cube", size=config.cube_size,
            position=np.asarray(center), mass=config.cube_mass,
            color=np.array([1., .55, .05]), physics_material=material,
        ))
    else:
        add_reference_to_stage(profile["asset_path"], "/World/GraspObject")
        cube = world.scene.add(SingleRigidPrim(
            "/World/GraspObject", name="grasp_object", position=np.asarray(center), mass=config.cube_mass,
        ))
    stage = omni.usd.get_context().get_stage()
    finger_colliders = 0
    finger_drives = configure_finger_drives(stage)
    for prim in stage.Traverse():
        finger = any(part in str(prim.GetPath()).split("/") for part in ("gripper_link1", "gripper_link2"))
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not (finger or (str(prim.GetPath()) == cube.prim_path or str(prim.GetPath()).startswith(cube.prim_path + "/"))):
            continue
        # Small objects require smaller contact offsets than the default 20 mm.
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr().Set(.001)
        contact.CreateRestOffsetAttr().Set(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        if finger:
            finger_colliders += 1
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
    if finger_colliders < 2:
        raise RuntimeError("Could not find both Nero finger colliders in the imported URDF")
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prim)
    body.CreateSolverPositionIterationCountAttr().Set(64)
    body.CreateSolverVelocityIterationCountAttr().Set(8)
    settings = dict(object_name=config.object_name, object_mass_kg=config.cube_mass,
                    object_profile=profile, finger_drives=finger_drives, finger_colliders=finger_colliders,
                    stiffness_N_per_m=FINGER_STIFFNESS, damping_Ns_per_m=FINGER_DAMPING,
                    max_force_N_per_finger=FINGER_MAX_FORCE, static_friction=1.2, dynamic_friction=1.,
                    contact_offset_m=.001, solve_articulation_contact_last=True,
                    solver_position_iterations=64, solver_velocity_iterations=8)
    print(f"GRASP_PHYSICS {settings}", flush=True)
    return cube, settings


def configure_dual_grasp_contacts(world, robot_roots):
    """Apply the tested grasp contact settings to the selected dual-arm fingers."""
    import omni.usd
    from isaacsim.core.api.materials import PhysicsMaterial
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    drives = configure_finger_drives(stage, robot_roots)
    world.get_physics_context().set_solve_articulation_contact_last(True)
    material = PhysicsMaterial("/World/DualGraspMaterial", static_friction=1.2,
                               dynamic_friction=1., restitution=0.)
    # URDF collision references are instanceable. Make only the finger collision
    # containers editable so contact/material overrides apply to the actual arms.
    for root in robot_roots:
        for finger in ("gripper_link1", "gripper_link2"):
            collisions = stage.GetPrimAtPath(f"{root}/{finger}/collisions")
            if collisions.IsValid() and collisions.IsInstance():
                collisions.SetInstanceable(False)
    counts = {root: 0 for root in robot_roots}
    for prim in stage.Traverse():
        root = next((root for root in robot_roots
                     if str(prim.GetPath()).startswith(root + "/")), None)
        if (root is None or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not any(part in str(prim.GetPath()).split("/")
                           for part in ("gripper_link1", "gripper_link2"))):
            continue
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr().Set(.001)
        contact.CreateRestOffsetAttr().Set(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics")
        if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
        counts[root] += 1
    if any(count < 2 for count in counts.values()):
        raise RuntimeError(f"Could not find both finger colliders on each arm: {counts}")
    print(f"DUAL_GRASP_PHYSICS finger_drives={len(drives)} colliders={counts} "
          f"max_force_N={FINGER_MAX_FORCE} contact_offset_m=0.001", flush=True)
