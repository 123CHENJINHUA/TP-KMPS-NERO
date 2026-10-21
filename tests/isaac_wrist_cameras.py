"""Isaac-only integration: configured viewports, rendered RGB, and physical link following."""
import sys

from sim_bridge.app import arguments
extra_cameras = '--extra-cameras' in sys.argv
sys.argv = [sys.argv[0], '--mode', 'dual', '--headless', '--steps', '1',
            '--scene-usd', 'assets/scenes/0928_grasp.usd']
if extra_cameras:
    sys.argv += ['--wrist-camera-config', 'assets/cameras/wrist_cameras_extra_example.json']
args = arguments()
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, UsdGeom, Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    from sim_bridge import dual
    from sim_bridge.wrist_cameras import create_wrist_cameras, load_camera_config

    class Finished(Exception):
        pass

    class CheckingJoystick:
        def __init__(self, *unused):
            world = World.instance()
            stage = omni.usd.get_context().get_stage()
            config = load_camera_config(args.wrist_camera_config)
            roots = ['/World/nero', '/World/nero2']
            cameras = create_wrist_cameras(stage, roots, config, show_views=True)
            rgbs, products, initial = [], [], []
            links = {}
            robots = [world.scene.get_object('nero1'), world.scene.get_object('nero2')]
            for index, camera in enumerate(cameras):
                assert tuple(camera['window'].viewport_api.resolution) == tuple(camera['definition']['resolution'])
                product = rep.create.render_product(camera['path'], tuple(camera['definition']['resolution']))
                rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                rgb.attach([product])
                products.append(product)
                rgbs.append(rgb)
                if camera['definition']['arm'] is not None and camera['parent'] not in links:
                    link = SingleRigidPrim(camera['parent'], name=f'check_tool_{index}',
                                           reset_xform_properties=False)
                    link.initialize()
                    links[camera['parent']] = link
                initial.append(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                    stage.GetPrimAtPath(camera['path'])).ExtractTranslation())
            # Exercise an actual camera orientation edit, independent of the manipulator.
            rotated_camera = UsdGeom.Xformable(stage.GetPrimAtPath(cameras[0]['path']))
            rotation = rotated_camera.GetRotateXYZOp().Get()
            rotated_camera.GetRotateXYZOp().Set(Gf.Vec3f(rotation[0], rotation[1] + 17., rotation[2]))
            for index, robot in enumerate(robots):
                target = robot.get_joint_positions().copy()
                target[robot.dof_names.index('joint1')] += .25 if index == 0 else -.25
                robot.apply_action(ArticulationAction(joint_positions=target))
            for _ in range(240):
                for robot in robots:
                    robot.set_joint_efforts(robot._articulation_view.get_generalized_gravity_forces()[0])
                world.step(render=True)
            for camera, start, rgb in zip(cameras, initial, rgbs):
                definition = camera['definition']
                width, height = definition['resolution']
                data = rgb.get_data()
                assert data.shape[:2] == (height, width), data.shape
                assert np.std(data[:, :, :3]) > 1., 'Camera rendered a blank frame'
                if definition['arm'] is not None:
                    position, quaternion = links[camera['parent']].get_world_pose()
                    expected = position + Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply(
                        definition['translation_m'])
                else:
                    expected = np.array(start)
                actual = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                    stage.GetPrimAtPath(camera['path']))
                length = definition['visualization']['axis_length_m']
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                for suffix, local in [('/Body/Housing', (0., 0., .014)),
                                      ('/Axes/X_Positive/Tip', (length * .9, 0., 0.)),
                                      ('/Axes/OpticalDirection_MinusZ/Tip',
                                       (max(.06, length * 1.2), 0., -length * .9))]:
                    marker = stage.GetPrimAtPath(camera['visualization'] + suffix)
                    expected_marker = actual.Transform(Gf.Vec3d(*local))
                    actual_marker = cache.GetLocalToWorldTransform(marker).ExtractTranslation()
                    assert np.linalg.norm(np.array(expected_marker) - np.array(actual_marker)) < 1e-6
                for prim in Usd.PrimRange(stage.GetPrimAtPath(camera['visualization'])):
                    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI), prim.GetPath()
                    assert not prim.HasAPI(UsdPhysics.CollisionAPI), prim.GetPath()
                print(f"CAMERA_VISUALIZATION_CHECK path={camera['path']} following=True physics=False", flush=True)
                camera_position = np.array(actual.ExtractTranslation())
                error = np.linalg.norm(expected - camera_position)
                movement = np.linalg.norm(camera_position - np.array(start))
                assert error < .0005, (camera['path'], error)
                if definition['arm'] is not None:
                    assert movement > .002, (camera['path'], movement)
                else:
                    assert movement < .000001, (camera['path'], movement)
                print(f"WRIST_CAMERA_CHECK path={camera['path']} rgb_shape={data.shape} "
                      f'follow_error_m={error:.8f} movement_m={movement:.5f}', flush=True)
                rgb.detach()
            for product in products:
                product.destroy()
            for camera in cameras:
                camera['window'].destroy()
            print('WRIST_CAMERA_INTEGRATION_PASS', flush=True)
            raise Finished()

    dual.XboxJoystick = CheckingJoystick
    try:
        dual.run_dual(app, args)
    except Finished:
        pass
except BaseException:
    import traceback
    traceback.print_exc()
    app.app.post_quit(1)
    raise
finally:
    app.close()
