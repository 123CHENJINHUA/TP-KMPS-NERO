"""Isaac-only regression: camera edits, timeline restarts, and disposed views.

Run with Isaac Sim's python.sh and the project on PYTHONPATH.
This test runs simulated arms only and never opens CAN or Xbox hardware.
"""
import sys

from sim_bridge.app import arguments

test_mode = 'teach-sync' if '--teach-sync' in sys.argv else 'dual'
sys.argv = [sys.argv[0], '--mode', test_mode, '--headless', '--steps', '240',
            '--scene-usd', 'assets/scenes/0928_grasp.usd']
args = arguments()
from isaacsim import SimulationApp

app = SimulationApp({'headless': True})
try:
    import numpy as np
    import omni.kit.commands
    import omni.usd
    import isaacsim.core.api as core_api
    import isaacsim.core.prims as core_prims
    import sim_bridge.teach_can as teach_can
    from isaacsim.core.utils.viewports import create_viewport_for_camera
    from sim_bridge import dual

    views = []
    events = []
    original_articulation = core_prims.Articulation

    class TrackedArticulation(original_articulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if 'dynamics' in self.name:
                views.append(self)

    class TestWorld(core_api.World):
        ticks = 0

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.ticks += 1
            if self.ticks in (140, 160, 250, 310):
                from pxr import UsdGeom
                from sim_bridge.wrist_cameras import load_camera_config
                config = load_camera_config()
                stage = omni.usd.get_context().get_stage()
                for side, root in zip(('left', 'right'), ('/World/nero', '/World/nero2')):
                    camera = stage.GetPrimAtPath(root + '/gripper_base/WristCamera')
                    assert camera.IsA(UsdGeom.Camera)
                    local = UsdGeom.Xformable(camera).GetLocalTransformation()
                    np.testing.assert_allclose(list(local.ExtractTranslation()),
                                               config[side]['translation_m'], atol=1e-7)
                events.append('wrist_camera_pose')
            if self.ticks == 150:
                status, _ = omni.kit.commands.execute(
                    'CreatePrim', prim_path='/World/UserScene/Environment/CheckCamera',
                    prim_type='Camera')
                assert status
                app.update()
                events.append('camera_created')
                print('LIFECYCLE_TEST camera_created', flush=True)
            elif self.ticks == 165:
                self.pause()
                app.update()
                create_viewport_for_camera('Lifecycle Camera',
                                           '/World/UserScene/Environment/CheckCamera', 320, 240)
                app.update()
                self.play()
                app.update()
                events.append('camera_viewport')
                print('LIFECYCLE_TEST camera_viewport', flush=True)
            elif self.ticks in (180, 240):
                self.stop()
                app.update()
                self.play()
                app.update()
                events.append('stop_play')
                print('LIFECYCLE_TEST stop_play', flush=True)
            elif self.ticks == 210:
                # Reproduce the exact missing-_physics_view state in the report.
                # The robot USD remains intact, so the bridge must rebuild the view.
                views[-1].__del__()
                assert not hasattr(views[-1], '_physics_view')
                events.append('disposed_dynamics')
                print('LIFECYCLE_TEST disposed_dynamics', flush=True)
            elif self.ticks == 260:
                stage = omni.usd.get_context().get_stage()
                stage.DefinePrim('/World/UserScene/Environment/UnrelatedTemporaryPrim', 'Xform')
                stage.RemovePrim('/World/UserScene/Environment/UnrelatedTemporaryPrim')
                events.append('scene_edit')
            elif self.ticks == 285:
                robot = self.scene.get_object('nero1')
                robot._articulation_view.__del__()
                events.append('disposed_robot')
                print('LIFECYCLE_TEST disposed_robot', flush=True)
            return result

    class NoJoystick:
        message = 'Lifecycle test; no hardware'
        pressed = set()
        buttons = set()
        def __init__(self, *args):
            pass
        def poll(self):
            return False
        def close(self):
            pass

    class NoCan:
        def __init__(self, *args, **kwargs):
            pass
        def poll(self):
            pass
        def snapshot(self, channel):
            return dict(joints=dual.HOME.copy(), opening=.05)
        def close(self):
            pass

    teach_can.TeachCanReader = NoCan
    core_api.World = TestWorld
    core_prims.Articulation = TrackedArticulation
    dual.XboxJoystick = NoJoystick
    dual.run_dual(app, args)
    assert events.count('stop_play') == 2 and 'disposed_dynamics' in events
    for robot_name in ('nero1', 'nero2'):
        robot = core_api.World.instance().scene.get_object(robot_name)
        assert robot.handles_initialized
        assert np.isfinite(robot.get_joint_positions()).all()
        from sim_bridge.grasp import FINGER_STIFFNESS, FINGER_DAMPING, FINGER_MAX_FORCE
        indices = [robot.dof_names.index(name) for name in ('gripper_joint1', 'gripper_joint2')]
        controller = robot.get_articulation_controller()
        kps, kds = controller.get_gains()
        np.testing.assert_allclose(kps[indices], FINGER_STIFFNESS)
        np.testing.assert_allclose(kds[indices], FINGER_DAMPING)
        np.testing.assert_allclose(controller.get_max_efforts()[indices], FINGER_MAX_FORCE)
    print('LIFECYCLE_TEST_PASS mode=' + test_mode + ' ' + ','.join(events), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    app.app.post_quit(1)
    raise
finally:
    app.close()
