"""Export a private cell and physical near-hole reset states; no training."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from ppo_assembly.paths import PROJECT,ROOT,ASSETS
faulthandler.enable()
faulthandler.dump_traceback_later(45,repeat=True)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--force',action='store_true',help='Replace this experiment\'s generated assets')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
cli=parser.parse_args()
if not cli.experience:cli.experience=str(Path(os.environ['ISAAC_PATH'])/'apps/isaacsim.exp.base.python.kit')
if (ASSETS/'cell.usdc').exists() and not cli.force:
    raise SystemExit('Assets already exist. Use --force only when intentionally regenerating them.')
print('PPO_PREPARE launching AppLauncher',flush=True)
launcher=AppLauncher(dict(vars(cli),create_new_stage=True));app=launcher.app
print('PPO_PREPARE app launched',flush=True)

def prepare():
    from pxr import Usd,UsdPhysics,UsdLux,PhysxSchema
    from assembly_collection.simulation import AssemblySimulation
    from assembly_collection.geometry import compose,inverse,canonical_screw_target,pose,rotation
    from assembly_collection.physics import verify_dynamic_parts
    source=PROJECT/'assembly_collection/validation/handoff_full_09'
    args=json.loads((source/'settings.json').read_text())
    args['scene_usd']=PROJECT/'assets/scenes/keel_assembly.usda'
    args['output_dir']=ASSETS;args['headless']=cli.headless
    sim=AssemblySimulation(app,SimpleNamespace(**args))
    with np.load(source/'episode_001/trajectory.npz') as data:
        names=data['stage_names'][data['stage_ids']]
        candidates=np.flatnonzero(np.array([str(n).startswith('screw4_pre_align_') for n in names]))
        i=int(candidates[-1])
        fixture={k:data[k][i].tolist() for k in ('tcp_poses','desired_tcp_poses','joint_positions',
            'joint_targets','finger_joint_positions','keel_poses','screw_poses')}
        fixture.update(keel_target=data['keel_target'].tolist(),wall_pose=data['wall_pose'].tolist(),
            wall_holes=data['wall_holes'].tolist(),screw_targets=data['screw_targets'].tolist(),
            source_recording=str(source.relative_to(PROJECT)/'episode_001/trajectory.npz'),
            source_sha256=hashlib.sha256((source/'episode_001/trajectory.npz').read_bytes()).hexdigest(),
            source_stage=str(names[i]),source_time_s=float(data['times'][i]))
    paths=[sim.keel_path,sim.environment+'/Screw04']
    verify_dynamic_parts(sim.stage,paths)
    sim.world.stop()
    for n in (1,2,3):sim.stage.GetPrimAtPath(sim.environment+f'/Screw{n:02d}').SetActive(False)
    PhysxSchema.PhysxContactReportAPI.Apply(sim.stage.GetPrimAtPath(paths[1])).CreateThresholdAttr(0.)
    # Export only this new file. Flattening embeds robot meshes and authored
    # geometry; original scene/robot sources are never saved or overwritten.
    ASSETS.mkdir(parents=True,exist_ok=True)
    sim.stage.SetDefaultPrim(sim.stage.GetPrimAtPath('/World'))
    sim.stage.Flatten().Export(str(ASSETS/'cell.usdc'))
    exported=Usd.Stage.Open(str(ASSETS/'cell.usdc'))
    for p in list(exported.Traverse()):
        if p.IsA(UsdPhysics.Scene) or p.HasAPI(UsdLux.LightAPI):exported.RemovePrim(p.GetPath())
    for path in ('/World/Goal1','/World/Goal2'):exported.RemovePrim(path)
    exported.GetRootLayer().Save()
    fixture['left_robot_root']=sim.arms[0]['root']
    fixture['right_robot_root']=sim.arms[1]['root']
    fixture['physics_check']=verify_dynamic_parts(exported,paths)
    fixture['reset_prepositioned']=True;fixture['includes_pickup']=False
    fixture['installation_complete']=False
    right_tcp=np.array(fixture['tcp_poses'][1]);bolt=np.array(fixture['screw_poses'][3])
    grasp=compose(inverse(bolt),right_tcp)
    fixture['bolt_to_tcp']=grasp.tolist()
    fixture['tcp_to_bolt']=inverse(grasp).tolist()
    goal=canonical_screw_target(np.array(fixture['screw_targets'][3]),grasp)
    fixture['bolt_goal']=goal.tolist()
    normal=rotation(np.array(fixture['wall_pose'])).apply([0.,-1.,0.])
    horizontal=rotation(np.array(fixture['wall_pose'])).apply([1.,0.,0.])
    up=np.array([0.,0.,1.])
    fixture['basis']=np.column_stack((normal,horizontal,up)).tolist()
    rng=np.random.default_rng(7391);resets=[];levels=[];splits=[];joint_states=[];tcp_states=[]
    # A finite, disclosed reset library avoids per-episode CPU/global IK.
    # Training and held-out evaluation have disjoint generated states.
    for level,(radial,tilt,distance) in enumerate(((.00025,.003,.032),(.0008,.012,.038),(.0015,.025,.044))):
        for split,count in ((0,64),(1,32)):
            accepted=0;attempts=0
            while accepted<count:
                attempts+=1
                if attempts>count*20:raise RuntimeError(f'Reset poses not reachable: level={level}')
                start=goal.copy()
                start[:3]+=normal*(distance+rng.uniform(-.002,.002))+horizontal*rng.uniform(-radial,radial)+up*rng.uniform(-radial,radial)
                rr=Rotation.from_rotvec(horizontal*rng.uniform(-tilt,tilt)+up*rng.uniform(-tilt,tilt))
                start[3:]=(rr*rotation(start)).as_quat()[[3,0,1,2]]
                wanted=compose(start,grasp)
                try:q=sim.solve(1,wanted,seed=np.array(fixture['joint_positions'][1]))
                except RuntimeError:continue
                resets.append(start);tcp_states.append(wanted);joint_states.append(q);levels.append(level);splits.append(split)
                accepted+=1
        print(f'PPO_ASSETS reset_level={level} generated',flush=True)
    np.savez_compressed(ASSETS/'reset_states.npz',bolt_poses=np.array(resets),tcp_poses=np.array(tcp_states),
        joint_positions=np.array(joint_states),levels=np.array(levels),splits=np.array(splits))
    fixture['cell_sha256']=hashlib.sha256((ASSETS/'cell.usdc').read_bytes()).hexdigest()
    (ASSETS/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    print(f'PPO_ASSETS_DONE {ASSETS} resets={len(resets)} assembly_joints=0 gravity_enabled=true',flush=True)
    faulthandler.cancel_dump_traceback_later()

code=0
try:prepare()
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:
    app.app.post_quit(code);app.close()
raise SystemExit(code)
