"""Calibrate insertion contact forces using the preserved position baseline."""
import argparse,json,os
from pathlib import Path
from isaaclab.app import AppLauncher
from ppo_assembly.paths import ROOT,local_path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--episodes',type=int,default=1)
parser.add_argument('--num-envs',type=int,default=1)
parser.add_argument('--run-dir',default='ppo_assembly/validation/force_probe_01')
AppLauncher.add_app_launcher_args(parser);cli=parser.parse_args()
if not cli.experience:cli.experience=str(Path(os.environ['ISAAC_PATH'])/'apps/isaacsim.exp.base.python.kit')
out=local_path(cli.run_dir);out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):parser.error('Use a new output directory')
launcher=AppLauncher(dict(vars(cli),create_new_stage=True));app=launcher.app
code=0
try:
    import torch,numpy as np
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    from ppo_assembly.legacy_v1.environment import BoltInsertionEnv,BoltInsertionCfg
    class ProbeEnv(BoltInsertionEnv):
        def _setup_scene(self):
            super()._setup_scene()
            env='/World/envs/env_.*/Cell/UserScene/Environment'
            filters=[env+'/AssemblyWall/Plate',env+'/Geometry']
            filters += [env+f'/AssemblyWall/FrictionBores/Hole4Ring{ring}Sector{sector:02d}'
                for ring in (1,2) for sector in range(32)]
            self.insertion_contacts=ContactSensor(ContactSensorCfg(prim_path=env+'/Screw04',
                update_period=0.,history_length=1,track_friction_forces=True,
                max_contact_data_count_per_prim=512,
                filter_prim_paths_expr=filters))
            self.scene.sensors['insertion_contacts']=self.insertion_contacts
    cfg=BoltInsertionCfg();cfg.scene.num_envs=cli.num_envs;cfg.sim.device=cli.device
    cfg.evaluation=True;cfg.retention_seconds=3.
    env=ProbeEnv(cfg);env.attach_csv(out/'episodes.csv');env.reset()
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics
    stage=get_current_stage()
    collision_paths=[str(p.GetPath()) for p in stage.Traverse() if 'AssemblyWall' in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI)]
    print('FORCE_FILTER_DEBUG '+json.dumps(dict(wall_colliders=collision_paths[:8],
        rigid_bodies=[str(p.GetPath()) for p in stage.Traverse() if 'Environment' in str(p.GetPath()) and p.HasAPI(UsdPhysics.RigidBodyAPI)])),flush=True)
    trace=[]
    with torch.inference_mode():
        for step in range(1000):
            data=env.insertion_contacts.data
            normal=data.force_matrix_w.sum((1,2));friction=data.friction_forces_w.sum((1,2))
            force=normal+friction
            m=env._measure()
            if step in (0,50,80,100):print('FORCE_SAMPLE '+json.dumps(dict(step=step,depth=m['depth'].tolist(),
                filtered_normal=normal.tolist(),filtered_friction=friction.tolist(),net_normal=env.insertion_contacts.data.net_forces_w.tolist(),
                hand_force=m['hand_force'].tolist())),flush=True)
            trace.append(np.column_stack((env.phase.cpu().numpy(),m['depth'].cpu().numpy(),
                normal.cpu().numpy(),friction.cpu().numpy(),force.cpu().numpy())))
            env.step(env.expert_action())
            if env.completed_episodes>=cli.episodes:break
    array=np.array(trace);np.savez_compressed(out/'forces.npz',values=array,
        names=np.array(['phase','depth_m','normal_x_N','normal_y_N','normal_z_N',
            'friction_x_N','friction_y_N','friction_z_N','total_x_N','total_y_N','total_z_N']))
    active=array[:,:,0]==1;forces=array[:,:,8:11][active]
    summary=dict(episodes=env.records,successful=sum(r['success'] for r in env.records),
        normal_peak_N=float(np.linalg.norm(array[:,:,2:5],axis=-1)[active].max()),
        friction_peak_N=float(np.linalg.norm(array[:,:,5:8],axis=-1)[active].max()),
        total_peak_N=float(np.linalg.norm(forces,axis=-1).max()),
        axial_peak_N=float(forces[:,0].max()),lateral_peak_N=float(np.linalg.norm(forces[:,1:],axis=-1).max()),
        force_quantiles_N=np.quantile(forces,[.5,.9,.99],axis=0).tolist(),filter_patterns=env.insertion_contacts.cfg.filter_prim_paths_expr,
        filter_count=env.insertion_contacts.contact_physx_view.filter_count)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('FORCE_PROBE '+json.dumps(summary),flush=True)
    env.close()
    if summary['total_peak_N']<.01:raise RuntimeError('Insertion contact filter returned no physical forces')
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:app.app.post_quit(code);app.close()
raise SystemExit(code)
