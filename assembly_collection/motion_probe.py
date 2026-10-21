"""Reproduce empty-wrist tracking with two IK seeds; never a training demo."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.utils.types import ArticulationAction
from assembly_collection.geometry import errors, pose
from assembly_collection.simulation import AssemblySimulation, TrialFailed, write_json


def collect(app,args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sim=AssemblySimulation(app,args)
    # Captured before an empty-hand retreat. Initialization is explicitly a
    # diagnostic state placement, not a physical pickup/installation episode.
    initial_q=np.array([.793427646,1.225593567,.629386544,1.046523452,
        .776980817,.268988788,-1.400548100])
    wanted=np.array([-.277964502,-.066336782,1.4,
        -.641809461,.268062118,.664837227,-.272423895])
    normal_command=sim.command_tcp
    trials=[]
    for mode in ('actual_joint_seed','command_joint_seed','cached_actual_joint_seed'):
        sim.command_tcp=normal_command
        sim.initialize(np.zeros(2))
        for i,a in enumerate(sim.arms):
            q=initial_q if i==1 else np.array([0.,-.4,0.,1.,0.,.2,0.])
            full=np.zeros(a['robot'].num_dof);full[a['indices']]=q
            full[a['fingers']]=[.035,-.035]
            a['robot'].set_joint_positions(full)
            a['robot'].set_joint_velocities(np.zeros(len(full)))
            a['command']=full.copy();a['opening']=.07;a['desired']=sim.tcp(i)
            a['robot'].apply_action(ArticulationAction(joint_positions=full))
        if mode!='cached_actual_joint_seed':
            def command(index,target):
                a=sim.arms[index];ix=a['indices']
                seed=a['command'][ix] if mode=='command_joint_seed' else a['robot'].get_joint_positions()[ix]
                q=sim.solve(index,target,seed=seed)
                a['command'][ix]+=np.clip(q-a['command'][ix],-.8/args.physics_hz,.8/args.physics_hz)
                a['desired']=target.copy()
                a['marker'].set_world_pose(target[:3],target[3:])
            sim.command_tcp=command
        sim.record=False;sim.wait(1.);sim.rows=[];sim.labels=[];sim.record=True
        result=dict(mode=mode,success=False,prepositioned=True,robot_demonstration=False)
        try:
            sim.move(1,wanted,'empty_raise')
            result['success']=True
        except TrialFailed as exc:result['error']=str(exc)
        actual=sim.tcp(1)
        a=sim.arms[1]
        p,r=a['solver'].compute_forward_kinematics('gripper_base',a['command'][a['indices']])
        result.update(actual_error=list(errors(actual,wanted)),
            command_fk_error=list(errors(pose(p,Rotation.from_matrix(r)),wanted)),
            final_joint_positions=a['robot'].get_joint_positions()[a['indices']].tolist(),
            final_joint_targets=a['command'][a['indices']].tolist(),
            arm_gains=[v.tolist() for v in a['robot'].get_articulation_controller().get_gains()])
        trials.append(result)
        path=args.output_dir/f'{mode}.npz'
        sim.save_recording(path,False)
        with np.load(path) as data:bench={k:data[k] for k in data.files}
        bench.update(task='empty_wrist_motion_bench_v1',prepositioned=True,robot_demonstration=False)
        np.savez_compressed(path,**bench)
        print('MOTION_PROBE '+json.dumps(result),flush=True)
    summary=dict(task='empty_wrist_motion_bench_v1',prepositioned=True,robot_demonstration=False,
        settings={k:str(v) if hasattr(v,'is_file') else v for k,v in vars(args).items()},trials=trials,
        success=trials[-1]['success'])
    write_json(args.output_dir/'summary.json',summary)
    return 0 if summary['success'] else 1
