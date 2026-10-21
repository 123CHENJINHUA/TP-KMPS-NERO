"""Plot measured assembly trajectories without starting Isaac Sim."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'assembly_collection/.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assembly_collection.geometry import rotation

def plot_episode(trajectory, output):
    with np.load(trajectory,allow_pickle=False) as data:
        times=data['times'];tcp=data['tcp_poses'];keel=data['keel_poses']
        names=data['stage_names'][data['stage_ids']]
        fig=plt.figure(figsize=(13,9),layout='constrained')
        spatial=fig.add_subplot(2,2,1,projection='3d')
        for i,label in enumerate(('Left TCP','Right TCP')):
            spatial.plot(*tcp[:,i,:3].T,label=label,linewidth=1.)
        spatial.plot(*keel[:,:3].T,label='Keel',color='tab:green',linewidth=1.5)
        spatial.scatter(*data['wall_holes'].T,color='black',label='Wall holes',s=25)
        spatial.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title='Measured motion')
        spatial.legend(fontsize=8)
        spatial.set_box_aspect((1,1,1))

        aligned=fig.add_subplot(2,2,2)
        error=np.linalg.norm(keel[:,:3]-data['keel_target'][:3],axis=1)*1000
        aligned.plot(times,error,color='tab:green',label='Keel position error')
        tolerance=1. if 'left_support_active' in data and bool(data['left_support_active'][-1]) else 2.
        aligned.axhline(tolerance,color='gray',linestyle='--',linewidth=.8,label=f'Pose tolerance: {tolerance:g} mm')
        aligned.set(yscale='symlog',xlabel='Simulation time (s)',ylabel='Error (mm)',title='Keel alignment and gravity response')
        aligned.set_ylim(bottom=0.)
        aligned.legend(fontsize=8)

        insertion=fig.add_subplot(2,2,3)
        normal=rotation(data['wall_pose']).apply([0,-1,0])
        for i in range(4):
            poses=data['screw_poses'][:,i]
            rot=Rotation.from_quat(poses[:,[4,5,6,3]])
            tip=poses[:,:3]-.028*rot.apply(np.tile([0.,0.,1.],(len(times),1)))
            depth=-(tip-data['wall_holes'][i])@normal*1000
            selected=(data['active_screw']==i)&np.array([str(n).startswith(f'screw{i+1}_') and
                any(phase in str(n) for phase in ('_pre_align','_insert','_fasten')) for n in names])
            if selected.any():
                # Continue plotting after both hands release and while other
                # screws are handled, so a slipping pin remains visible.
                start=np.flatnonzero(selected)[0]
                insertion.plot(times[start:],depth[start:],label=f'Screw {i+1}')
        insertion.axhspan(22.,26.5,color='tab:green',alpha=.12,label='Insertion tolerance before release')
        insertion.set(xlabel='Simulation time (s)',ylabel='Tip depth into wall (mm)',title='Measured screw insertion')
        insertion.legend(fontsize=8)

        grippers=fig.add_subplot(2,2,4)
        gaps=(data['finger_joint_positions'][...,0]-data['finger_joint_positions'][...,1])*1000
        for i,label in enumerate(('Left','Right')):
            line,=grippers.plot(times,gaps[:,i],label=f'{label} measured')
            grippers.plot(times,data['gripper_commands'][:,i]*1000,color=line.get_color(),
                linestyle='--',alpha=.45,linewidth=.8,label=f'{label} command')
        grippers.set(xlabel='Simulation time (s)',ylabel='Total finger gap (mm)',title='Physical grasp and release')
        if 'left_release_time_s' in data and float(data['left_release_time_s'])>=0:
            for ax in (aligned,insertion,grippers):
                ax.axvline(float(data['left_release_time_s']),color='tab:purple',linestyle=':',linewidth=1.,
                    label='Left hand releases after right pair' if ax is grippers else None)
        grippers.legend(fontsize=8)
        for ax in (aligned,insertion,grippers):ax.grid(alpha=.2)
        status=('LEGACY FIXED-JOINT PROXY' if str(data['task'])=='dual_keel_assembly_v1' else
            'INSTALLATION COMPLETE (FRICTION MODEL)' if 'installation_complete' in data and bool(data['installation_complete']) else
            'PASSIVE FRICTION STAGE PASSED' if bool(data['success']) and str(data['fastening_model'])=='passive_compliant_bore_friction' else
            'CONTACT STAGE PASSED (NO ANCHOR)' if bool(data['success']) else 'CONTACT STAGE REJECTED')
        fig.suptitle(f'Dual keel assembly: {status} | {trajectory.parent.name} | {len(times)} samples',fontsize=13)
        fig.savefig(output,dpi=150)
        plt.close(fig)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',help='Episode directory or trajectory.npz')
    parser.add_argument('--output',help='PNG path inside this project')
    args=parser.parse_args()
    source=Path(args.episode).resolve()
    if source.is_dir():source=source/'trajectory.npz'
    output=Path(args.output).resolve() if args.output else source.with_name('validation.png')
    if not output.is_relative_to(ROOT):parser.error('Output must be inside this project')
    if not source.is_file():parser.error('trajectory.npz does not exist')
    if not output.parent.is_dir():parser.error('Output directory does not exist')
    print(plot_episode(source,output))
