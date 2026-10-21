"""Audit synchronized assembly NPZ files and list successful episodes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assembly_collection.geometry import errors, insertion_metrics, keel_hole_metrics, rotation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    successful=[]
    for report_path in sorted(Path(directory).glob('episode_*/report.json')):
        report=json.loads(report_path.read_text())
        trajectory=report_path.with_name('trajectory.npz')
        if not trajectory.exists():
            if report['success']:raise ValueError(f'Successful episode lacks trajectory: {report_path}')
            print(f'REJECT {report_path.parent.name}: {report.get("error","missing data")}')
            continue
        with np.load(trajectory,allow_pickle=False) as data:
            if str(data['task'])=='dual_keel_assembly_v1':
                print(f'LEGACY_PROXY {report_path.parent.name}: fixed-joint model; excluded from contact-only demonstrations')
                continue
            n=len(data['times'])
            shapes=dict(tcp_poses=(n,2,7),desired_tcp_poses=(n,2,7),joint_positions=(n,2,7),
                joint_targets=(n,2,7),finger_joint_positions=(n,2,2),gripper_commands=(n,2),
                keel_poses=(n,7),screw_poses=(n,4,7),stage_ids=(n,),active_screw=(n,),
                insertion_states=(n,4),left_support_active=(n,),keel_linear_velocities=(n,3),screw_linear_velocities=(n,4,3))
            require(n>0 and np.isfinite(data['times']).all() and np.all(np.diff(data['times'])>0),f'Invalid times: {trajectory}')
            for name,shape in shapes.items():
                require(data[name].shape==shape and np.isfinite(data[name]).all(),f'Invalid {name}: {trajectory}')
            for name in ('tcp_poses','desired_tcp_poses','keel_poses','screw_poses'):
                require(np.allclose(np.linalg.norm(data[name][...,3:],axis=-1),1.,atol=1e-3),f'Invalid {name} quaternions: {trajectory}')
            require(str(data['task']) in ('dual_keel_contact_insertion_v2','dual_keel_friction_insertion_v3'),f'Unknown task: {trajectory}')
            require(str(data['fastening_model']) in ('none','passive_compliant_bore_friction') and int(data['assembly_joints'])==0
                and bool(data['gravity_enabled']),f'Artificially constrained parts: {trajectory}')
            if str(data['task'])=='dual_keel_friction_insertion_v3':
                require(bool(data['friction_retention_verified'])==bool(report['friction_retention_verified']),f'Retention label mismatch: {trajectory}')
                if report['success'] and report['stage'] in ('insert-one','insert-all','full','right-pair','release-test'):
                    checks=report.get('retention_checks',[])
                    require(len(checks)==report['expected_screws'] and all(c['verified'] for c in checks),f'Missing free bolt retention check: {trajectory}')
                    require(all(c['observation_seconds']>=3 and not c['hand_contacts'] and c['gripper_gap_m']>.06
                        and c['speed_m_per_s']<.001 and c['position_excursion_m']<.00025
                        and c['stability_window_seconds']>.9 and abs(c['depth_loss_m'])<.001 for c in checks),f'Bolt retention not verified: {trajectory}')
            if str(data['anchor_model'])=='not_modelled' and 'installation_complete' in data:
                require(not bool(data['installation_complete']),f'Unmodelled anchor incorrectly marked complete: {trajectory}')
            if 'installation_complete' in data and bool(data['installation_complete']):
                require(report['success'] and report['stage']=='full' and report['expected_screws']==4
                    and bool(data['independently_mounted']),f'Assembly completion unverified: {trajectory}')
            require(bool(data['success'])==report['success'],f'Conflicting success labels: {trajectory}')
            if 'installation_complete' in data:
                require(bool(data['installation_complete'])==report.get('installation_complete',False),f'Conflicting completion labels: {trajectory}')
            require(bool(data['independently_mounted'])==report.get('independently_mounted',False),f'Conflicting mounting labels: {trajectory}')
            require(((data['stage_ids']>=0)&(data['stage_ids']<len(data['stage_names']))).all(),f'Invalid stage IDs: {trajectory}')
            require(((data['active_screw']>=-1)&(data['active_screw']<=3)).all(),f'Invalid screw IDs: {trajectory}')
            if report['success'] and report['stage'] in ('insert-one','full','right-pair','insert-all','release-test'):
                require(len(report['screws'])==report.get('expected_screws',len(report['screws'])),f'Missing installed screws: {trajectory}')
                normal=rotation(data['wall_pose']).apply([0,-1,0])
                for screw in report['screws']:
                    index=screw['screw_index']-1
                    m=insertion_metrics(data['screw_poses'][-1,index],data['wall_holes'][index],normal)
                    require(.020<m['depth_m']<.0265 and max(m['radial_error_m'],m['entry_radial_error_m'])<.00055
                        and m['axis_error_rad']<.05,f'Screw moved after physical release: {trajectory}, {index+1}')
                    shoulder=float(data['bolt_shoulder_radius_m']) if 'bolt_shoulder_radius_m' in data else 0.
                    m=keel_hole_metrics(data['screw_poses'][-1,index],data['keel_poses'][-1],index,shoulder or .004)
                    margin=.00005 if shoulder else .00015
                    limit=m['keel_hole_clearance_m']+.00010
                    require(m['keel_shaft_crosses_plate'] and m['keel_hole_radial_error_m']<limit,
                        f'Screw misses actual keel hole: {trajectory}, {index+1}')
                    supported=screw.get('left_support_active_at_insertion',not (report['stage']=='full' and index<2))
                    insertion_limit=screw['keel_hole_clearance_m']-margin if supported else screw['keel_hole_clearance_m']+.0001
                    require(.022<screw['depth_m']<.0265 and max(screw['radial_error_m'],screw['entry_radial_error_m'])<.00045
                        and screw['axis_error_rad']<.02 and screw['keel_shaft_crosses_plate']
                        and screw['keel_hole_radial_error_m']<insertion_limit,
                        f'Insertion before release lacked centering clearance: {trajectory}, {index+1}')
                if report['stage'] in ('insert-all','insert-one'):
                    d,a=errors(data['keel_poses'][-1],data['keel_target'])
                    require(d<.001 and a<.01,f'Keel moved out of support alignment: {trajectory}')
                    require(bool(data['left_support_active'][-1]) and not bool(data['independently_mounted']),
                        f'Supported insertion incorrectly labelled as mounting: {trajectory}')
                    require(.6*.038<data['finger_joint_positions'][-1,0,0]-data['finger_joint_positions'][-1,0,1]<1.4*.038,
                        f'Left support grasp lost: {trajectory}')
                else:
                    require(not bool(data['left_support_active'][-1]) and bool(data['independently_mounted']),
                        f'Free-release claim has not been verified: {trajectory}')
                    require(np.min(data['finger_joint_positions'][-1,:,0]-data['finger_joint_positions'][-1,:,1])>.06,
                        f'Grippers have not released: {trajectory}')
                    d,a=errors(data['keel_poses'][-1],data['keel_target'])
                    require(d<.002 and a<.02,f'Unassisted keel moved: {trajectory}')
                    if report['stage'] in ('full','right-pair'):
                        order=report['screw_order']
                        require(set(order[:2])=={3,4},f'Left hand released before right pair: {trajectory}')
                        supported=np.flatnonzero(data['left_support_active'])
                        require(len(supported)>0,f'Missing initial support: {trajectory}')
                        off=np.flatnonzero(~data['left_support_active'] & (np.arange(n)>supported[0]))
                        require(len(off)>0 and not data['left_support_active'][off[0]:].any(),f'Left hand still follows keel: {trajectory}')
                        require(data['insertion_states'][off[0],2:4].all() and not data['insertion_states'][off[0],:2].any(),
                            f'Handoff not performed after exactly two right bolts: {trajectory}')
                        release=report['right_pair_release'];mounted=report['mounted_check']
                        require(release['independently_mounted'] and not release['hand_contacts'] and min(release['gripper_gaps_m'])>.06,
                            f'Missing physical right-pair release observation: {trajectory}')
                        require(mounted['independently_mounted'] and not mounted['hand_contacts'] and min(mounted['gripper_gaps_m'])>.06,
                            f'Missing final free gravity observation: {trajectory}')
                        require(bool(data['release_after_right_pair']) and np.isclose(float(data['left_release_time_s']),release['release_time_s'])
                            and data['times'][off[0]]>=release['release_time_s'],f'Invalid handoff time: {trajectory}')
                        require(all(c['time_s']<release['release_time_s'] for c in report['retention_checks'][:2]),
                            f'Handoff before both bolt retention checks: {trajectory}')
        if report['success']:
            successful.append(str(trajectory))
            model=report.get('fastening_model','none')
            print(f'PASS {report_path.parent.name}: {report["stage"]}, {n} samples, retention={model}')
        else:
            print(f'REJECT {report_path.parent.name}: {report.get("error","failed")}, {n} samples retained')
    return successful


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    p.add_argument('--require-success',action='store_true')
    args=p.parse_args()
    good=audit(args.directory)
    print(f'Successful episodes: {len(good)}')
    if args.require_success and not good:raise SystemExit(1)
