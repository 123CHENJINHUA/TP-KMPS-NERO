"""Independent Isaac Sim assembly collector. Launch with scripts/assembly.sh."""
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['preflight', 'keel-grasp', 'keel-align', 'screw-grasp', 'insert-one', 'insert-all', 'full', 'right-pair', 'release-test', 'retention-check', 'support-check', 'motion-check'], default='full')
    p.add_argument('--scene-usd', default='assets/scenes/keel_assembly.usda')
    p.add_argument('--demos', type=int, default=1)
    p.add_argument('--screw-count', type=int, choices=range(1,5), default=4)
    p.add_argument('--first-screw', type=int, choices=range(1,5), default=4)
    p.add_argument('--left-grip-x', type=float, help='Keel-local rail grasp coordinate, metres')
    p.add_argument('--left-tilt-deg', type=float, help='Tilt about keel-local Y for support clearance')
    p.add_argument('--keel-start-yaw', type=float, help='Starting keel yaw, degrees')
    p.add_argument('--pickup-rest-height', type=float, help='Physical pickup supports above the table, metres')
    p.add_argument('--screw-clock-deg', type=float, help='Per-hole gripper clocking to clear the support hand')
    p.add_argument('--screw-grip-offset', type=float, default=.012, help='Lateral grip offset on the head, metres')
    p.add_argument('--elbow-policy', choices=['first', 'separated'], default='first', help='Initial IK branch selection; separated prefers a high left elbow and low right elbow')
    p.add_argument('--left-elbow-height', type=float, default=0., help='Optional elbow height target while supporting, metres; zero disables posture adjustment')
    p.add_argument('--lower-hole-route', choices=['above', 'below'], default='above', help='Candidate corridor for bottom holes, with collision checks retained')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keel-jitter', type=float, default=0., help='Initial XY variation in metres, up to 0.01')
    p.add_argument('--wall-offset-x', type=float, default=.12, help='Move wall toward bases in this isolated live scene, metres')
    p.add_argument('--keel-offset-x', type=float, default=.055, help='Move keel start toward bases to clear the wall, metres')
    p.add_argument('--rack-offset-y', type=float, default=.040, help='Move rack away from the lower wall-hole approach, metres')
    p.add_argument('--screw-head-height', type=float, help='Extended grip head avoids keel ribs, metres')
    p.add_argument('--bolt-retention', choices=['friction','none'], default='friction')
    p.add_argument('--bore-interference', type=float, default=.00004, help='Radial compression in passive contact rings, metres')
    p.add_argument('--bore-static-friction', type=float, default=2.5)
    p.add_argument('--bore-dynamic-friction', type=float, default=2.)
    p.add_argument('--bore-stiffness', type=float, default=2000., help='Compliant contact stiffness, N/m')
    p.add_argument('--bore-damping', type=float, default=10., help='Compliant contact damping, Ns/m')
    p.add_argument('--keel-static-friction', type=float, help='Friction between actual keel bore and bolt shaft')
    p.add_argument('--keel-dynamic-friction', type=float)
    p.add_argument('--handoff-settle', type=float, help='Lower the held plate onto the two shafts before opening, metres')
    p.add_argument('--bolt-shoulder-radius', type=float, help='Optional 3 mm bearing neck outside the wall; radius in metres')
    p.add_argument('--bolt-shoulder-lead-in', type=float, default=.002, help='Lead-in length on the later left bolts, metres')
    p.add_argument('--finger-stiffness', type=float, help='Right hand finger force stiffness, N/m; effort remains capped at 10 N')
    p.add_argument('--retention-seconds', type=float, default=10., help='Observation after right hand clearance, seconds')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--keep-open', action='store_true', help='Keep GUI open after completing the trial')
    p.add_argument('--physics-hz', type=int, default=120)
    p.add_argument('--record-hz', type=int, default=30)
    p.add_argument('--speed', type=float, default=.025, help='Cartesian transport speed, m/s')
    p.add_argument('--insert-speed', type=float, default=.004, help='Insertion speed, m/s')
    p.add_argument('--approach-distance', type=float, help='Bolt stand-off while moving past the keel rails, metres')
    p.add_argument('--output-dir', default='recordings/contact_insertion_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    a = p.parse_args()
    # Stage diagnostics keep the previously verified plain-shaft profile.
    # Handoff and its component checks use the same geometry and grasp.
    handoff=a.stage not in ('insert-one','insert-all','release-test','retention-check')
    defaults=dict(left_grip_x=0.,left_tilt_deg=-65. if handoff else 0.,
        keel_start_yaw=90. if handoff else 0.,pickup_rest_height=.03 if handoff else 0.,
        screw_clock_deg=45. if handoff else 0.,screw_head_height=.05 if handoff else .04,
        keel_static_friction=.4,keel_dynamic_friction=.3,
        bolt_shoulder_radius=.005 if handoff else 0.,handoff_settle=.0002 if handoff else .0012,
        approach_distance=.060 if handoff else .045,finger_stiffness=600.)
    for name,value in defaults.items():
        if getattr(a,name) is None:setattr(a,name,value)
    if a.stage in ('full','right-pair'):
        if a.first_screw not in (3,4) or (a.stage=='full' and a.screw_count<2):
            p.error('Right-pair handoff requires first-screw 3 or 4 and at least two screws')
    if a.demos < 1 or a.seed < 0 or not 0 <= a.keel_jitter <= .01:
        p.error('Use positive demos, nonnegative seed, and keel-jitter in [0, 0.01]')
    if not 0 <= a.wall_offset_x <= .15 or not 0 <= a.keel_offset_x <= .065:
        p.error('Use wall-offset-x in [0,0.15] and keel-offset-x in [0,0.065]')
    if not 0 <= a.rack_offset_y <= .10:
        p.error('Use rack-offset-y in [0,0.10]')
    if not -.055 <= a.left_grip_x <= .055 or not -70 <= a.left_tilt_deg <= 70 or not -180 <= a.keel_start_yaw <= 180:
        p.error('Use left-grip-x in [-0.055,0.055], left-tilt-deg in [-70,70], keel-start-yaw in [-180,180]')
    if not 0 <= a.pickup_rest_height <= .05:
        p.error('Use pickup-rest-height in [0,0.05]')
    if not 0 <= a.screw_clock_deg <= 90:
        p.error('Use screw-clock-deg in [0,90]')
    if not 0 <= a.screw_grip_offset <= .020:
        p.error('Use screw-grip-offset in [0,0.020]')
    if a.left_elbow_height and not 1.15 <= a.left_elbow_height <= 1.45:
        p.error('Use left-elbow-height 0 or in [1.15,1.45]')
    if not .040 <= a.screw_head_height <= .05:
        p.error('Use screw-head-height in [0.040,0.05] metres')
    if not 0 <= a.bore_interference <= .00015 or not 0 <= a.bore_dynamic_friction <= a.bore_static_friction <= 5:
        p.error('Use bore-interference in [0,0.00015] and 0 <= dynamic friction <= static friction <= 5')
    if not 100 <= a.bore_stiffness <= 50000 or not 0 <= a.bore_damping <= 100 or not 3 <= a.retention_seconds <= 120:
        p.error('Use bore-stiffness in [100,50000], damping in [0,100], retention-seconds in [3,120]')
    if not 0 <= a.keel_dynamic_friction <= a.keel_static_friction <= 5:
        p.error('Use 0 <= keel dynamic friction <= static friction <= 5')
    if not 0 <= a.handoff_settle <= .0015:
        p.error('Use handoff-settle in [0,0.0015] metres')
    if a.bolt_shoulder_radius and not .004 < a.bolt_shoulder_radius <= .005:
        p.error('Use bolt-shoulder-radius 0 or in (0.004,0.005] metres')
    if not .0005<=a.bolt_shoulder_lead_in<=.0025 or not 600<=a.finger_stiffness<=2000:
        p.error('Use bolt-shoulder-lead-in in [0.0005,0.0025] and finger-stiffness in [600,2000]')
    if a.physics_hz < 120 or a.record_hz < 1 or a.physics_hz % a.record_hz:
        p.error('physics-hz >=120 and must be divisible by record-hz')
    if not .002 <= a.speed <= .08 or not .001 <= a.insert_speed <= .01:
        p.error('Use speed in [0.002,0.08] and insert-speed in [0.001,0.01]')
    if not .045 <= a.approach_distance <= .080:
        p.error('Use approach-distance in [0.045,0.080] metres')
    if a.stage in ('support-check','retention-check') and a.bolt_retention!='friction':
        p.error('Bench stages require bolt-retention friction')
    for name in ('scene_usd', 'output_dir'):
        path = Path(getattr(a,name)).expanduser()
        path = (path if path.is_absolute() else ROOT/path).resolve()
        if not path.is_relative_to(ROOT):
            p.error(f'{name} must be inside this project')
        setattr(a,name,path)
    if not a.scene_usd.is_file():
        p.error('Scene USD does not exist')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use a new or empty output directory')
    a.source_files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT/'assembly_collection').glob('*.py'))}
    return a


if __name__ == '__main__':
    args = arguments()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless})
    exit_code = 0
    try:
        if args.stage=='retention-check':
            from assembly_collection.retention_probe import collect
        elif args.stage=='support-check':
            from assembly_collection.pair_support_probe import collect
        elif args.stage=='motion-check':
            from assembly_collection.motion_probe import collect
        else:
            from assembly_collection.simulation import collect
        exit_code = collect(app, args)
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        app.app.post_quit(exit_code)
        app.close()
    sys.exit(exit_code)
