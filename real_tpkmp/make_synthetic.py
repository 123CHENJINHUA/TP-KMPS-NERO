"""Generate clearly tagged offline fixtures. These data can never execute on hardware."""
from .bootstrap import new_directory, write_json, utc_now
import argparse
import numpy as np
from .data import config, save_record, physical_signature, frame_document
from .kinematics import tcp_pose


def generate(directory, cfg=None, demos=6):
    cfg = cfg or config()
    output = new_directory(directory)
    for i in range(demos):
        shift = (i-(demos-1)/2)*.001
        home = np.array([-.3, -.3, .3, 1., .1, .1, 0.])
        pick = home+np.array([.08+shift, .12, 0, .12, 0, 0, .04])
        lift = pick+np.array([0, -.05, 0, -.1, 0, 0, 0])
        place = home+np.array([-.1, .14+shift, .04, .08, 0, 0, -.05])
        retreat = place+np.array([0, -.07, 0, -.1, 0, 0, 0])
        ends = [(home, pick), (pick, pick), (pick, lift), (lift, place), (place, place), (place, retreat)]
        durations = [3., 3., 2., 4., 3., 2.]
        clock, times, stages, joints, commands, measured = 0., [], [], [], [], []
        for stage, ((a, b), duration) in enumerate(zip(ends, durations)):
            u = np.arange(0, duration, 1/cfg['record_hz']) / duration
            smooth = 10*u**3-15*u**4+6*u**5
            joints.extend(a[None]+smooth[:, None]*(b-a)[None])
            times.extend(clock+u*duration)
            stages.extend([stage]*len(u))
            cmd = .08 if stage in (0, 4, 5) else 0.
            commands.extend([cmd]*len(u))
            measured.extend([cmd if cmd else .02]*len(u))
            clock += duration
        q = np.array(joints)
        tcp = np.array([tcp_pose(row, cfg) for row in q])
        frames = np.array([tcp_pose(row, cfg) for row in (home, pick, place)])
        meta = dict(schema_version=1, robot='nero', kind='demonstration', synthetic=True,
                    signature=physical_signature(cfg), config=cfg, gripper_mode='width',
                    success=True, success_source='synthetic_fixture', joint_source='measured',
                    created_utc=utc_now(), task='pick_place')
        save_record(output/f'demo_{i+1:03d}.npz', meta, time_s=np.array(times), stage=np.array(stages, int),
                    joints_rad=q, tcp=tcp, gripper_command=np.array(commands), gripper_measured=np.array(measured),
                    gripper_force_N=np.full(len(q), 4.), frames=frames)
        write_json(output/f'demo_{i+1:03d}_frames.json', frame_document(frames, cfg, synthetic=True))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--demos', type=int, default=6)
    args = parser.parse_args()
    if args.demos < 1:
        parser.error('--demos 必须为正整数')
    print(generate(args.out, demos=args.demos))


if __name__ == '__main__':
    main()
