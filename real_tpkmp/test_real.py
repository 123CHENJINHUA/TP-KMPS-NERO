"""Default: offline prediction/IK preview. Explicit --plan ... --execute: real follower."""
from .bootstrap import new_directory, write_json, digest, utc_now
import argparse
import time
import numpy as np
from .data import config, read_record, save_record
from .kinematics import pose_error
from .planning import make_plan, verify_plan
from .learning import errors


def preview(path, data, actual=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 4.5))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(*data['tcp'][:, :3].T, label='planned TCP')
    if actual is not None and len(actual):
        ax.plot(*actual[:, :3].T, label='measured TCP')
    ax.scatter(*data['frames'][:, :3].T, label='start/pick/place')
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='TCP in robot base')
    ax.legend(fontsize=7)
    ax = fig.add_subplot(132)
    for j in range(7):
        ax.plot(data['time_s'], np.rad2deg(data['joints_rad'][:, j]), label=f'J{j+1}')
    ax.legend(fontsize=7, ncol=2)
    ax.set(xlabel='time (s)', ylabel='joint targets (deg)')
    ax = fig.add_subplot(133)
    ax.plot(data['time_s'], data['gripper_command'])
    ax.set(xlabel='time (s)', ylabel='gripper native target (m or deg)')
    fig.subplots_adjust(left=.03, right=.99, top=.90, bottom=.15, wspace=.35)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_start(state, data, cfg, mode):
    position, angle = pose_error(state['tcp'], data['tcp'][0])
    if (np.max(abs(state['joints_rad']-data['joints_rad'][0])) > cfg['start_joint_tolerance_rad']
            or position > cfg['start_position_tolerance_m'] or angle > cfg['start_angle_tolerance_rad']):
        raise ValueError(f'从臂不在轨迹起点（{position*1000:.1f}mm/{np.rad2deg(angle):.1f}deg）；'
                         '先用主臂移到预览起点，再停止主臂发令；脚本不会自动回零/移到起点')
    tolerance = .005 if mode == 'width' else 5.
    if state['gripper_mode'] != mode or abs(state['gripper_measured']-data['gripper_command'][0]) > tolerance:
        raise ValueError('起始夹爪模式/开度与计划不一致；请先匹配示教起始开度')


def check_tracking(state, joints, pose, cfg):
    position, angle = pose_error(state['tcp'], pose)
    if (np.max(abs(state['joints_rad']-joints)) > cfg['max_tracking_joint_error_rad']
            or position > cfg['max_tracking_position_error_m'] or angle > cfg['max_tracking_angle_error_rad']):
        raise ValueError(f'跟踪误差过大：{position*1000:.1f}mm / {np.rad2deg(angle):.1f}deg')


def execute(plan_path, meta, data, cfg, output):
    # This function is reached only through explicit --execute, after offline checks.
    from .hardware import Reader
    verify_plan(meta, data, cfg)
    reader = Reader(cfg)
    rows, failure, stop_failure, completed = [], None, None, False
    motion_started = False
    try:
        reader.wait_ready(False)
        print('正在被动检查 CAN 控制权和反馈；主臂必须停止发令，电脑需继续连接从臂总线。', flush=True)
        begin = time.monotonic()
        while time.monotonic()-begin < 1.2:
            reader.sample(False)
            time.sleep(.02)
        reader.require_exclusive()
        reader.require_enabled()
        state = reader.sample(False)
        check_start(state, data, cfg, meta['gripper_mode'])
        stationary = []
        for _ in range(15):
            stationary.append(reader.sample(False)['joints_rad'])
            time.sleep(.02)
        if np.max(np.ptp(stationary, axis=0)) > .01:
            raise ValueError('从臂尚未停稳')
        reader.require_exclusive()
        check_start(reader.sample(False), data, cfg, meta['gripper_mode'])
        foreign_stamp = reader.foreign_stamp()
        reader.attach_writer()
        reader.require_no_new_controller(foreign_stamp)
        motion_started = True
        robot, gripper = reader.robot, reader.gripper
        robot.set_speed_percent(int(cfg['sdk_speed_percent']))
        robot.set_motion_mode('j')
        print(f'执行 {data["time_s"][-1]:.1f}s；Ctrl+C 触发 SDK 阻尼急停。', flush=True)
        started = time.monotonic()
        last_status = -10.
        for i, target_time in enumerate(data['time_s']):
            time.sleep(max(0, started+target_time-time.monotonic()))
            elapsed = time.monotonic()-started
            if elapsed-target_time > cfg['max_sample_gap_s']:
                raise ValueError('控制循环超时，停止回放')
            reader.require_no_new_controller(foreign_stamp)
            reader.require_enabled()
            state = reader.sample(False)
            check_tracking(state, data['joints_rad'][i], data['tcp'][i], cfg)
            if state['gripper_mode'] != meta['gripper_mode']:
                raise ValueError('执行时夹爪反馈模式改变')
            robot.move_j(data['joints_rad'][i].tolist())
            command = float(data['gripper_command'][i])
            force = float(data['gripper_force_N'][i])
            if meta['gripper_mode'] == 'width':
                gripper.move_gripper_m(command, force)
            else:
                gripper.move_gripper_deg(command, force)
            rows.append((elapsed, i, state))
            if elapsed-last_status > 2:
                print(f'{elapsed:.1f}/{data["time_s"][-1]:.1f}s | phase={data["phase"][i]:.3f}', flush=True)
                last_status = elapsed
        deadline = time.monotonic()+cfg['settle_timeout_s']
        while True:
            reader.require_no_new_controller(foreign_stamp)
            reader.require_enabled()
            state = reader.sample(False)
            check_tracking(state, data['joints_rad'][-1], data['tcp'][-1], cfg)
            p, a = pose_error(state['tcp'], data['tcp'][-1])
            joint_error = np.max(abs(state['joints_rad']-data['joints_rad'][-1]))
            if p <= cfg['start_position_tolerance_m'] and a <= cfg['start_angle_tolerance_rad'] and joint_error <= cfg['start_joint_tolerance_rad']:
                completed = True
                break
            if time.monotonic() >= deadline:
                raise ValueError('最终位置未在等待时间内到达')
            robot.move_j(data['joints_rad'][-1].tolist())
            time.sleep(1/cfg['control_hz'])
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or 'operator_interrupt'
        if motion_started:
            try:
                reader.robot.electronic_emergency_stop()
            except Exception as stop_exc:
                stop_failure = str(stop_exc)
        print(f'执行中止：{failure}', flush=True)
    finally:
        reader.close()
    if rows:
        actual = np.array([row[2]['tcp'] for row in rows])
        indices = np.array([row[1] for row in rows])
        metric = errors(actual, data['tcp'][indices])
        tracking_passed = (completed and metric['position_rms_m'] <= cfg['tracking_rms_position_limit_m']
                           and metric['rotation_rms_rad'] <= cfg['tracking_rms_angle_limit_rad'])
        save_record(output/'feedback.npz', dict(schema_version=1, robot='nero', kind='execution', synthetic=False,
                    plan_sha256=digest(plan_path), completed=completed, created_utc=utc_now()),
                    time_s=np.array([row[0] for row in rows]), plan_indices=indices, tcp=actual,
                    joints_rad=np.array([row[2]['joints_rad'] for row in rows]),
                    gripper_measured=np.array([row[2]['gripper_measured'] for row in rows]))
        preview(output/'execution.png', data, actual)
    else:
        metric, tracking_passed = {}, False
    write_json(output/'execution.json', dict(completed=completed, tracking_passed=tracking_passed,
                physical_task_success='operator_must_check_object_pick_and_place', failure=failure,
                emergency_stop_send_failure=stop_failure, metrics=metric, plan_sha256=digest(plan_path)))
    print(f'轨迹跟踪通过={tracking_passed}；请检查实际是否抓住并放稳物体。报告：{output / "execution.json"}')
    if failure or not tracking_passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--frames')
    parser.add_argument('--plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--can')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.execute:
        if not args.plan or args.model or args.frames:
            parser.error('执行仅接受已预览的 --plan ... --execute，不可直接执行模型')
        meta, data = read_record(args.plan, 'plan')
        cfg = config(args.config) if args.config else meta['config'].copy()
        if {k: v for k, v in cfg.items() if k != 'channel'} != {k: v for k, v in meta['config'].items() if k != 'channel'}:
            raise ValueError('计划生成后配置已改变，请重新规划')
        if args.can:
            cfg['channel'] = args.can
        verify_plan(meta, data, cfg)  # Before opening a CAN socket.
        execute(args.plan, meta, data, cfg, new_directory(args.out))
    else:
        if not args.model or not args.frames or args.plan:
            parser.error('离线预览需要 --model ... --frames ... --out ...')
        meta, _ = read_record(args.model, 'model')
        cfg = config(args.config) if args.config else meta['config'].copy()
        if args.can:
            cfg['channel'] = args.can
        plan_meta, plan, report = make_plan(args.model, args.frames, cfg)
        output = new_directory(args.out)
        save_record(output/'plan.npz', plan_meta, **plan)
        write_json(output/'preview.json', report)
        preview(output/'preview.png', plan)
        print(f'仅离线计算，未连接 CAN。计划：{output / "plan.npz"}\n预览：{output / "preview.png"}')
        print(f'时长={report["duration_s"]:.1f}s；留出通过={report["heldout_passed"]}；可执行={report["executable"]}')
        if report.get('execution_blocked_reasons'):
            print(f'参考系执行阻止原因：{report["execution_blocked_reasons"]}')
        print(f'起点七轴角度(deg)：{np.round(report["start_joints_deg"], 2).tolist()}；'
              f'夹爪 {report["gripper_mode"]}={report["start_gripper_native"]:.4f}')


if __name__ == '__main__':
    main()
