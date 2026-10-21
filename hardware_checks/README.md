# 双 Nero 硬件只读检查

本目录内脚本只接收 CAN 报文、读取 SDK 缓存；不调用 `enable`、`set_leader_mode`、`move_*` 或夹爪控制函数。只使用当前项目内的 `pyAgxArm/`，依赖也安装到本目录的 `.vendor/`。示教器广播的关节和夹爪指令回显，分别来自 0x155/0x156/0x157/0x170 和 0x159；实际机械臂与实际夹爪反馈是另外的报文。

先接好两台机械臂的 USB-CAN 适配器，使左右臂分别稳定对应 `can0` / `can1`。若 `probe_can.py` 报 `can0 exists but is DOWN` 或 `can1 exists but is DOWN`，在本机终端输入密码，按 pyAgxArm 官方文档将两路接口设为 1 Mbps 并启用：

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
ip -details link show can0
ip -details link show can1
```

两路应显示 `UP`；若已经是 `UP`，不要重复设置，直接运行下面的接收检查。接口每次重新插拔或重启后可能需要重新启用。此操作只配置主机 CAN 接口，不向机械臂发送运动命令。当前 SDK 对 Nero 固件 1.21 及以上使用 `v121` 配置；若实际固件不同，用 `--firmware` 选择。两个检查脚本都不会配置 CAN 波特率或更改机器人模式。

1. 安装缺少的 `python-can`（仅写入 `hardware_checks/.vendor/`）：

   ```bash
   bash hardware_checks/install_deps.sh
   /home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/read_dual_nero.py --self-check
   ```

2. 无第三方依赖地检查两路总线是否在收报文：

   ```bash
   /home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/probe_can.py --seconds 5
   ```

   输出每路的帧数、每秒帧数、CAN ID；接口缺失、未启用或五秒内没有报文，状态码为 2。可通过 `--report hardware_checks/reports/can_probe.json` 保存报告（文件必须是新路径）。

3. 查看真实关节与 AGX 试教夹爪反馈：

   ```bash
   /home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/read_dual_nero.py --seconds 10
   ```

   每路输出 `get_leader_joint_angles()` 的示教关节角（弧度），并通过 `get_gripper_ctrl_states()` 读取示教夹爪的 0x159 控制回显。`leader_gripper` 的开度和力是示教器给出的目标值，**不是实际夹爪测量值**。若另外收到真实机械臂关节和夹爪状态，也会分别显示为 `measured` 和 `physical_gripper`；试教器单独连接时这两项为空是正常的。`mode=width` 的夹爪值单位为米，`mode=angle` 为度。报告保存在 `hardware_checks/reports/`；两路均有持续示教关节及示教夹爪回显才返回 0。脚本会等待 7 个关节对应的 CAN 分包全部收到，避免把刚连接时未初始化的关节误判为零位。

   运行期间依次轻微移动 `can0` 对应示教臂的一个关节、再移动 `can1` 对应示教臂的一个关节：应只有对应通道的 `leader` 角度随之变化。随后分别开合两个试教夹爪，确认各自的 `leader_gripper` 值变化。这样也能检查左右 CAN 通道有没有接反。

若只收到 CAN 帧、没有完整示教关节反馈，先核对固件版本和左右 CAN 通道。若有示教关节反馈但无 `leader_gripper`，检查 0x159 帧和 AGX 试教夹爪的接线、供电。收到 `leader_gripper` 只说明控制回显正常，不能证明真实夹爪已按指令运动。所有报告均写在当前项目内。


仿真抓取向示教夹爪提供阻力的接口核查，见 [力反馈可行性说明](force_feedback_feasibility.md)。公开 SDK 尚未明确提供 Leader 端实时反馈力控制接口；夹爪控制回显中的力值不能作为触觉反馈已生效的证据。
