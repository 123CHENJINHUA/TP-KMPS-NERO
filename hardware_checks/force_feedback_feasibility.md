# 仿真抓取向 AGX 示教夹爪反馈力：资料核查

核查日期：2026-09-29。适用当前项目的双 Nero 示教同步（can0/can1，AGX 示教夹爪）。本次只阅读资料、SDK 和已有报告，没有连接硬件、发 CAN 指令或修改同步控制。

## 结论

Isaac Sim 端可以读取仿真夹指与物体的接触力；当前程序没有将该力发送到真实夹爪。公开 pyAgxArm 文档和 AGX 夹爪驱动没有明确的“Leader 示教端接收外部实时反馈力”接口，因此不能仅凭 `agx_gripper` 类型或 SDK 的 `force` 参数保证能做触觉反馈。这里的结论是公开接口尚未确认，并非证明硬件一定不能实现。

## 接口含义

| 接口/报文 | 文档与实现含义 | 对真实手感的意义 |
|---|---|---|
| `get_gripper_ctrl_states()` / `0x159` | Leader 夹爪控制指令回显；包括目标开度及力参数 | 读取控制回显不会产生反馈阻力 |
| `get_gripper_status()` / `0x2A8` | 实际夹爪位置、夹持力和驱动状态上报 | 能读真实夹爪状态，不表示可将该报文作为示教端控制输入 |
| `move_gripper_m(value, force)`、`move_gripper_deg(value, force)` | 位置目标与目标夹持力控制；SDK 发送 `0x159` | 文档没有说明它们能在 Leader 模式保持手动示教并叠加反作用力；直接发送可能改变运动行为 |
| `set_gripper_teaching_pendant_param(... teaching_friction=...)` / `0x47D` | 示教手感配置，范围 1–10；文档说明数值越大越容易开合、所需用力越小 | 是离散配置档位，没有实时带符号反馈力参数，也没有承诺高频调整能力 |
| Nero `set_leader_mode()` | 零力拖动主导模式 | 不自动建立仿真到示教端的力反馈通路 |
| Nero `move_mit()` | 七个机械臂关节的底层控制 | 不能据此推定末端示教夹爪有相同接口 |

本地 `hardware_checks/reports/read_20260927_151944.json` 中，can0/can1 的示教夹爪力回显在所有记录样本中均为 3.0 N，实际夹爪状态更新计数均为 0。这不是抓取力测量，也不是示教夹爪可接受反馈力的验证。当前 `sim_bridge/teach_can.py` 只接收报文，不发送报文。

## 仿真端可实现的工作

给两臂的 `gripper_link1`、`gripper_link2` 配置接触力读取，按抓取目标过滤指尖接触，分别统计沿夹爪闭合方向的压缩力。不要将两指相反方向的力直接求矢量和，否则可能抵消；也不要把夹爪电机目标力或驱动上限当成实际物体接触力。

Isaac 5.1 的 `RigidPrim.get_contact_force_matrix()` 可以读取指定对象之间的接触量。应在创建时配置接触过滤，读取时传入真实物理步长 `dt=1 / physics_hz`；默认 `dt=1` 实际返回接触冲量。当前仿真每指最大驱动力 10 N，这是仿真驱动参数，不是示教夹爪的允许反馈力。

第一阶段可显示左右臂实际接触力和夹爪开度、记录随时间变化的曲线，在仿真中验证双指接触与抓取效果，不需要向真实机械臂发指令。

## 真正接入手感前需要确认的硬件信息

请向厂家确认当前 AGX 示教夹爪的具体型号/固件是否具备主动反馈执行器，以及：

1. 保持 Leader 手动开合时，能否叠加外部阻力、反馈力矩或指定电流？提供对应 CAN 指令/SDK 接口。
2. 若通过模拟从端状态帧返回力，请明确指定报文及字段；公开 `0x2A8` 定义只说明状态上报，不能据此推断 Leader 固件会使用它产生力反馈。
3. 力、电流或力矩的单位、符号方向、推荐更新频率、允许范围、通信超时后的行为。
4. `teaching_friction` 是否仅为手感参数，是否允许运行中调整、是否有写入频率/持久化限制。若只支持此档位，可评估“接触后增阻”的粗略提示，但不能称为连续真实力反馈。

支持确认后，再做“仿真接触力 → 滤波与标定 → 限幅/变化率限制 → 示教夹爪反馈”的独立通路。需处理仿真暂停、失去接触、通信断开及急停时的反馈释放。左右开度现均采用直接映射；若后续加入力反馈，反馈作用方向仍需要分别标定。

若通过普通 `0x159` 运动指令试做反馈，当前读取器会把同一通道收到的此类指令也当作示教输入，需要明确区分发送与真实输入，不能直接接进现有循环。

## 官方资料

- [AgxGripper API 文档](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/effector/agx_gripper/agx_gripper_api.md)
- [AGX 夹爪驱动实现](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/driver.py)
- [AGX 夹爪协议解析及发送映射](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/parser.py)
- [Nero API 文档](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero_api.md)
- [Isaac Sim 5.1 RigidPrim API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.RigidPrim.get_contact_force_matrix)
