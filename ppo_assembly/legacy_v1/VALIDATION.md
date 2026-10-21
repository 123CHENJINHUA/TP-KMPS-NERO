# 独立 PPO 实验验证

验证日期：2026-10-07。使用已有 `/home/cjh/anaconda3/envs/isaaclab`，Isaac Lab 2.3.2、Isaac Sim 5.1、RSL-RL 5.0.1、RTX 4070 Ti SUPER。

这里区分环境的物理可行性与策略训练效果。**没有执行正式长时间训练，也没有得到已收敛的装配策略。** 第一阶段只训练孔前单颗插入；左臂持续物理扶住 keel，拾取和完整安装尚未学习。

## 已完成检查

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 初始状态生成 | 每级 64 个训练、32 个留出状态，共 288 个 | `assets/reset_states.npz`、`assets/fixture.json` |
| 孔贯穿与位姿几何 | 4 项测试通过，与原独立 NumPy 几何实现对照 | `./ppo_assembly/run.sh test` |
| 单环境物理插入 | 1/1 成功，松爪撤离后观察 10 秒 | `validation/expert_05/summary.json` |
| 两个 GPU 并行环境物理插入 | 2/2 成功，分别松爪后观察 10 秒 | `validation/parallel_expert_01/summary.json` |
| PPO 参数更新 | 2 环境 × 32 步 × 2 更新，参数实际变化 | `runs/smoke_01/summary.json` |
| 可视化训练入口 | 1 环境 × 32 步 × 1 更新，GUI、提示和模型保存正常 | `runs/gui_smoke_01/summary.json` |
| 模型回放及自动复位 | GUI 下加载 checkpoint，运行两个完整回合 | `validation/gui_replay_01/summary.json` |
| checkpoint 内容 | 策略、价值网络、优化器与元数据齐全，52 个张量全部有限 | `validation/checkpoint_check.json` |
| 恢复训练 | 加载短测模型和优化器，2 环境再更新 1 次，参数实际变化 | `runs/resume_smoke_01/summary.json` |
| Ctrl+C 保存 | 第一次更新后发送 SIGINT，确认 `interrupted=true`、模型保存及退出码 0 | `runs/interrupt_smoke_03/summary.json`、`validation/interrupt_receipt_03.json` |
| TensorBoard | 本机服务返回两个短测实验的曲线列表；测试服务已关闭 | `validation/tensorboard_tags.json` |
| 曲线 PNG | 已生成并查看损失、探索幅度、插入几何曲线 | `runs/smoke_01/training_curves.png` |

物理成功来自 `check --controller expert`，不是 PPO。三个成功回合最终插入深度为 23.88–23.93 mm，墙孔入口轴线偏差为 0.178–0.181 mm，右爪开口约 70 mm，残余手部接触力为 0 N。记录的两个 env 0 在最后 10 秒的位置变化范围分别约 0.071 mm、0.051 mm。零件重力开启，装配固定约束数为 0。

只进行了极短的 PPO 更新，因此回放两个回合 **0/2 成功**，均正确识别夹持丢失并复位；这验证了加载和评估流程，不能证明学会插入。没有完成回合的短训练，平均回合奖励暂未有样本是正常的。

## 启动与控制修正

启动入口明确使用已有 SDK 的 `isaacsim.exp.base.python.kit`。早期使用完整 UI experience 时，本机曾在 OmniGraph 图像插件启动处出现原生崩溃；最终入口缩小到必要 experience 后，上述检查均正常启动和退出。历史诊断保存在 `validation/startup_history/`。

最初的反馈控制器直接绕手腕调整角度，螺栓离手腕约 168 mm，角度修正引入横向位移，在入孔时出现夹持滑移。改为围绕指令中的螺栓中心旋转后，接触插入通过。此修正只改变机器人目标，不把零件吸附、锁定或传送到目标。

## 运行与结果解释

训练与验证指令见 [README.md](README.md)。正式训练必须使用新输出目录。成功率要结合独立留出回合与 `episodes.csv` 判断，训练奖励高不能代替实际入孔与松手保持。

当前保持测试中左臂仍支撑 keel；这不能作为“两颗螺栓后左臂松手”或四颗安装成功的证据。后续应先验证两颗安装及左臂松手任务，再扩展学习范围。
