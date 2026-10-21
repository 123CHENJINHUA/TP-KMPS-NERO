# 独立 PPO 实验验证：位置与力 v2

当前验证日期：2026-10-08。任务 ID：`NeroSingleBoltInsertPositionForce-v2`，58 维观测、6 维动作。沿用已有 Isaac Lab 环境和同一物理场景；未修改原装配程序和 USD 来源。

## 当前版本验证

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 法向力及摩擦力标定 | 两环境的旧位置基线 2/2 插入成功；正常插入轴向峰值约 7.27 N，侧向峰值约 1.26 N | `validation/force_probe_04/summary.json`、`forces.npz` |
| 位置／力控制行为与几何 | 8 项测试通过：过力退让、侧向力限制、自由空间不追力、力代价及孔贯穿 | `./ppo_assembly/run.sh test` |
| 两环境位置／力插入 | 2/2 成功，开爪撤离后均保持 10 秒；深度约 24.01 mm，位置变化范围约 0.081–0.099 mm | `validation/position_force_expert_01/summary.json` |
| 物理约束检查 | 4 个动态零件，重力开启，装配固定约束为 0 | 同上目录的 `configuration.json` |
| 16 环境 PPO 短更新 | 16×32×2=1,024 控制步，58 维力观测、6 维动作及损失有限，参数实际变化并保存模型 | `runs/position_force_smoke_01/summary.json` |
| 新增训练曲线 | 轴向力、侧向力、目标力、控制步峰值及力奖励代价均有日志 | 上述目录的 TensorBoard 与 `training_curves.png` |
| 旧模型兼容检查 | 在仿真启动前明确拒绝 v1 checkpoint，未覆盖旧模型 | `validation/old_model_rejection.log` |
| 新模型可视化回放与独立复位 | 加载 v2 模型，两个环境在第 200、220 步分别结束回合并复位，力读数和窗口提示正常 | `validation/position_force_gui_replay_01/summary.json` |

力反馈环境两个成功回合的合力峰值分别约 8.52 N、8.60 N，侧向峰值约 1.58 N、1.55 N。成功来自确定性的环境检查控制器，不代表短测 PPO 已收敛。

仅更新两次的 v2 模型回放 **0/2 成功**，都识别为夹持丢失；该检查验证加载、可视化和不同步复位，不能作为插入策略已经学会的证据。

接触力按孔坐标系分解，包含墙板、当前孔的 64 个摩擦环扇区及 keel 的法向力和摩擦力，排除手指夹持力。早期用父节点或宽泛通配符过滤时，PhysX 返回空读数或匹配数量错误；已改为每个碰撞体一对一过滤，标定检查确认接触力非零。未以夹爪夹持力、零读数或人为施加的外力代替插入反力。

这些力是仿真接触测量，尚未加入接触力矩或真实腕部传感器的重力、惯性补偿。目标力 2–8 N、软限 8 N、合力硬限 12 N 和侧向硬限 5 N 为当前场景设置；其他物体和摩擦模型需重新标定。

训练范围仍为左臂物理支撑下的单颗插入。未执行正式长训练，尚未证明完整双臂安装或最优轨迹。运行与奖励公式见 [README.md](README.md)。

## v1 历史结果

以下为引入力观测前的历史结果，旧实现快照位于 `legacy_v1/`，不能作为当前力反馈版本的训练成绩。

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
