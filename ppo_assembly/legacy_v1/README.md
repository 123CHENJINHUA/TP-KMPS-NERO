# Nero 单颗插入 PPO：Isaac Lab 独立实验

使用已有环境 `/home/cjh/anaconda3/envs/isaaclab`、Isaac Lab 2.3.2 和 RSL-RL PPO。`run.sh` 自动激活 conda，载入现有 Isaac Sim 的运行路径，无需修改环境或原程序。所有新代码、生成场景、训练模型、日志和检查结果都位于 `ppo_assembly/`。

## 当前训练范围

第一阶段学习右臂从孔前对齐并插入右上第 4 颗螺栓。左臂用真实夹持持续扶住 keel。每回合复位时，将两臂和零件放到已夹持、孔前的位置，再开始物理仿真；**拾取、远距离搬运、左臂松手和完整四颗安装尚未由这个策略学习**。

策略输出 5 个连续动作：向孔推进、水平调整、竖直调整、两轴倾角调整。倾角动作绕指令中螺栓夹持中心旋转，再换算成手腕目标，避免绕远处手腕旋转时带来很大的横向摆动。圆柱自由自转和夹爪开度不由本阶段策略控制。底层使用 Isaac Lab 的阻尼微分 IK，关节限速，避免全局 IK 分支切换造成大幅摆臂。

零件保持动态、重力开启，装配关节为 0；只在回合复位时设置初始零件位姿。插入过程使用接触与摩擦，策略阶段没有自动螺栓位姿跟随。实际插入条件持续通过后，环境自动打开右爪、撤离并观察自由保持；保持成功才获得成功奖励。

训练成功只代表**左臂仍扶持的单颗插入**，数据始终标记 `includes_pickup=false`、`reset_prepositioned=true`、`installation_complete=false`。

## 自己启动训练

在项目根目录运行。每次使用新输出目录。

```bash
# 资产已生成。只有首次缺少资产或明确要重新导出时才执行：
./ppo_assembly/run.sh prepare --headless

# 先检查物理环境，与 PPO 权重无关。
./ppo_assembly/run.sh check --controller expert --episodes 2 \
  --run-dir ppo_assembly/checks/expert_01

# 带 Isaac Sim 窗口的 PPO 训练，先用 1 个环境观察。
./ppo_assembly/run.sh train --num-envs 1 --iterations 2000 \
  --difficulty easy --run-dir ppo_assembly/runs/ppo_visual_01

# 关闭仿真画面，用 16 个环境并行训练。
./ppo_assembly/run.sh train --num-envs 16 --iterations 2000 \
  --difficulty auto --headless --run-dir ppo_assembly/runs/ppo_batch_01
```

`expert` 是用于检查环境的确定性反馈控制器，并不是训练完成的 PPO 模型。已有检查通过 3 次接触插入与松手保持；这不能说明 PPO 已经学会插入。正式训练请使用新目录，不要把 `runs/smoke_01` 当作可用装配策略。

`--iterations` 是 PPO 更新次数；默认每次收集每环境 32 个控制步。16 个环境训练 2000 次更新，对应 1,024,000 个环境控制步。它是预算，不是收敛保证。当前策略使用仿真位姿观测，尚未使用相机图像。

窗口中的提示显示 env 0 的插入深度、阶段、累计回合数和难度。左臂姿态稳定、右臂初期尝试不成功是可能的；看成功率及独立验证，不能只看训练奖励。

## 实时曲线和回放

在第二个终端执行：

```bash
./ppo_assembly/run.sh dashboard --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。查看奖励、成功率、损失、探索幅度、插入深度和轴线误差。TensorBoard 可同时比较多个训练目录。

```bash
# 输出一张训练曲线 PNG。
./ppo_assembly/run.sh curves ppo_assembly/runs/ppo_visual_01

# 用当前 checkpoint 可视化回放 5 回合。
./ppo_assembly/run.sh play \
  --checkpoint ppo_assembly/runs/ppo_visual_01/latest.pt \
  --episodes 5 --difficulty easy --seed 1000 \
  --run-dir ppo_assembly/evaluations/play_01

# 留出初始状态验证，默认每颗松手后观察 10 秒。
./ppo_assembly/run.sh play --headless \
  --checkpoint ppo_assembly/runs/ppo_batch_01/latest.pt \
  --episodes 100 --difficulty hard --seed 1000 --max-steps 50000 \
  --run-dir ppo_assembly/evaluations/heldout_01

# 从 checkpoint 继续训练，保存到新目录；iterations 为追加更新次数。
./ppo_assembly/run.sh train --headless --num-envs 16 \
  --checkpoint ppo_assembly/runs/ppo_batch_01/latest.pt \
  --iterations 1000 --difficulty auto \
  --run-dir ppo_assembly/runs/ppo_resume_01
```

checkpoint 每 50 次更新保存一次，`latest.pt` 同步更新。`Ctrl+C` 会保存当前模型。可视化回放采用策略确定性输出，不继续训练。训练期间允许读取已保存的 checkpoint 回放。

窗口里的暂停键可暂停物理过程；结束训练推荐终端 `Ctrl+C`。不要在训练期间增删物理对象或更改场景结构。若回放达到 `--max-steps` 仍未完成指定回合，`summary.json` 会标记 `budget_exhausted=true`，这不能作为完整评估结果。平均回合奖励要等首个回合结束后才出现在 TensorBoard 中。

## 难度、物理与奖励

| 难度 | 初始横向偏移 | 初始倾角 | 距插入目标的轴向距离 |
| --- | --- | --- | --- |
| easy | 每轴 ±0.25 mm | 每轴 ±0.003 rad | 32±2 mm |
| medium | 每轴 ±0.8 mm | 每轴 ±0.012 rad | 38±2 mm |
| hard | 每轴 ±1.5 mm | 每轴 ±0.025 rad | 44±2 mm |

每级有 64 个训练状态和 32 个独立生成的留出状态。`auto` 从 easy 开始，最近 50 个回合成功率达到 75% 后升级。这是有限状态库，不能作为任意初始位置泛化的证明。验证采用留出状态和不同采样 seed。

资产已经生成，可直接训练。`prepare --force` 的重新导出使用原装配目录中的 `validation/handoff_full_09/episode_001/trajectory.npz` 作为真实夹持姿态依据，读取现有场景和机械臂模型；请保留这些来源。生成文件写入本目录，不覆盖来源文件。

物理 120 Hz，控制 10 Hz。单步轴向最多 0.8 mm、横向每轴最多 0.4 mm、倾角每轴最多 0.003 rad。夹指最大力 10 N，摩擦孔、肩部及 keel 孔结构复用已经验证的装配模型；场景被导出到新的 `assets/cell.usdc`。

奖励包括插入进展、横向与轴线对齐、耗时及动作变化代价。掉落、越界、失去左臂扶持、非预期机械臂接触和松手保持失败会终止回合。实际插入条件包含深度、墙孔轴线误差、实际 keel 孔贯穿和肩部间隙；自由保持还检查爪子已打开、没有残留手部接触及位置稳定。

训练保持观察默认 3 秒；`check` / `play` 默认 10 秒，可用 `--retention-seconds` 延长。奖励不会以“到达目标 TCP”代替物理成功。

## 文件与结果

| 文件 | 用途 |
| --- | --- |
| `environment.py` | Isaac Lab 场景、控制、复位、奖励和成功判定 |
| `main.py` | PPO 配置、训练、checkpoint、回放及检查入口 |
| `task_math.py` | 批量位姿与孔贯穿计算 |
| `prepare.py` | 独立资产和初始状态生成 |
| `assets/fixture.json` | 起始夹持依据、坐标系和物理检查 |
| `runs/*/configuration.json` | 环境参数、来源 SHA-256 和算法参数 |
| `runs/*/episodes.csv` | 回合成功、原因、时间、深度与接触信息 |
| `runs/*/events.out.tfevents.*` | TensorBoard 数据 |
| `runs/*/model_*.pt`、`latest.pt` | 策略、价值网络、优化器状态和任务元数据 |
| `evaluations/*/summary.json`、`rollout.npz` | 策略评估与实际轨迹 |

需要调整参数时，环境参数在 `BoltInsertionCfg`，PPO 参数在 `main.py` 的 `agent_config`。先更改并验证单颗任务，再扩大并行数量。原装配脚本继续保留为成功率和完整流程基线。

本次仅做环境与短更新检查，未代替你执行长时间训练，也未宣称得到收敛或最优策略。实际验证结果见 [VALIDATION.md](VALIDATION.md)。
