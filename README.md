# TP-KMPs-nero-isaacsim

在 **Isaac Sim 5.1** 中采集 Nero 示教、训练 TP-KMP、测试新目标位置下的轨迹跟踪。

遥控器、键盘、双臂试教、自动采集、装配和训练的复制粘贴指令，统一见 [常用指令说明](COMMANDS.md)。

真机主从臂的单只从臂抓取/搬运示教、TP-KMP 训练及真机验证，见独立目录 [真机 TP-KMP 说明](real_tpkmp/README.md)。

RealSense D435i + 5×7 ChArUco 的眼在手上/眼在手外标定，见独立目录 [手眼标定说明](hand_eye_calibration/README.md)。

流程：**仿真 Nero → 采集实际末端轨迹和目标参考系 → TP-KMP 训练/泛化 → 仿真回放 → 误差报告**。

## 项目内容

- `assets/nero/`：从本机 nero_ws 模型生成的独立 URDF 和 STL，包含七轴 Nero 和双指夹爪。
- `assets/lula.yaml`：Isaac Sim 原生 Lula 逆运动学配置。
- `sim_bridge/app.py`：交互示教、自动示教、预测轨迹回放及物理跟踪检查。
- `sim_bridge/grasp.py`、`grasp_scene.py`：小方块抓取流程、物理场景、夹爪时序和抓取结果检查。
- `nero_bridge/data.py`、`model.py`：复用 TP-KMPs-nero 的数据格式和算法调用。
- `interactive_incremental_learning/`：原 TP-KMP 算法副本；无需修改原项目。
- `scripts/`：启动仿真和离线学习的命令。

常规仿真模式不需要启动 ROS、MoveIt、CAN 或真实机械臂；新增的 `--mode teach-sync` 使用两路 SocketCAN 被动读取示教臂，不导入真机驱动，也不向真机发送控制报文。
仿真入口生成的示教、技能和预测都带 `synthetic=true`，与真机执行拒绝规则兼容；`real_tpkmp/` 的真实采集使用独立格式与来源标记。

## 环境

本机默认使用：

```text
Isaac Sim: /home/cjh/Downloads/isaacsim/_build/linux-x86_64/release
算法 Python: /home/cjh/anaconda3/envs/tpkmp/bin/python
```

仿真使用 Isaac 自带 Python；训练使用已有科学计算环境，两者通过 NPZ 文件交换数据。
如需更换路径，设置 `ISAAC_SIM_PATH` 和 `TPKMP_PYTHON`。
算法依赖见 `requirements.txt`，没有额外安装新依赖。

请从本机图形桌面的终端启动交互模式。不要先 source 真机工作区或启动真实控制程序。
首次启动会编译 RTX 着色器，可能需要数分钟；无窗口测试可加 `--headless`。

## 先用自动采集跑通流程

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim

# 在 PhysX 仿真中实际运动，采集 5 条不同目标位置的接近轨迹
./scripts/sim.sh --mode collect --headless \
  --output-dir recordings/auto_01 --demos 5 --duration 8

# 训练 TP-KMP
./scripts/learn.sh train --demos recordings/auto_01/demo_*.npz \
  --output skills/reach_v1.npz

# 在第一条示教的目标位置预测
./scripts/learn.sh predict --skill skills/reach_v1.npz \
  --output output/reproduce.npz

# 在仿真中执行预测轨迹，输出误差
./scripts/sim.sh --mode replay --headless \
  --trajectory output/reproduce.npz --duration 8 \
  --output-dir output/reproduce_test
```

去掉 `--headless` 可以看到自动采集或回放过程。自动采集用于验证流程，使用 HOME 附近的小范围接近动作，
不会代替你自己的任务示教，也不代表抓取任务成功。

所有文件禁止覆盖；再次运行请更换 `auto_01`、技能文件名或结果目录。

## 自动抓取小方块并采集

使用新增的 **`--mode grasp`**。原来的 `collect` 仍然是自由空间接近示教。
抓取模式创建橙色 `/World/GraspCube`（边长 30 mm、质量 30 g）和有碰撞的小平台，
每次在 `[0.30, 0, 0.165]` m 附近随机改变方块的 XY 位置。
机械臂从方块上方开始，执行 **张开 → 下降 → 夹紧 → 抬升 60 mm → 保持**。

```bash
# 先做 2 次可视化抓取，检查真实接触效果
./scripts/sim.sh --mode grasp --demos 2 --duration 10 \
  --output-dir recordings/grasp_check_01

# 确认抓取效果后，自动采集 10 次；加 --headless 可关闭窗口
./scripts/sim.sh --mode grasp --demos 10 --duration 10 \
  --seed 42 --grasp-spread 0.012 --output-dir recordings/grasp_01
```

`--grasp-spread` 是 X、Y 各方向的随机半范围，单位为米，允许 `0..0.02`；
设为 `0` 可重复测试固定位置。`--duration` 至少 8 秒，同一组训练示教使用相同时间。

需要更丰富的位置和多组数据时，可以一次启动采集 **3 组，每组 12 次**：

```bash
./scripts/sim.sh --mode grasp --grasp-groups 3 --demos 12 --duration 10 \
  --grasp-x-range 0.28 0.34 --grasp-y-range -0.05 0.05 \
  --grasp-sampling stratified --seed 42 \
  --output-dir recordings/grasp_rich_01
```

X 覆盖 28–34 cm，Y 覆盖 −5–5 cm，方块中心高度仍为 16.5 cm，大小、方向和抓取动作保持一致。
这一区域约为原默认 XY 面积的 10.4 倍；通过了基于 URDF 的边界和边缘中点运动学检查，
每次执行仍会进行 Lula IK 预检查，并以物理反馈判断抓取成功。
`stratified` 将每个轴划成 12 个区间，每个区间采一个随机位置，再随机配对 X/Y，减少随机位置扎堆。
这保证每个轴的覆盖，不代表遍历所有 12×12 个二维网格。`random` 保留原来的均匀随机抽样。

- `--grasp-x-range MIN MAX` 和 `--grasp-y-range MIN MAX` 分别覆盖对应轴的 `--grasp-spread` 设置。
- 支持的范围限制是 X `[0.28, 0.34]`、Y `[-0.05, 0.05]` m；可以缩小范围，不能越界。
- `--grasp-groups 3` 在同一个仿真进程中连续采集，种子依次为 42、43、44。`--demos 12` 指每组尝试次数，总计 36 次。
- 多组数据写入 `group_001/`、`group_002/`、`group_003/`，各自包含示教和 `summary.json`。
- 每组的 `positions.json` 保存计划位置、抽样方法、边界和种子；`attempt_*.json` 同时保留计划位置和稳定后的实际位置。
- 根目录的 `summary.json` 汇总所有组，`physics.json` 保存共用物理配置。单组采集仍使用原来的目录布局。

例如，用前两组训练，将第三组的位置留作泛化验证：

```bash
./scripts/learn.sh train \
  --demos recordings/grasp_rich_01/group_001/demo_*.npz \
          recordings/grasp_rich_01/group_002/demo_*.npz \
  --output skills/grasp_rich_v1.npz --samples 100 --components 8
```

每次初始化会将机械臂复位到预抓取位姿，并将方块放回平台；初始化不计入示教。
录制开始后，方块只通过重力、碰撞和摩擦运动，没有吸附、固定约束或跟随末端的位姿赋值。
程序检查整条路径的 IK 可达性，但没有通用避障规划；请保持这个小平台场景不变。

输出目录必须是新的或空的：

- `demo_XXX.npz`：通过抓取检查的示教，包含实际 TCP、七轴关节、两个夹爪关节、夹爪目标开度、方块位姿和动作阶段。
- `failed_XXX.npz`：有足够采样数据的失败尝试，仅供诊断，不应加入训练。
- `attempt_XXX.json`：每次尝试的结果或失败原因。
- `positions.json`：这一组的全部计划位置、范围、抽样方式和种子。
- `physics.json`：实际配置的夹爪驱动类型、增益、力上限和接触参数；包含导入时的原驱动类型。
- `summary.json`：成功数量和所有尝试结果；任何一次失败都会使命令以失败状态退出，已经成功保存的数据仍保留。

夹爪使用明确的 `force` 驱动：刚度 600 N/m、阻尼 10 N·s/m、每指最大 10 N。
两指和方块的静摩擦系数为 1.2、动摩擦系数为 1.0，同时启用关节接触最后求解。
`Position` 目标类型并不决定驱动是 `force` 还是 `acceleration`，两者相同数值的增益不代表相同夹持力；
参见 NVIDIA 的 [URDF 驱动说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html)。
如果失败，先查看报告中的 `reason`、`pre_lift_finger_gap_m` 和 `max_cube_lift_m`：
抬升前间距约 30 mm、随后方块掉落而夹爪闭合到底，说明发生了夹持丢失，优先检查驱动和摩擦接触。
正常的抓取判定失败会打印 `GRASP_COLLECTION_FAILED` 并以状态码 2 退出，不输出 Python 异常堆栈。

成功要求在最后保持阶段持续抬高方块至少 39 mm，方块位于两指间，夹爪没有空夹到底，
相对滑动不超过 8 mm，并保持至少 0.5 秒。这个检查基于实际仿真反馈，不以机械臂抬起或闭合命令作为成功依据。
任务参考系固定为方块抓取前的实际位姿；被抓起后的方块位姿单独记录，不改变任务参考系。

训练及可视化回放：

```bash
./scripts/learn.sh train --demos recordings/grasp_01/demo_*.npz \
  --output skills/grasp_v1.npz --samples 100 --components 8
./scripts/learn.sh predict --skill skills/grasp_v1.npz \
  --output output/grasp_reproduce.npz
./scripts/sim.sh --mode replay --trajectory output/grasp_reproduce.npz \
  --duration 10 --output-dir output/grasp_replay_01
```

至少需要 2 条成功示教。不要混合抓取和旧的接近示教；训练接口会拒绝混合类型和标记为失败的数据。
TP-KMP 学习 TCP 轨迹，夹爪目标开度按照示教相位同步保留，不额外训练抓取策略。
回放自动创建实体方块和平台，执行夹爪时序，输出 `grasp_tracking.npz`、`grasp_result.json` 和跟踪误差报告。
回放时长不得短于采集时长；轨迹跟踪和抓取检查都通过才报告成功。
预测轨迹的平滑可能影响接触位置，必须检查 `grasp_result.json`，不能仅凭训练完成判定成功。

本模式目前只针对小范围内竖直放置的小方块，未实现任意物体、障碍物场景或放置任务。
基础抓取已由用户实测通过（`recordings/grasp_check_02` 为 2/2 成功）。扩展位置范围通过了离线数据和运动学测试，
当前开发环境缺少 NVIDIA 设备，尚未采集扩展范围的物理抓取数据。

## 抓取 keel 模型

物体库位于 `assets/objects/`，已加入 `keel` 的原始 STL/USD、米制仿真模型和抓取配置。
用户确认尺寸为 150×98×23 mm，仿真质量设为 80 g；沿两条凸壁的外侧夹持，名义夹持宽度 38 mm。

```bash
./scripts/sim.sh --mode grasp --object keel --grasp-spread 0 \
  --demos 2 --duration 10 --output-dir recordings/keel_view_01
```

已完成两次无窗口物理抓取，均抬升约 59 mm 并稳定保持 1 秒。
结果在 `recordings/keel_check_01/summary.json`，两条成功示教也保存在该目录。

### 随机位置自动采集

在项目根目录执行。单组生成 12 个位置，自动放置 keel、执行抓取并记录数据：

```bash
./scripts/sim.sh --mode grasp --object keel \
  --demos 12 --duration 10 --grasp-spread 0.012 \
  --grasp-sampling stratified --seed 42 \
  --output-dir recordings/keel_random_01
```

`stratified` 使用分层随机采样，使位置分布更均匀；改为 `random` 可使用独立均匀随机采样。
上述范围为原抓取位置 X/Y 各 ±12 mm，即 X ∈ [0.288, 0.312] m、Y ∈ [-0.012, 0.012] m；
物体高度和朝向保持不变。每次动作时长为 10 秒，另有初始化和检查时间。
本次 `recordings/keel_random_groups_01/summary.json` 记录该范围内 36/36 次抓取成功；新增采集仍会逐次进行 IK 和物理抓取检查。

一次采集 3 组、每组 12 次（共 36 次尝试）：

```bash
./scripts/sim.sh --mode grasp --object keel \
  --grasp-groups 3 --demos 12 --duration 10 --grasp-spread 0.012 \
  --grasp-sampling stratified --seed 42 \
  --output-dir recordings/keel_random_groups_01
```

- 多组结果分别写入 `group_001/`、`group_002/`、`group_003/`，随机种子依次为 42、43、44。
- `--demos` 是每组尝试次数，不保证全部成功。成功示教保存为 `demo_*.npz`，失败记录为 `failed_*.npz`；训练时只选择成功示教。
- 每组的 `positions.json` 保存采样位置，`summary.json` 保存抓取结果；多组根目录另有总汇总。
- 默认显示仿真窗口；添加 `--headless` 可无窗口采集。
- 输出目录必须是新目录或空目录。再次采集不同数据时，同时更换 `--seed` 和 `--output-dir`。

### 训练 keel 抓取技能

在项目根目录执行。对于上述三组采集，用前两组的成功示教训练，第三组保留用于未参与训练的位置验证：

```bash
./scripts/learn.sh train \
  --demos recordings/keel_random_groups_01/group_001/demo_*.npz \
          recordings/keel_random_groups_01/group_002/demo_*.npz \
  --output skills/keel_v1.npz --samples 100 --components 8
```

本次 `recordings/keel_random_groups_01/summary.json` 记录 36/36 次采集成功，
因此上述命令使用 24 条示教，`group_003/` 的 12 条示教不参与训练。
脚本默认使用 `/home/cjh/anaconda3/envs/tpkmp/bin/python`，无需另外激活 conda 环境或启动 Isaac Sim。
`--samples 100` 将每条示教重采样为 100 个点，`--components 8` 设置混合模型的分量数。
训练完成后输出 `Saved skills/keel_v1.npz`，技能文件保存在该路径；训练完成不等于通过物理抓取验证。
已有同名技能时请更换 `--output` 文件名，以保留旧版本。

如果采集的是前面的单组示例，改用：

```bash
./scripts/learn.sh train --demos recordings/keel_random_01/demo_*.npz \
  --output skills/keel_single_v1.npz --samples 100 --components 8
```

只使用 `demo_*.npz`，不要加入 `failed_*.npz`，也不要混入原方块的数据。
如使用了自定义采集目录，请同步替换命令中的路径。

### 验证 keel 的位置泛化能力

先完成上面的前两组训练，确保 `skills/keel_v1.npz` 存在，且第三组没有加入训练。
在项目根目录依次执行下面三步，以第三组第一个未参与训练的位置为例。

1. 导出测试目标位置（必须先生成 JSON，避免找不到目标文件）：

```bash
./scripts/learn.sh frames \
  --demo recordings/keel_random_groups_01/group_003/demo_001.npz \
  --output output/keel_validation_001_target.json
```

2. 使用学到的技能，为这个目标位置预测轨迹：

```bash
./scripts/learn.sh predict --skill skills/keel_v1.npz \
  --frames output/keel_validation_001_target.json \
  --output output/keel_validation_001_prediction.npz
```

3. 在 Isaac Sim 窗口中执行预测轨迹，观察夹取、抬升和保持过程：

```bash
./scripts/sim.sh --mode replay --object keel \
  --trajectory output/keel_validation_001_prediction.npz \
  --duration 10 --output-dir output/keel_validation_001_replay
```

这里仅从测试示教读取目标位姿，不直接回放测试示教的运动轨迹。
回放会加载 keel 模型，并检查真实物理抓取结果。添加 `--headless` 可无窗口验证。

查看 `output/keel_validation_001_replay/` 下的报告：

- `grasp_result.json`：`passed` 表示物理抓取是否通过，包含抬升、保持等检查结果。
- `metrics.json`：`passed` 表示末端轨迹跟踪是否通过，包含位置及姿态误差。
- 两个报告都为 `passed: true` 才算该次验证通过。预测文件生成成功不代表抓取成功。
- 如果在初始化或 IK 预检查时失败，可能尚未生成报告；应查看终端错误，并将该次计为未通过。

要测试全部第三组位置，将上述三步里的 `demo_001` 改为 `demo_002` 至 `demo_012`，
同时将所有输出路径里的 `keel_validation_001` 对应改为 `keel_validation_002` 至 `keel_validation_012`。
统计通过次数 / 12，得到这一组的验证通过率；不要只报告成功的样本。
重复同一个测试时也要换一套新的输出路径，避免已有 JSON 或报告导致文件冲突。

这验证的是当前位置范围内、同一个 keel 物体且姿态不变时的位置泛化能力，
不代表已验证任意物体、旋转姿态或范围外的位置。若第三组也参与过训练，应另采新位置作为测试集。

keel 数据应单独训练，不能直接把原方块技能当作 keel 技能使用。训练文件会保留物体名称，
回放会自动加载对应模型。详见 [物体目录说明](assets/objects/README.md)。

## 双 Nero 真机硬件读取检查

`hardware_checks/` 提供 `can0` / `can1` 被动收帧和 pyAgxArm 双 Nero / AGX 夹爪反馈检查脚本。
这些脚本不向真机发送运动或模式切换命令；依赖和报告都留在项目目录内。
运行方法见 [硬件检查说明](hardware_checks/README.md)。

## 固定支架上的双机械臂 Xbox 控制

连接手柄后，在项目根目录运行：

```bash
./scripts/sim.sh --mode dual --control xbox
```

场景生成两台独立的 Nero、各自的固定支架与夹爪。默认支架顶面离地 **0.5 m**，
两台基座相距 **0.9 m**，分别位于 `[0, -0.45, 0.5]` 和 `[0, 0.45, 0.5]` m；
左臂从原来的 +90° 顺时针旋转 90°，右臂从原来的 -90° 逆时针旋转 90°；现在两台底座的 Z 轴朝向角均为 0°，机械臂都朝世界坐标 +X 方向。
第 1 台为红色末端目标，第 2 台为蓝色末端目标；面板始终显示当前控制的机械臂。
机器人基座通过 URDF 固定基座设置锁定，支架使用固定碰撞体。

手柄沿用下节的摇杆、扳机、A/B 和十字键操作。**按一次 Y 在两台机械臂之间切换**，
初始控制第 1 台。切换后松开所有按钮和扳机、摇杆回中，面板显示 `active` 后再移动。
切换不会移动另一台机械臂：两台机械臂各自保存目标位姿和夹爪开度。
Menu（三横线）暂停/启用手柄控制；面板的 **Reset Both Arms** 将两台机械臂恢复到初始位姿。
这是手柄依次控制两台机械臂的仿真模式，尚未加入双臂协同轨迹录制；真机示教同步见下一节。

如需调整支架尺寸（单位：米），例如：

```bash
./scripts/sim.sh --mode dual --control xbox \
  --dual-spacing 1.0 --dual-mount-height 0.6
```

`--dual-spacing` 允许 0.6–1.5 m，`--dual-mount-height` 允许 0.3–1.0 m。
先确认两台手臂和支架之间的间隙，再操作末端靠近对方；当前 IK 会拒绝不可达目标，
但没有双臂互相避障规划。手柄断连、时间轴暂停或切换后都需先回中才能继续控制。

## 双路真机示教同步到仿真

先按 [硬件检查说明](hardware_checks/README.md) 确认 can0 和 can1 均为 UP，且 `read_dual_nero.py` 能连续读到两路 `leader` 和 `leader_gripper`。在项目根目录启动 Isaac Sim：

```bash
./scripts/sim.sh --mode teach-sync
```

此模式复用上面的双臂和固定支架场景，默认 **can0 → Nero 1（红色）**、**can1 → Nero 2（蓝色）**。无需 Xbox，也不会订阅键盘控制；机械臂本体只接收 SocketCAN 报文。示教臂的七轴角度直接映射到仿真 `joint1`–`joint7`，0x159 夹爪宽度目标映射到两根夹指。界面中的 **Sync enabled** 可随时暂停或恢复同步，关闭窗口即可退出。两路开始同步前都必须收到完整的关节及宽度模式夹爪帧；CAN 数据超过 0.25 秒未更新时，对应仿真臂保持上次目标。未启用 CAN 或启动 5 秒后仍缺少报文会报错退出。

```bash
# 需要交换左右臂时
./scripts/sim.sh --mode teach-sync --teach-left-can can1 --teach-right-can can0

# 先做 120 帧无界面检查
./scripts/sim.sh --mode teach-sync --headless --steps 120
```

仿真关节位置目标按 `--joint-speed` 限速；若跟随明显滞后，可适当提高此参数。示教器角度若超出仿真 URDF 限位，界面与终端会标出关节，并将仿真目标限制在模型范围内；这部分姿态无法做到完全重合。试教夹爪的 0x159 是**目标开度回显**，不是夹爪实际位置或抓取力反馈。示教同步模式会把最新夹爪目标直接交给仿真关节驱动，不再使用 Xbox 模式的渐进开合限速；面板显示仿真夹爪的实际/目标开度（mm），便于判断剩余延迟是否来自物理驱动。此模式只做实时同步预览，尚不录制示教数据。

首次运行请分别轻微移动 can0、can1 的一个关节，再分别开合试教夹爪，确认只有对应的仿真臂和夹爪响应。如有固定的角度偏差，先记录各关节的偏差和方向，再做零位标定；不要直接改动真机控制参数。

左右试教夹爪均采用直接开度映射：示教器开度增大时，对应仿真夹爪张开；示教器开度减小时，对应仿真夹爪闭合。硬件方向已修正，右侧不再进行软件反转。开度仍限制在 0–100 mm。

## 对照机械臂在 Isaac Sim 中摆放并保存场景

直接启动双臂示教同步，机械臂、支架和额外场景会显示在**同一个舞台、同一套世界坐标**里：

```bash
./scripts/sim.sh --mode teach-sync
# 已有保存的环境时：
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/workcell_已保存文件.usd
```

在 Isaac Sim 的 Stage 面板展开 `/World/UserScene/Environment`，把桌子、障碍物或待抓取物体放在**这个节点下面**，边看机械臂边调整位置。新建物体若默认出现在 `/World` 下，可在 Stage 面板把它拖入 `Environment`；自定义材质也应放在 `/World/UserScene` 下，才会一起导出。可以先暂停时间轴再摆放，避免物体在物理仿真中移动。固定物体需要 Collider；会被抓取、会掉落的物体还需要 Rigid Body 和 Collider。

调整完点击 **Dual Nero Teaching Sync** 面板中的 **Save Environment USD**。程序会在本项目的 `assets/scenes/` 下生成一个带时间戳的 `workcell_*.usd`，面板和终端都会显示完整路径。导出仅包含 `/World/UserScene` 下的环境及本次摆放修改，**不包含机械臂、支架、地面或 TCP 目标块**。下一次运行时把该文件传给 `--scene-usd`，即可在同一位置重新加载；继续调整后再次点击按钮会保存一个新版本。Xbox 双臂模式也有相同按钮，并支持 `--mode dual --scene-usd ...`。

不要用 Isaac Sim 的 **File → Save As** 将运行中的整个双臂舞台保存后直接当作 `--scene-usd`，那会把脚本生成的机械臂和支架也保存进去。放在 `/World/UserScene/Environment` 以外的新物体不会进入 **Save Environment USD** 导出的文件。如需单独编辑环境文件，也可以打开 [环境模板](assets/scenes/workcell_template.usda)；但对照机械臂摆放时，推荐上面的组合舞台工作流。物体的随机摆放、自动复位和示教录制仍需相应脚本。

### 0928 场景中的 keel 抓取物理设置

`0928.usd` 的 `/World/Environment/Geometry`（蓝色 keel）原来只有 Collider，没有 Rigid Body，因此是静态碰撞体，夹爪不能把它搬动。`/World/Environment/Cube`（灰色桌面）原来没有 Collider，不能支撑动态物体。修复版本保存在 `assets/scenes/0928_grasp.usd`，原文件保留。启动：

```bash
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd
```

修复版本只给 keel 网格添加动态刚体，使用原物体配置中的 **80 g 仿真假定质量**（不是实测质量）；给灰色桌面添加静态碰撞体。零件使用 convexDecomposition，桌面使用 convexHull，接触距离为 1 mm，静/动摩擦系数为 1.2/1.0。摆放坐标保留；零件初始悬空约 30 mm，播放后会在重力作用下落到桌面。检查记录见 `assets/scenes/0928_grasp_check.json`。

双臂模式的夹爪现在也使用单臂抓取已经采用的物理参数：力驱动、每指最大 10 N、刚度 600 N/m、阻尼 10 N·s/m，指尖碰撞距离 1 mm，凸分解碰撞和高摩擦材质。示教夹爪仍直接发送最新开度目标。

以后在 Isaac Sim 中添加模型后，选中**零件自己的节点**，在右键 Add → Physics 菜单添加 Rigid Body 和 Collider；确认 Rigid Body Enabled 打开、Kinematic Enabled 关闭。固定桌面只添加 Collider。不要把 Rigid Body 添加到同时包含桌面和零件的 `Environment` 上，否则它们会成为同一刚体。设置好后点击 **Save Environment USD** 保存新版本。keel 的外形为 150×98×23 mm，夹爪最大开度 100 mm；抓取时应对准较窄的凸起/夹持区域，避免沿 150 mm 长边夹取，98 mm 宽边也只有很小的间隙。

在双臂场景中添加相机时，可先暂停时间轴，在 `/World/UserScene/Environment` 下使用 **Create → Camera** 创建相机，再打开 **Tools → Sensors → Camera Inspector**，选择相机并点击 **Create Viewport** 显示画面。点击 **Save Environment USD** 会保存该相机及其摆放参数；相机视口窗口可在下一次启动后重新打开。

### keel 双臂装配场景

在 `0929_grasp.usd` 的桌面和可抓取 keel 上新增了 [装配场景](assets/scenes/keel_assembly.usda)：桌面远侧有一面宽 360 mm、高 320 mm、厚 30 mm 的固定混凝土板，正面朝向两台机械臂的基座（世界 +X 方向），墙上四个真实贯通的安装孔与 keel 的四孔位置相配；桌面右侧有带四个定位孔的矮螺丝架，架上竖放四枚可抓取的动态螺丝。原场景文件未改动。启动命令：

```bash
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/keel_assembly.usda
```

keel 左侧两孔直径约 12 mm、右侧两孔约 10.5 mm，孔中心相距 123 × 68 mm。演示螺丝的杆径为 8 mm，墙孔直径为 9 mm，螺丝头直径 18 mm、高 14 mm，便于夹爪夹持。安装时需要将 keel 从水平姿态转为竖直，四孔对准墙孔，再沿世界 -X 方向插入螺丝。场景已配置刚体与碰撞，但目前没有螺纹、旋紧动作或锁紧约束，插入后不会自动形成可靠的固定连接。

几何参数在 [场景生成脚本](scripts/build_keel_assembly_scene.py) 顶部以及 `make_wall`、`make_rack` 中；需要调整位置或尺寸时，用 Isaac Sim 的 Python 运行 `scripts/build_keel_assembly_scene.py --force` 重新生成，或直接在组合舞台编辑后通过 **Save Environment USD** 另存。重新生成会覆盖生成的 `keel_assembly.usda`。

程序在每次下发关节目标前检查双臂控制和重力补偿的物理视图。场景编辑或停止后重新播放导致视图失效时，会重新初始化；若旧视图已被清理，会按保存的机械臂路径重建，并恢复夹爪力限制和关节增益。恢复成功时终端打印 `DUAL_PHYSICS_RESTORED arms=2`。添加相机无需关闭程序；停止状态下等待重新播放，暂停不会强制跳回 Home。

## 两个机械臂末端的相机

`dual` 和 `teach-sync` 模式默认在两臂末端各创建一台相机，GUI 启动后自动打开 **Nero 1 Wrist Camera**、**Nero 2 Wrist Camera** 两个画面窗口，原来的整体场景视图仍可使用：

```bash
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd
# Xbox 双臂控制也会自动创建：
./scripts/sim.sh --mode dual --scene-usd assets/scenes/0928_grasp.usd
```

相机路径分别为 `/World/nero/gripper_base/WristCamera`、`/World/nero2/gripper_base/WristCamera`。相机直接继承末端的运动，不添加刚体、碰撞体或额外固定关节，也不依靠逐帧改写世界位姿来跟随。父节点是 `gripper_base`，因此夹爪开合不会改变相机的安装位姿。

长期使用的参数放在 [末端相机配置](assets/cameras/wrist_cameras.json)。修改 `left`、`right` 中的 `translation_m`（米）和 `rotation_xyz_deg`（绕局部 X/Y/Z 的欧拉角，度），再重新启动即可。两臂独立配置，所有平移和旋转都相对于各自的 `gripper_base`，不是世界坐标。默认两侧都是位置 `[0.05, 0, 0.02]`、角度 `[180, -25, 0]`，把相机放在末端旁边并朝向手指前方；这是调试位姿，不代表真实硬件安装标定。

JSON 示例（只展示需要调整的安装部分）：

```json
"left": {
  "translation_m": [0.05, 0.0, 0.02],
  "rotation_xyz_deg": [180.0, -25.0, 0.0]
},
"right": {
  "translation_m": [0.05, 0.0, 0.02],
  "rotation_xyz_deg": [180.0, -25.0, 0.0]
}
```

JSON 中相机朝向按 USD Camera 约定解释：自身 `-Z` 为光轴，`+Y` 为画面上方。公共参数 `resolution` 默认 `[640, 480]`；`focal_length_mm` 是焦距（毫米），`horizontal_aperture_mm` 是成像面宽度（毫米），程序会换算到 USD 的镜头单位；`clipping_range_m` 默认 `[0.01, 10]`。这两个窗口显示实时仿真图像，目前不会自动录制 RGB 或深度。

可临时在 Isaac Sim 中选中 WristCamera 调整 Transform 并查看效果，然后把确定的局部位姿写回配置文件。**Save Environment USD 只保存环境，末端相机的长期安装参数保存在 JSON 中，每次启动重新创建。**环境中之前手动添加的固定 Camera 不会被删除。

```bash
# 使用项目内另一份配置：
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd \
  --wrist-camera-config assets/cameras/wrist_cameras.json
# 保留相机节点，暂时不打开相机画面窗口：
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd --no-camera-views
# 不创建配置中的相机：
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd --no-wrist-cameras
```

无界面启动不会创建相机窗口。需要重新打开画面时，在 Camera Inspector 中刷新并选择对应 WristCamera。

继续添加相机时，在同一份 JSON 的 `extra_cameras` 列表中增加条目，保留原有 `left`、`right` 和公共参数。例如把默认的 `"extra_cameras": []` 替换为：

```json
"extra_cameras": [
  {
    "name": "CameraSide",
    "parent": "left",
    "translation_m": [-0.05, 0.0, 0.02],
    "rotation_xyz_deg": [180.0, 25.0, 0.0],
    "show_view": true
  },
  {
    "name": "CameraTop",
    "parent": "environment",
    "translation_m": [-0.35, 0.0, 1.3],
    "rotation_xyz_deg": [0.0, 0.0, 0.0],
    "show_view": true
  }
]
```

| `parent` | 挂载节点 | 位姿的参考坐标系 |
|---|---|---|
| `left` | `/World/nero/gripper_base` | 左臂末端，随左臂移动 |
| `right` | `/World/nero2/gripper_base` | 右臂末端，随右臂移动 |
| `environment` | `/World/UserScene/Environment` | 环境节点，通常固定在场景中 |

每个条目创建一台额外相机，原有两台末端相机继续保留。`name` 使用字母、数字、下划线，首字符不能是数字，同一父节点下不能重名；`left`、`right` 下的 `WristCamera` 已被默认相机占用。位置单位为米，角度单位为度，均相对于表中的父节点。环境相机的 `[0, 0, 0]` 旋转使光轴指向该节点的局部 `-Z`，可用于向下看桌面。示例位置需要按自己的场景调整。

额外相机默认继承公共镜头参数；也可以在条目中单独设置 `resolution`、`focal_length_mm`、`horizontal_aperture_mm` 和 `clipping_range_m`。`show_view: false` 保留相机节点但不自动打开其窗口。`--no-camera-views` 关闭所有配置相机的自动窗口；`--no-wrist-cameras` 则禁用整份相机配置，包括新增的环境相机。

已提供可直接运行的 [四相机示例配置](assets/cameras/wrist_cameras_extra_example.json)，包含两台默认末端相机、左臂侧面相机和环境俯视相机：

```bash
./scripts/sim.sh --mode teach-sync --scene-usd assets/scenes/0928_grasp.usd \
  --wrist-camera-config assets/cameras/wrist_cameras_extra_example.json
```

修改 JSON 后重启程序生效，无需修改 Python 创建代码。环境下的额外相机可随 **Save Environment USD** 保存，但启动时同名相机的位姿和镜头参数仍以 JSON 为准；末端相机仍只通过 JSON 保留安装参数。新增相机显示实时画面，不会自动录制图像。多开画面会增加渲染开销，可降低分辨率或关闭不需要的窗口。


### 相机外壳、局部坐标轴和拍摄方向

[查看相机外壳与方向箭头的实际渲染预览](assets/cameras/camera_visualization_preview.png)。

程序创建的每台相机现在默认带一个类似 RealSense 的横条外壳，尺寸为 90×25×20 mm，正面有三个镜头示意。这是用于确认安装位姿的可视化模型，不是某款 RealSense 的准确 CAD，也没有新增深度传感器或碰撞体。光学原点仍为原来的 Camera 节点，外壳位于镜头后方。

相机下面的 `Visualization` 子节点包含外壳和方向箭头，直接继承相机的局部位姿。旋转 Camera 或移动机械臂时，它们会一起转动；无需选中相机。红、绿、蓝箭头分别表示相机的局部 +X、+Y、+Z，黄色箭头表示实际拍摄方向 −Z。黄色箭头放在镜头旁边，便于从整体场景判断相机朝向。

请在 Stage 中选中 **Camera 节点本身**调整 Transform，而不是单独调整 `Visualization`、`Body` 或 `Axes`。可展开以下路径观察：

- `/World/nero/gripper_base/WristCamera/Visualization`
- `/World/nero2/gripper_base/WristCamera/Visualization`

JSON 的公共显示设置为：

```json
"visualization": {
  "show_body": true,
  "show_axes": true,
  "axis_length_m": 0.08
}
```

`show_body` 控制外壳显示，`show_axes` 控制全部方向箭头，`axis_length_m` 为箭头长度（米，默认 8 cm）。也可以在 `left`、`right` 或某个 `extra_cameras` 条目内加入 `visualization` 覆盖对应相机的显示参数；未指定的字段继承公共配置。修改后重启生效。在 GUI 中临时隐藏时，可选中对应的 `Body` 或 `Axes`，将 Visibility 改为 invisible。

外壳和箭头是场景中的可见几何，也可能出现在其他相机的画面中。采集纯净图像时可将公共 `show_body`、`show_axes` 都设为 `false`，并删除逐相机的显示覆盖设置；相机节点和图像功能会保留。现有环境相机保存规则不变，相机长期参数仍以 JSON 为准。

### 保存和恢复窗口布局

当前通过 Python 启动时，顶部 **Layouts** 菜单可能只有 UI Toggle Visibility、Fullscreen Mode，没有 Save/Load 入口。双臂控制面板现在提供独立按钮：

- **Save UI Layout**：保存当前窗口排列、停靠和尺寸到 `assets/layouts/dual_camera_layout.json`；再次保存会更新同一份布局。
- **Load UI Layout**：等相机窗口和控制面板创建完后，点击恢复之前保存的排列。此操作不强制关闭新增的相机/控制窗口。

`dual` 和 `teach-sync` 两种模式都有这些按钮；修改程序后重新启动可见。窗口布局通过 JSON 保存，与场景 USD、相机安装位姿分别保存。

如果正在运行的旧窗口已调整好布局，可先打开 **Window → Script Editor**，粘贴以下代码并点击 Run（Ctrl+Enter），无需重启即可保存当前布局：

```python
from sim_bridge.ui_layout import save_layout
save_layout()
```

当前会话恢复布局也可在 Script Editor 中运行：

```python
from sim_bridge.ui_layout import load_layout
load_layout()
```

需要保存多个布局时，为函数传入项目内不同文件名，例如 `save_layout("assets/layouts/dual_camera_02.json")`，恢复时使用相同路径。新增布局文件仍存放在本项目中。

## 使用 Xbox 控制机械臂

连接 Xbox 手柄（建议先使用 USB），在项目根目录启动：

```bash
./scripts/sim.sh --mode interactive --control xbox \
  --output-dir recordings/xbox_01
```

交互模式现在默认使用 Xbox；`--control xbox` 可省略。Xbox 模式不订阅机械臂键盘控制事件。
窗口打开后保持时间轴播放，松开所有摇杆、扳机和按钮，待面板显示 `active` 后操作。
通过 Linux `/dev/input/js*` 读取手柄，不需要启动 ROS、`joy_node` 或 `joy_pkg`，也不需要安装 pygame。

映射依据 `Auto_control_ws/src/remote_ctl/joy_pkg/joy_pkg/joy_trans.py`、`joy2robot.py` 和
`gripper_forwarder.py`。平移和旋转沿当前 **TCP（gripper_base）坐标系**，随末端姿态变化，
不是固定世界坐标系。因此 LT 的 +Z 不一定是画面中的向上。

| Xbox 操作 | 动作 |
|---|---|
| 左摇杆上 / 下 | TCP X 正 / 负方向平移 |
| 左摇杆右 / 左 | TCP Y 正 / 负方向平移 |
| LT / RT | TCP Z 正 / 负方向平移；同时按下时相互抵消 |
| 右摇杆右 / 左 | 绕 TCP X 负 / 正方向旋转 |
| 右摇杆上 / 下 | 绕 TCP Y 正 / 负方向旋转 |
| A / B | 绕 TCP Z 正 / 负方向旋转；松开即停止增加旋转目标 |
| 十字键左 / 右 | 夹爪闭合 / 打开一步，步长为全行程的 3%；按住每 0.2 秒重复 |
| 十字键上 | 完全闭合 |
| 十字键下 | 设置为 35% 开度（原包的 65% 闭合量） |
| Menu / Start（三横线） | 暂停 / 启用手柄控制，重新启用后需回中并松开所有按钮 |

双臂模式中 Y 切换机械臂；单臂模式中 Y 不执行动作。原包的 X、LB/RB 控制外部电机，当前场景中不使用。
原包夹爪的 100% 表示闭合，这里保留其操作方向，适配 Nero 的 0–100 mm 总开度。
启动仍沿用 Nero 的半开状态；不会仅因接上手柄而自动闭合。

默认最大平移速度 **30 mm/s**、各轴旋转速度 **15°/s**、死区 **0.12**，按仿真步长积分。
保留参考代码的轴方向，但采用适合当前仿真的速度，不直接使用参考程序的速度倍率。
例如调整速度、死区或指定多个手柄中的一个：

```bash
./scripts/sim.sh --mode interactive --control xbox \
  --gamepad-device /dev/input/js0 \
  --gamepad-speed 0.02 --gamepad-angle-speed 10 --gamepad-deadzone 0.15 \
  --output-dir recordings/xbox_02
```

断连、暂停时间轴、复位或关闭手柄控制后，需要回中才能重新控制。
松手或断连后停止增加末端目标，机械臂仍可完成已下发的小步目标；已经下发的夹爪开度目标保留。
不可达的 IK 目标不会累计到红色控制点，录制期间 IK 失败会丢弃该条示教。

在采集面板用鼠标点击 **Record / Save / Discard / Reset Home**，用手柄完成运动，
不需要键盘操作。此交互模式仍录制 TCP 和机械臂关节，尚不录制夹爪动作，
不能用它替代前面的 `--mode grasp` 物理抓取示教采集。

若面板提示未连接，可先检查 `ls /dev/input/js*`。没有设备时检查 USB 连接及系统是否识别手柄；
若系统已经识别手柄但没有 `js*`，可加载 Linux joydev 驱动：`sudo modprobe joydev`。
若提示 `Permission denied`，为当前桌面用户配置对应设备的读取权限，无需以 root 启动 Isaac Sim。
程序支持通过驱动语义代码识别常见 USB 和蓝牙布局；不支持的布局会明确提示，建议先改用 USB。

已通过 30 项离线测试及无手柄条件下的 Isaac Sim 30 步启动检查。
开发时未检测到已连接的 Xbox，实际硬件按键与蓝牙驱动兼容性仍需连接后确认。

## 在 Isaac Sim 窗口中手动采集

启动：

```bash
./scripts/sim.sh --mode interactive --control keyboard --output-dir recordings/manual_01
```

窗口中有 Nero、地面和两个标记，以及 **Nero TP-KMP Collection** 面板：

| 场景对象 | 用途 |
|---|---|
| `/World/TCPGoal`，红色小方块 | 期望的末端位姿；Nero 通过 IK 和关节位置驱动跟随它 |
| `/World/Target`，绿色方块 | TP-KMP 任务参考系；例如目标物体的位置和方向 |

一次完整示教：

以下为显式选择 `--control keyboard` 后的旧版键盘/拖动操作；Xbox 操作见上一节。在采集面板勾选 **Keyboard control**，点击视口后操作。
按键改变红色 `TCPGoal` 的期望位姿，机械臂仍通过 IK 跟随；绿色目标不受影响。

| 按键 | 末端目标变化（世界坐标系，与基座坐标系重合） |
|---|---|
| `I` / `K` | X 正 / 负方向 |
| `J` / `L` | Y 正 / 负方向 |
| `U` / `O` | Z 正 / 负方向（上 / 下） |
| 顶部数字键 `1` / `2` | 绕 X 轴正 / 负旋转 |
| 顶部数字键 `3` / `4` | 绕 Y 轴正 / 负旋转 |
| 顶部数字键 `5` / `6` | 绕 Z 轴正 / 负旋转 |
| `Z` / `X` | 夹爪完全打开 / 闭合（按一次执行至目标） |
| `Esc` | 关闭键盘控制 |

默认每次按键平移 **1 mm** 或旋转 **1°**；长按使用系统按键重复，速度受系统重复设置影响。
松开后停止增加目标位姿，机械臂可能继续跟随至已下发的目标。操作其他面板或输入文字前先关闭键盘控制。
不可达的键盘目标会被拒绝；若正在录制，沿用原来的 IK 失败处理，丢弃当前录制。
可通过启动参数调整每次增量，例如：

```bash
./scripts/sim.sh --mode interactive --control keyboard --output-dir recordings/keyboard_01 \
  --keyboard-step 0.0005 --keyboard-angle 0.5
```

键盘示教时，下方第 2、4 步的拖动操作直接换成按键即可，Record / Save 等采集流程相同。

1. 在 Stage 面板选中绿色 `Target`，用平移/旋转工具将它放在这条示教对应的目标位置。
2. 选中红色 `TCPGoal`，用变换工具将 Nero 移到示教起点，等待机械臂跟上。
3. 点击采集面板的 **Record**。
4. 缓慢拖动红色 `TCPGoal`，完成“接近目标 → 到达目标上方”等动作。需要姿态变化时使用旋转工具。
5. 末端停稳后点击 **Save**，终端和面板会显示保存路径。
6. 使用 **Reset Home** 或拖动控制点准备下一条。移动绿色目标，再开始下一条录制。

可以通过 Stage 面板选择精确对象；视口常用 `W` 切换平移、`E` 切换旋转。
录制时不要移动绿色目标：本实现每条示教只有一个固定任务参考系，检测到目标移动会丢弃当前录制。
IK 失败时也会丢弃当前录制并显示提示；将红色控制点移回可达范围后重新开始。

- **Discard**：丢弃当前未保存的示教。
- **Reset Home**：仅在没有录制时恢复仿真初始关节位置；复位过程不录入数据。
- 暂停/停止时间轴会丢弃未保存录制。请保持播放，使用面板中的复位按钮。
- 关闭窗口前必须点击 Save；关闭时不会自动保存半条示教。
- 每个技能单独使用一个录制目录。建议 3–5 条，动作顺序和节奏尽量一致。

仿真直接提供目标参考系真值，无需相机检测或手眼标定。
记录的 TCP 是 `gripper_base`，不是夹爪指尖。

夹爪可用面板中的 **Open Gripper / Close Gripper / Half Open** 控制，无需打开键盘开关；
启用 **Keyboard control** 后，也可以按 `Z` 打开、`X` 闭合。时间轴必须处于播放状态。
两个滑动关节分别在 `[0, 0.05]` m 和 `[-0.05, 0]` m 范围内对称运动，
每指目标速度限制为 20 mm/s。启动和 Reset Home 恢复半开状态（关节位置 ±0.025 m）。
按一次开合键会执行至目标，松开按键或关闭键盘控制不会取消已下发的开合动作。
手动 `interactive` 及旧的 `collect` 示教仍只记录机械臂轨迹，不记录夹爪开合时序；这些数据的回放保持半开。
`grasp` 示教会记录夹爪时序，并在训练和回放时保留。
绿色 Target 是无碰撞标记，不能夹起；自动抓取模式中的橙色 GraspCube 才是实体物块。

训练自己的录制：

```bash
./scripts/learn.sh train --demos recordings/manual_01/demo_*.npz \
  --output skills/manual_v1.npz --samples 100 --components 8
./scripts/learn.sh predict --skill skills/manual_v1.npz \
  --output output/manual_reproduce.npz
./scripts/sim.sh --mode replay --trajectory output/manual_reproduce.npz \
  --duration 10 --output-dir output/manual_test
```

## 测试新目标位置下的泛化

从某条示教的目标坐标系生成一个向 Y 方向平移 10 mm 的测试目标：

```bash
./scripts/learn.sh frames --demo recordings/auto_01/demo_001.npz \
  --offset 0 0.01 0 --output output/new_target.json

./scripts/learn.sh predict --skill skills/reach_v1.npz \
  --frames output/new_target.json --output output/generalized.npz

./scripts/sim.sh --mode replay --trajectory output/generalized.npz \
  --duration 8 --output-dir output/generalized_test
```

也可以编辑 JSON 中的完整目标位姿，格式为 `[x, y, z, qw, qx, qy, qz]`，位置单位为米，
坐标系是 Nero `base_link`。仿真基座与世界坐标系重合并固定。
回放时绿色目标自动使用预测文件中的参考系。

目前使用一个 `target` 参考系，因此整体轨迹包括起点都会随参考系变化。
回放会先通过 IK 将仿真机械臂移到预测起点，再开始统计跟踪误差。

## 结果与调参

回放结果目录包含：

- `tracking.npz`：同一仿真采样时刻的 `time`、`desired`、`actual`。
- `metrics.json`：位置/旋转误差的 RMSE、最大值、终点值，以及阈值和 `passed`。

默认要求全程最大位置误差不超过 10 mm、最大姿态误差不超过 0.1 rad。
IK 失败或跟踪误差超限会报告失败；误差超限时仍保存报告，便于排查。
这评估的是机械臂对 TP-KMP 预测轨迹的跟踪，不是物体抓取成功率。

可调整：

```bash
./scripts/sim.sh --help
```

`--duration` 控制动作时间；`--joint-speed` 控制每步关节目标变化；`--kp` / `--kd` 控制仿真位置驱动。
关节目标的限速以上一帧已下发目标为基准，避免把重力造成的下垂不断带入下一帧目标。
每个物理步使用 PhysX 的 `get_generalized_gravity_forces()` 施加重力补偿力矩，世界重力保持开启。
若已打开旧版本窗口，需要保存当前数据并关闭窗口，再重新运行脚本，修复才会生效。
默认物理频率 120 Hz，采样 30 Hz，频率参数分别是 `--physics-hz` / `--record-hz`。
增加动作时间通常可降低跟踪误差；不要用放宽阈值来代替检查明显异常的轨迹。

采集数据每行为 `[t, x, y, z, qw, qx, qy, qz]`，同时保留实际七轴关节位置。
末端位姿由 PhysX 关节反馈经原生 Lula 正运动学得到，不是保存期望控制点位置。
训练阶段复用原算法：线性位置插值、SLERP 姿态插值和归一化时间，不做 DTW。

## 范围与来源

项目提供 Nero **自由空间运动技能**采集和小方块抓取采集。绿色目标方块是无碰撞参考标记，
机械臂自碰撞在导入时关闭，Lula IK 不规划避障路径；因此通过测试不代表真机碰撞安全。
地面、机械臂及抓取模式的橙色方块由 PhysX 模拟；未加入 RGB-D 数据或手柄输入。

机器人几何和运动学来自本机 nero_ws；资产许可证保留在 `assets/LICENSE.nero`。
TP-KMP 算法及许可证来自 TP-KMPs，训练接口复用 TP-KMPs-nero。
新增实现采用 Isaac Sim 5.1 本地 API，并参考 NVIDIA 官方
[URDF 导入文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.asset.importer.urdf/docs/index.html)
和 [Lula 运动学说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/manipulators/concepts/kinematics_solver.html)。

验证记录见 `VALIDATION.md`。离线测试可运行：

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

上抬检查（保持重力开启，目标为初始 TCP 上方 20 mm）：

```bash
./scripts/sim.sh --mode smoke --headless --output-dir output/lift_check
```

检查要求终点位置误差不超过 2 mm、姿态误差不超过 0.02 rad，结果保存为 `smoke.json`。
