# 双臂安装：右侧两颗完成后松开左臂

独立入口为 `scripts/assembly.sh`，控制代码在 `assembly_collection/`。所有新脚本、文档和数据均在本项目内；仿真入口不连接 CAN 或真实机械臂。场景配置仅覆盖独立运行的内存舞台。

## 当前流程

默认 `--stage full` 执行以下流程：

1. 左臂实际抓取 keel，搬运、旋转并对准墙孔。
2. 右臂抓取并插入右上第 4 颗，打开夹爪、撤离，检查默认 10 秒的摩擦保持。
3. 同样安装右下第 3 颗。两颗的实际入墙深度、轴线和 keel 孔贯穿均须通过。
4. 左臂将持有的 keel 缓慢下移约 0.2 mm，让孔边与螺栓肩部建立承载接触；随后停止物体跟随、打开夹爪并撤离。观察重力下的实际运动，通过后回到待机位。
5. 右臂继续安装左下第 1 颗、左上第 2 颗。左臂保持空手待机。
6. 双手打开撤离，再观察 10 秒，重新检查 keel 和四颗螺栓，全部通过才标记 `installation_complete=true`。

面对 keel 正面时，孔号及安装顺序如下。右侧两颗完成前不松开左臂。

| 位置 | 左列 | 右列 |
| --- | --- | --- |
| 上排 | 2，最后安装 | 4，第一颗 |
| 下排 | 1，第三颗 | 3，第二颗 |

`--stage right-pair` 只完成前四步及最终自由保持检查，适合先单独验证松手；它不是四颗安装完成示范。`full` 已有独立的松手流程，不再是 `insert-all` 别名。历史 `insert-one` / `insert-all` 仍用于左臂持续扶持的诊断。

**整个过程不建立零件装配关节。** keel 和四颗螺栓始终为重力开启的动态刚体，搬运、插入及松手后的承载通过实际接触计算。松手后若失稳，程序保存失败，不重新抓回、冻结或追随零件来修改结果。当前结果与历史诊断见 [VALIDATION.md](VALIDATION.md)。

当前版本已实际通过原位 1 次，以及同一进程两次初始 XY 随机 ±1 mm 的采集和复位；三条完整数据均通过审计。已验证的是当前布局及这两个小偏移，更大范围或其他布局需要重新验证。

## 直接运行

在项目根目录执行，每次使用新的输出文件夹。

```bash
# 先验证右侧两颗和左臂松手。
./scripts/assembly.sh --stage right-pair --keep-open \
  --output-dir recordings/right_pair_01

# 完整四颗安装，默认顺序 4、3、1、2。
./scripts/assembly.sh --stage full --keep-open \
  --output-dir recordings/assembly_handoff_01

# 原位重复采集三次，包括自动复位。
./scripts/assembly.sh --stage full --demos 3 --headless \
  --output-dir recordings/assembly_repeat_01

# keel 初始 XY 各随机 ±1 mm。先小批验证，再扩大采集。
./scripts/assembly.sh --stage full --demos 5 --seed 42 \
  --keel-jitter 0.001 --headless \
  --output-dir recordings/assembly_random_01

# 检查并列出合格示范；失败轨迹仍保留但不计入成功数据。
/home/cjh/anaconda3/envs/tpkmp/bin/python -B \
  assembly_collection/check_dataset.py recordings/assembly_random_01 --require-success

# 绘制单次的实际/目标轨迹、夹爪与扶持状态。
/home/cjh/anaconda3/envs/tpkmp/bin/python -B \
  assembly_collection/visualize.py recordings/assembly_handoff_01/episode_001
```

`--headless` 关闭画面；`--keep-open` 保留 GUI，失败时暂停在失败位置。运行时暂停时间轴或关闭窗口会使本次失败。`summary.json` 汇总整个批次，`episode_*/report.json` 保存每次详细原因；批次至少一次失败时入口退出非零。数据检查器的 `--require-success` 要求至少有一条合格轨迹，成功条数需以输出列表为准。

## 分项验证

```bash
# 检查装配端点可达，不证明整段路径无碰撞。
./scripts/assembly.sh --stage preflight --headless \
  --output-dir recordings/handoff_preflight_01

# 空载撤离诊断，比较实时重算 IK 与缓存目标关节解。
./scripts/assembly.sh --stage motion-check --headless \
  --output-dir recordings/empty_wrist_probe_01

# 分别检查左臂拾取、贴墙或右臂拾取。各次都从初始场景开始。
./scripts/assembly.sh --stage keel-grasp --keep-open \
  --output-dir recordings/handoff_keel_grasp_01
./scripts/assembly.sh --stage keel-align --keep-open \
  --output-dir recordings/handoff_keel_align_01
./scripts/assembly.sh --stage screw-grasp --keep-open \
  --output-dir recordings/handoff_screw_grasp_01

# 预放螺栓，比较单颗和右侧两颗的自由承载。
./scripts/assembly.sh --stage support-check --headless \
  --output-dir recordings/pair_support_probe_01

# 预放单颗，比较孔内有/无摩擦及有限外拔扰动。
./scripts/assembly.sh --stage retention-check --headless \
  --output-dir recordings/friction_probe_01

# 此前中部直立夹持、8 mm 光杆的单颗实际插入诊断。
./scripts/assembly.sh --stage insert-one --keep-open \
  --output-dir recordings/contact_insert_one_01

# 单颗松配合光杆的自由松手诊断，关闭接触环；预期会失稳。
./scripts/assembly.sh --stage release-test --screw-count 1 --bolt-retention none --headless \
  --output-dir recordings/contact_gravity_01
```

`support-check` 和 `retention-check` 都是预放零件的独立物理测试，标记 `preinserted=true`、`robot_demonstration=false`，不可当作机械臂示范。`motion-check` 预置空手关节姿态，标记 `prepositioned=true`、`robot_demonstration=false`，也不可用于训练。机器人采集只来自实际执行抓取、搬运和插入的阶段。

## 路径与抓取参数

采用七轴 IK、平滑位置插值和旋转插值，每段执行前检查 IK。目标变化时使用实际关节角作为初值；目标不变时缓存同一关节解，避免七轴冗余解反复切换后无法收敛，关节指令仍限速至 0.8 rad/s。左臂夹持轨道，用实际 keel 位姿反馈贴墙对孔；右臂抬高螺栓、转向、到孔上方后下降接近，沿墙法向分步插入。插入根据实际螺栓位姿校正，并检查实际手部是否跟上驱动目标；明显落后时暂停叠加修正，保留承载所需的驱动补偿。螺栓预对齐和接触中只校正圆孔轴线，允许圆柱自由自转。开爪时停止物体跟随，保持测得的手部位姿，减少带载驱动补偿卸载时的回弹。不以 TCP 到达代替插入成功。圆柱螺栓在料架上的自转由 `canonical_screw_target` 补偿，再施加所选夹爪轴向转角，保持可重复的夹爪朝向。两臂接触会立即拒绝本次试验。

它是针对当前布局的分段控制，未实现任意场景自动避障。改变场景或抓取参数后必须重新执行实际接触验证。

默认 `full`、`right-pair`、分项抓取/对齐检查及 `support-check` 使用以下参数；显式命令参数可以覆盖。`insert-one`、`insert-all`、`release-test`、`retention-check` 保留此前直立夹持、40 mm 夹持头、无肩部的诊断参数，详细配置及采集时的 Python 源文件 SHA-256 均保存到 `settings.json`。

| 参数 | 松手流程默认值 | 用途 |
| --- | --- | --- |
| `--left-grip-x` | `0` m | keel 局部轨道中部 |
| `--left-tilt-deg` | `-65`° | 腕部向左侧倾斜，让出右侧插入空间 |
| `--keel-start-yaw` | `90`° | 初始摆放使倾斜拾取可达 |
| `--pickup-rest-height` | `0.030` m | 真实碰撞支座，给倾斜夹指留出空间 |
| `--screw-clock-deg` | `45`° | 各孔选择正/负转角，改善下排孔腕部关节行程 |
| `--approach-distance` | `0.060` m | 下降时让 28 mm 螺杆尖端绕过 keel 凸起，再沿轴线插入 |
| `--screw-grip-offset` | `0.012` m | 左右孔对应的夹持头外侧偏置 |
| `--screw-head-height` | `0.050` m | 加长夹持头，给宽夹指留出避开轨道的空间 |
| `--bolt-shoulder-radius` | `0.005` m | 直径 10 mm、长 3 mm 的短承载肩部；`0` 为无肩部 |
| `--bolt-shoulder-lead-in` | `0.002` m | 后装左侧两颗的导入段长度，右侧两颗使用完整圆柱肩部 |
| `--finger-stiffness` | `600` N/m | 右臂夹指刚度；左臂为 600 N/m，两侧每个夹指最大力均为 10 N |
| `--handoff-settle` | `0.0002` m | 两颗安装后，左臂持物下移指令 |
| `--keel-static-friction` / `--keel-dynamic-friction` | `0.4` / `0.3` | keel 孔边与螺栓肩部接触摩擦 |

右侧 keel 孔径 10.5 mm，左侧孔径 12 mm；10 mm 肩部均可进入。肩部仅位于 keel 底板厚度范围，进入墙孔的轴仍为直径 8 mm、长 28 mm。右侧两颗采用完整的 3 mm 圆柱肩部承载；后装的左侧两颗带 2 mm 导入段和 1 mm 圆柱段，可通过接触引导略微下沉的孔。肩部底端在目标位姿距墙面约 0.3 mm，避免进入直径 9 mm 的墙孔。螺栓假定质量仍为 25 g，未按真实紧固件标定。原先 8 mm 光杆与右侧孔有 1.25 mm 径向间隙，会造成两颗下的下沉和转动；短肩部把该间隙减为 0.25 mm，通过真实孔边接触承担 keel 重力。夹指刚度和 10 N 上限在每回合复位后重新应用。

默认孔序 4、3、1、2；`--first-screw 3` 可交换右侧两颗的先后，但改变顺序需重新验证。`full --screw-count 2` 或 `3` 会保存部分安装，只有四颗全部通过才为完成。`right-pair` 始终安装两颗。

基座位于 `[0, ±0.375, 1.0]` m。默认场景为 `assets/scenes/keel_assembly.usda`；独立舞台墙沿世界 X 正向偏移 120 mm、keel 初始 X 正向偏移 55 mm、料架和螺栓 Y 正向偏移 40 mm，分别用 `--wall-offset-x`、`--keel-offset-x`、`--rack-offset-y` 修改。`--lower-hole-route below`、`--elbow-policy separated`、`--left-elbow-height` 属于候选路径/姿态参数，当前默认从上方接近、不开启肘高调整。

## 墙孔摩擦保持

默认 `--bolt-retention friction` 在每个墙孔深度 8 mm、20 mm 处添加两圈宽 2 mm 的柔性接触环，32 个凸扇区组成，每端带导入倒角。8 mm 螺杆进入约 7.92 mm 的内径后，有限接触压缩产生孔壁压力，摩擦由 PhysX 计算。它始终启用，螺杆离开接触区域后保持作用消失；没有达到深度后添加固定连接的事件。

| 参数 | 默认值 |
| --- | --- |
| `--bore-interference` | `0.00004` m，径向压缩量 |
| `--bore-static-friction` / `--bore-dynamic-friction` | `2.5` / `2.0` |
| `--bore-stiffness` | `2000` N/m |
| `--bore-damping` | `10` Ns/m |
| `--retention-seconds` | `10` s |

这是用户选择的简化摩擦模型，不模拟螺纹、旋拧、膨胀或预紧。接触环保留螺栓，肩部与 keel 孔边接触承载；不能仅凭摩擦系数数值宣称安装成功。物理模型与边界见 [ANCHOR_MODEL.md](ANCHOR_MODEL.md)。

## 数据与判定

格式为 `dual_keel_friction_insertion_v3`，检查器兼容此前无接触环的 v2，并排除历史固定约束 v1。`success` 表示所选阶段通过；`right-pair` 成功是部分安装，`full` 四颗且最终双手撤离的检查通过才是完整安装。所有数据均为仿真，`anchoring_verified=false` 表示没有真实膨胀件的验证。

位姿为世界坐标 `[x,y,z,qw,qx,qy,qz]`，机械臂顺序左、右；长度米，角度弧度。默认物理 120 Hz、记录 30 Hz。

| 字段 | 形状与含义 |
| --- | --- |
| `times` | N 个同步时间 |
| `tcp_poses` / `desired_tcp_poses` | N×2×7，实际/期望末端位姿 |
| `joint_positions` / `joint_targets` | N×2×7，实际/期望七轴角度 |
| `finger_joint_positions` / `gripper_commands` | N×2×2 实际夹指位置 / N×2 目标开度 |
| `keel_poses` / `screw_poses` | N×7 / N×4×7，实际零件位姿 |
| `keel_linear_velocities` / `screw_linear_velocities` | N×3 / N×4×3，原始速度 |
| `stage_ids` / `stage_names` / `active_screw` | 动作阶段及当前零件下标 |
| `insertion_states` | N×4，插入后通过实际保持检查的状态 |
| `left_support_active` | N，左臂物体跟随状态；两颗后关闭 |
| `left_release_time_s` / `release_after_right_pair` | 实际交接时间及流程标志 |
| `wall_pose` / `wall_holes` / `keel_target` | 墙和安装目标 |
| `assembly_joints` / `gravity_enabled` | 必须为 0 / true |
| `friction_retention_verified` | 螺栓在右手打开撤离后保持通过 |
| `independently_mounted` | 最终双手撤开后 keel 和螺栓自行保持 |
| `installation_complete` | 四颗完整安装，且自由重力检查通过 |
| `bore_*` / `keel_*_friction` / `bolt_shoulder_radius_m` / `handoff_settle_m` | 实际物理参数 |
| `bolt_shoulder_lead_in_m` / `finger_stiffness_N_per_m` / `finger_effort_limit_N` | 每颗导入长度、左右夹指刚度和每指力上限 |

夹取要求实际抬升、夹指阻挡和低滑移。贴墙检查实际 keel 四孔。插入要求入墙 22～26.5 mm、孔口/尖端偏差小于 0.45 mm、轴线误差小于 0.02 rad，并贯穿实际底板两面。肩部版本使用真实肩部半径计算孔隙，插入时保留 0.05 mm 径向余量；光杆诊断保留 0.15 mm。

螺栓松手保持要求右手实际开度超过 60 mm、无手部接触，末尾约 1 秒逐物理帧位置变化小于 0.25 mm、净移动速度小于 1 mm/s、深度变化小于 1 mm，并再次检查实际孔贯穿。原始瞬时速度另行保存，避免把柔性接触振动当成持续滑出。

右手松开后允许孔边承载接触，即使左臂仍在扶持，采用孔隙加 0.1 mm 的几何容差；仍须检查入墙深度 20～26.5 mm、墙孔偏差小于 0.55 mm、轴线误差小于 0.05 rad。最终 keel 位姿误差须小于 2 mm、角度小于 0.02 rad，所有已装螺栓均须留在实际孔内，双手打开且无接触。首次松手失败会停止，后续零件失稳也会拒绝整次。

`report.json` 保存 `grasp_checks`、`insertion_events`、`retention_checks`、`bearing_settle`、`right_pair_release`、`mounted_check` 和 `physics_check`。检查器审计右侧两颗保持检查先于松手，随后左臂跟随不再开启；失败轨迹不计入合格示范。可视化输出在单次目录的 `validation.png`。

原 TP-KMP 单臂训练入口不能直接读取此双臂格式；后续训练需按动作阶段和机械臂建模，保留夹爪、接触及交接事件。当前脚本负责生成可审查的同步示范。

## 文件分工

`run.py` 管命令参数，`simulation.py` 管运动、反馈、检查和记录，`geometry.py` 管位姿/孔位，`physics.py` 管碰撞材料、肩部及无装配关节检查，`friction_bores.py` 管墙孔接触环，`pair_support_probe.py` 和 `retention_probe.py` 管独立承载/摩擦测试。
