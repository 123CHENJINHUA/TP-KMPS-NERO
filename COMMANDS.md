# 机械臂常用指令说明

按当前程序整理，更新于 2026-10-21。每个 Bash 代码块都从项目目录开始，可以整块复制到终端。选择需要的模式运行，不需要依次执行全文。

本机入口已配置好路径：仿真使用 Isaac Sim 自带 Python，TP-KMP 使用 `tpkmp` 环境，PPO 自动激活 `isaaclab` 环境。无需先手动打开 Isaac Sim。

采集、训练和验证输出使用时间戳，避免覆盖已有文件。一个代码块内的变量与后续命令要一起复制到同一个终端。

## 快速查找

| 要做什么 | 查看章节 | 当前范围 |
| --- | --- | --- |
| Xbox 控制双臂，Y 切换 | [1.1](#xbox-dual) | 双臂控制预览，不录制 |
| Xbox 控制单臂并录制 | [1.2](#xbox-single) | 记录 TCP 和关节，不记录夹爪时序 |
| 键盘控制与录制 | [2](#keyboard) | 当前键盘入口只支持单臂 |
| 现实双臂同步仿真 | [3](#teach-sync) | can0/can1 被动读取与同步，不录制 |
| 随机抓方块或 keel，采集数据 | [4](#auto-grasp) | 抓取数据包含夹爪时序 |
| TP-KMP 训练、泛化和回放 | [5](#tpkmp) | 单臂抓取示教格式 |
| 自动双臂安装、分步检查 | [6](#assembly) | 独立装配程序，不连接 CAN |
| PPO 位置与力训练、回放 | [7](#ppo) | 孔前单颗插入，左臂继续扶持 |
| 场景、相机及帮助 | [8](#scene) | 现有场景与参数 |
| D435i 手眼标定 | [9](#hand-eye) | 眼在手上/眼在手外，真实静止采样 |
| 真机 TP-KMP 采集、训练及执行 | [10](#real-tpkmp) | 共享 can0，单只从臂抓取/搬运 |
| D435i 视觉定位与自动抓取参考系 | [10.1](#real-vision) | SAM3 + CAD 配准，两种手眼安装 |
| Meshy 单图/多图生成物体模型 | [10.2](#meshy) | 网页/API → 实测尺度校正 → 视觉配置 |
| 单独验证视觉连续跟踪 | [10.3](#vision-tracking) | 相机预览、抖动/平移检查；不读 CAN |

## 1. 遥控器：Xbox

<a id="xbox-dual"></a>

### 1.1 双臂 Xbox：当前装配场景

包含桌面、keel、螺栓架和墙板，适合目前的双臂场景操作：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode dual --control xbox \
  --scene-usd assets/scenes/keel_assembly.usda
```

启动后松开摇杆、扳机和按钮，待面板显示 `active` 再操作。**按实体 Y 键切换机械臂**，切换后再次回中。

| 操作 | 动作 |
| --- | --- |
| 左摇杆上／下 | 沿 TCP X 正／负方向移动 |
| 左摇杆右／左 | 沿 TCP Y 正／负方向移动 |
| LT／RT | 沿 TCP Z 正／负方向移动 |
| 右摇杆右／左 | 绕 TCP X 负／正方向旋转 |
| 右摇杆上／下 | 绕 TCP Y 正／负方向旋转 |
| A／B | 绕 TCP Z 正／负方向旋转 |
| 十字键左／右 | 夹爪闭合／打开一步 |
| 十字键上 | 完全闭合 |
| 十字键下 | 35% 开度 |
| Y | 双臂模式切换当前机械臂 |
| Menu／Start | 暂停／启用手柄控制 |

移动方向相对于当前 TCP，不一定是画面中的上下左右。两只机械臂分别保留自己的目标和开度。

降低操作速度、指定 `js0`：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode dual --control xbox \
  --scene-usd assets/scenes/keel_assembly.usda \
  --gamepad-device /dev/input/js0 \
  --gamepad-speed 0.02 --gamepad-angle-speed 10 --gamepad-deadzone 0.15
```

<a id="xbox-single"></a>

### 1.2 单臂 Xbox 控制与录制

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode interactive --control xbox \
  --output-dir "recordings/xbox_cmd_$(date +%Y%m%d_%H%M%S)"
```

在 `Nero TP-KMP Collection` 面板点击 **Record → 用手柄运动 → Save**。同一条录制期间保持绿色 `Target` 不动；关闭窗口前先保存。

单臂模式的 Y 不切换机械臂。这个入口录制末端及关节轨迹，不录制夹爪动作；双臂 `dual` 当前没有录制入口。

检查手柄设备：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ls -l /dev/input/js*
```

<a id="keyboard"></a>

## 2. 键盘控制：单臂

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode interactive --control keyboard \
  --output-dir "recordings/keyboard_cmd_$(date +%Y%m%d_%H%M%S)"
```

在采集面板勾选 **Keyboard control**，点击视口后操作。

| 按键 | 动作，世界坐标系 |
| --- | --- |
| I／K | X 正／负方向移动 |
| J／L | Y 正／负方向移动 |
| U／O | Z 正／负方向移动 |
| 顶部数字 1／2 | 绕 X 正／负方向旋转 |
| 顶部数字 3／4 | 绕 Y 正／负方向旋转 |
| 顶部数字 5／6 | 绕 Z 正／负方向旋转 |
| Z／X | 夹爪完全打开／闭合 |
| Esc | 关闭键盘控制 |

录制使用面板的 **Record / Save / Discard**。默认每次按键 1 mm、1°，需要更细的增量时：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode interactive --control keyboard \
  --keyboard-step 0.0005 --keyboard-angle 0.5 \
  --output-dir "recordings/keyboard_fine_cmd_$(date +%Y%m%d_%H%M%S)"
```

**当前不能使用 `--mode dual --control keyboard`**，参数检查会拒绝。现有键盘入口也不能加载 `--scene-usd`；装配场景的双臂手动操作使用 1.1 的 Xbox 或 3 的硬件试教。

<a id="teach-sync"></a>

## 3. 双臂硬件试教同步

直接移动现实中的试教臂，仿真同步关节和 AGX 试教夹爪。默认 **can0 → 左臂 Nero 1，can1 → 右臂 Nero 2**。程序只接收报文，不向现实机械臂发送运动指令。

### 3.1 检查 CAN

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ip -details link show can0
ip -details link show can1
/home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/probe_can.py --seconds 5
```

下面只启用尚未 `UP` 的接口，已启用的接口保持原状：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
for teach_can_interface in can0 can1; do
  if ip link show "$teach_can_interface" | rg -q '<([^>]*,)?UP([,>])'; then
    continue
  fi
  sudo ip link set "$teach_can_interface" up type can bitrate 1000000
done
```
启动can口
sudo ip link set can0 up type can bitrate 1000000

检查两路关节和试教夹爪回显：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
/home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/read_dual_nero.py --seconds 10
```

### 3.2 同步到当前装配场景

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode teach-sync \
  --teach-left-can can0 --teach-right-can can1 \
  --scene-usd assets/scenes/keel_assembly.usda
```

如果现实左右臂对应反了，使用下面这一条：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode teach-sync \
  --teach-left-can can1 --teach-right-can can0 \
  --scene-usd assets/scenes/keel_assembly.usda
```

加载之前修复了刚体和碰撞的 `0928_grasp.usd`：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode teach-sync \
  --teach-left-can can0 --teach-right-can can1 \
  --scene-usd assets/scenes/0928_grasp.usd
```

仅做 120 帧无窗口同步检查：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode teach-sync --headless --steps 120 \
  --teach-left-can can0 --teach-right-can can1 \
  --scene-usd assets/scenes/keel_assembly.usda
```

面板中的 **Sync enabled** 暂停／恢复同步。左右夹爪均为直接开度映射，右侧没有软件反转。当前 `teach-sync` 只同步预览，**不会保存示教轨迹**；试教器回显的力也不能视为实际夹爪测量力。

<a id="auto-grasp"></a>

## 4. 自动抓取与数据采集

### 4.1 小方块：先测试两次

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode grasp --object cube --demos 2 --duration 10 \
  --output-dir "recordings/cube_check_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 4.2 小方块：丰富位置，三组各 12 次

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode grasp --object cube \
  --grasp-groups 3 --demos 12 --duration 10 \
  --grasp-x-range 0.28 0.34 --grasp-y-range -0.05 0.05 \
  --grasp-sampling stratified --seed 42 \
  --output-dir "recordings/cube_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 4.3 keel：固定位置测试两次

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode grasp --object keel --grasp-spread 0 \
  --demos 2 --duration 10 \
  --output-dir "recordings/keel_check_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 4.4 keel：随机位置，三组各 12 次

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode grasp --object keel \
  --grasp-groups 3 --demos 12 --duration 10 --grasp-spread 0.012 \
  --grasp-sampling stratified --seed 42 \
  --output-dir "recordings/keel_cmd_$(date +%Y%m%d_%H%M%S)"
```

这些指令默认显示抓取过程；添加 `--headless` 可无窗口采集。多组分别保存在 `group_001/`、`group_002/`、`group_003/`。训练只读取成功的 `demo_*.npz`，不读取 `failed_*.npz`。

### 4.5 只采接近轨迹，不抓取物体

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode collect --demos 5 --duration 8 \
  --output-dir "recordings/reach_cmd_$(date +%Y%m%d_%H%M%S)"
```

`collect` 是接近动作；实际自动夹取使用 `grasp`。

<a id="tpkmp"></a>

## 5. TP-KMP：训练与泛化验证

### 5.1 只训练：使用本机已有的 keel 三组数据

前两组训练，第三组保留用于验证：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/learn.sh train \
  --demos recordings/keel_random_groups_01/group_001/demo_*.npz \
          recordings/keel_random_groups_01/group_002/demo_*.npz \
  --output "skills/keel_only_cmd_$(date +%Y%m%d_%H%M%S).npz" \
  --samples 100 --components 8
```

### 5.2 自动选择三组 keel 数据，训练并做一次泛化验证

以下整块会：选择 keel 三组数据 → 用前两组训练 → 从第三组取一个未参与训练的目标 → 预测 → 在仿真中验证。

优先使用 4.4 生成的最新完整 `keel_cmd_*` 数据；没有这类数据时，使用本机已存在的 `recordings/keel_random_groups_01`。所有输出使用新时间戳。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
keel_data_dir=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B - <<'PY'
from pathlib import Path
candidates = [p for p in Path('recordings').glob('keel_cmd_*')
              if all(list((p / f'group_{i:03d}').glob('demo_*.npz')) for i in (1, 2, 3))]
print(max(candidates, key=lambda p: p.stat().st_mtime) if candidates
      else Path('recordings/keel_random_groups_01'))
PY
)
keel_test_demo=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B -c \
  'from pathlib import Path; import sys; print(sorted((Path(sys.argv[1]) / "group_003").glob("demo_*.npz"))[0])' \
  "$keel_data_dir")
keel_command_stamp=$(date +%Y%m%d_%H%M%S)
keel_skill_file="skills/keel_cmd_${keel_command_stamp}.npz"
keel_eval_dir="output/keel_cmd_${keel_command_stamp}"
./scripts/learn.sh train \
  --demos "$keel_data_dir"/group_001/demo_*.npz "$keel_data_dir"/group_002/demo_*.npz \
  --output "$keel_skill_file" --samples 100 --components 8 &&
./scripts/learn.sh frames --demo "$keel_test_demo" \
  --output "$keel_eval_dir/target.json" &&
./scripts/learn.sh predict --skill "$keel_skill_file" \
  --frames "$keel_eval_dir/target.json" --output "$keel_eval_dir/prediction.npz" &&
./scripts/sim.sh --mode replay --object keel \
  --trajectory "$keel_eval_dir/prediction.npz" --duration 10 \
  --output-dir "$keel_eval_dir/replay"
```

技能路径在 `keel_skill_file`。完整复制会同时完成一次泛化验证；只训练使用 5.1 的独立指令。

这一块先生成 `target.json`，再预测，避免之前的目标 JSON 不存在错误。检查 `replay/grasp_result.json` 和 `replay/metrics.json`，两者 `passed` 均为 `true` 才算通过。

### 5.3 使用已有 keel 技能验证第三组全部目标

下面可直接用本机已有 `skills/keel_v1.npz` 验证第三组全部目标；它只读取第三组目标位姿，不直接回放第三组示范轨迹：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
keel_batch_dir="output/keel_generalization_cmd_$(date +%Y%m%d_%H%M%S)"
for keel_demo in recordings/keel_random_groups_01/group_003/demo_*.npz; do
  keel_demo_name=$(basename "$keel_demo" .npz)
  ./scripts/learn.sh frames --demo "$keel_demo" \
    --output "$keel_batch_dir/${keel_demo_name}_target.json" &&
  ./scripts/learn.sh predict --skill skills/keel_v1.npz \
    --frames "$keel_batch_dir/${keel_demo_name}_target.json" \
    --output "$keel_batch_dir/${keel_demo_name}_prediction.npz" &&
  ./scripts/sim.sh --mode replay --object keel --headless \
    --trajectory "$keel_batch_dir/${keel_demo_name}_prediction.npz" --duration 10 \
    --output-dir "$keel_batch_dir/${keel_demo_name}_replay"
done
```

未通过或未生成报告的测试也计入总数。这个例子验证同一个 keel、同一采集范围内的位置泛化，前提是第三组没有参与该技能训练。

### 5.4 训练键盘／Xbox 单臂手动示教

先通过 1.2 或 2 的面板保存至少两条示教。下面自动选择本说明生成的、含至少两条示教的最新手动目录；尚未录制足够数据时不能执行训练：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
manual_demo_dir=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B - <<'PY'
from pathlib import Path
root = Path('recordings')
folders = [p for pattern in ('xbox_cmd_*', 'keyboard_cmd_*', 'keyboard_fine_cmd_*')
           for p in root.glob(pattern) if len(list(p.glob('demo_*.npz'))) >= 2]
if not folders:
    raise SystemExit('请先在单臂手动模式保存至少两条示教。')
print(max(folders, key=lambda p: p.stat().st_mtime))
PY
) &&
./scripts/learn.sh train --demos "$manual_demo_dir"/demo_*.npz \
  --output "skills/manual_cmd_$(date +%Y%m%d_%H%M%S).npz" \
  --samples 100 --components 8
```

<a id="assembly"></a>

## 6. 自动双臂装配与采集

使用独立的 `scripts/assembly.sh`。这套程序会抓取、搬运、插入；螺栓靠接触摩擦保持，不建立零件固定约束。

### 6.1 先检查端点可达性

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage preflight --headless \
  --output-dir "recordings/assembly_preflight_cmd_$(date +%Y%m%d_%H%M%S)"
```

这检查端点 IK，不代表整条路径或物理插入已经通过。

### 6.2 分步检查抓取与贴墙

每条都独立启动，从初始场景开始。按需要选择一条复制：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage keel-grasp --keep-open \
  --output-dir "recordings/assembly_keel_grasp_cmd_$(date +%Y%m%d_%H%M%S)"
```

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage keel-align --keep-open \
  --output-dir "recordings/assembly_keel_align_cmd_$(date +%Y%m%d_%H%M%S)"
```

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage screw-grasp --keep-open \
  --output-dir "recordings/assembly_screw_grasp_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 6.3 安装右侧两颗，再松开左臂

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage right-pair --keep-open \
  --output-dir "recordings/assembly_right_pair_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 6.4 完整安装四颗螺栓

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/assembly.sh --stage full --keep-open \
  --output-dir "recordings/assembly_full_cmd_$(date +%Y%m%d_%H%M%S)"
```

默认顺序 4、3、1、2。右侧两颗通过后左臂松手；四颗安装且最终双手撤离后稳定，才标记完整安装。

### 6.5 小范围随机摆放，采集并检查

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
assembly_data_dir="recordings/assembly_batch_cmd_$(date +%Y%m%d_%H%M%S)"
./scripts/assembly.sh --stage full --demos 5 --seed 42 \
  --keel-jitter 0.001 --headless --output-dir "$assembly_data_dir"
/home/cjh/anaconda3/envs/tpkmp/bin/python -B \
  assembly_collection/check_dataset.py "$assembly_data_dir" --require-success
```

随机范围为初始 keel XY 各 ±1 mm，失败数据也保留。检查器列出合格示范；`--require-success` 只要求至少一条合格，不表示全部成功。

画出本机已有装配数据的实际轨迹和夹爪／扶持状态：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
/home/cjh/anaconda3/envs/tpkmp/bin/python -B assembly_collection/visualize.py \
  assembly_collection/validation/handoff_full_09/episode_001 \
  --output "output/assembly_plot_cmd_$(date +%Y%m%d_%H%M%S).png"
```

装配数据为独立的双臂格式，不能直接传给 `learn.sh train`。当前完整装配也不是由 PPO 策略执行。

<a id="ppo"></a>

## 7. PPO：位置与力联合训练 v2

当前训练的是**左臂扶住 keel，右臂从孔前对齐、插入单颗螺栓、松爪并保持**。含实际法向力与摩擦力观测；还没有学习拾取、远距离搬运或完整四颗安装。

任务是 `NeroSingleBoltInsertPositionForce-v2`，58 维观测、6 维动作。旧位置版本模型不能用于续训。资产已生成，可以直接训练。

### 7.1 两个环境检查位置／力控制

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./ppo_assembly/run.sh check --num-envs 2 --controller expert --episodes 2 \
  --run-dir "ppo_assembly/checks/force_cmd_$(date +%Y%m%d_%H%M%S)"
```

`expert` 是环境检查控制器，不是训练好的 PPO 模型。检查会开爪、撤离并观察默认 10 秒的摩擦保持。

### 7.2 16 个并行环境，显示窗口

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./ppo_assembly/run.sh train --num-envs 16 --iterations 2000 --difficulty easy \
  --run-dir "ppo_assembly/runs/force_parallel_cmd_$(date +%Y%m%d_%H%M%S)"
```

16 组双臂场景共有 32 只机械臂，共享一个右臂策略。相机默认看第 0 个环境，拉远才能看到其他组；面板显示插入深度和轴向／侧向力。

### 7.3 16 个并行环境，无窗口训练并逐步增加难度

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./ppo_assembly/run.sh train --num-envs 16 --iterations 2000 \
  --difficulty auto --headless \
  --run-dir "ppo_assembly/runs/force_batch_cmd_$(date +%Y%m%d_%H%M%S)"
```

`iterations` 是更新次数，不是成功保证。`Ctrl+C` 保存当前模型。所有 PPO 模型、日志与验证仍在 `ppo_assembly/`。

### 7.4 实时训练曲线

在另一个终端复制：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./ppo_assembly/run.sh dashboard --port 6006
```

浏览器打开 <http://127.0.0.1:6006>。查看奖励、成功率及 `Force/*`、`Reward/force_cost`。平均回合奖励要等首个回合结束才出现。

### 7.5 自动选择最新 v2 模型，可视化回放

下面按保存时间选择 `force_*` 或 `position_force_*` 目录的最新 `latest.pt`。本机已有短测模型，所以可以直接运行；选中短测模型不代表它已学会插入。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ppo_model=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B -c \
  'from pathlib import Path; r=Path("ppo_assembly/runs"); p=[*r.glob("force_*/latest.pt"), *r.glob("position_force_*/latest.pt")]; print(max(p, key=lambda f: f.stat().st_mtime))')
./ppo_assembly/run.sh play --checkpoint "$ppo_model" --num-envs 1 \
  --episodes 5 --difficulty easy --seed 1000 --max-steps 2500 \
  --run-dir "ppo_assembly/evaluations/force_play_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 7.6 留出初始状态，验证较难位置

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ppo_model=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B -c \
  'from pathlib import Path; r=Path("ppo_assembly/runs"); p=[*r.glob("force_*/latest.pt"), *r.glob("position_force_*/latest.pt")]; print(max(p, key=lambda f: f.stat().st_mtime))')
./ppo_assembly/run.sh play --checkpoint "$ppo_model" --num-envs 1 --headless \
  --episodes 100 --difficulty hard --seed 1000 --max-steps 50000 \
  --run-dir "ppo_assembly/evaluations/force_generalization_cmd_$(date +%Y%m%d_%H%M%S)"
```

检查 `summary.json` 中的成功数和 `budget_exhausted`。使用留出状态，默认每次松手后观察 10 秒；这是孔前初始偏差的验证，不代表任意物体或完整装配泛化。

### 7.7 从最新 v2 模型继续训练

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ppo_model=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B -c \
  'from pathlib import Path; r=Path("ppo_assembly/runs"); p=[*r.glob("force_*/latest.pt"), *r.glob("position_force_*/latest.pt")]; print(max(p, key=lambda f: f.stat().st_mtime))')
./ppo_assembly/run.sh train --checkpoint "$ppo_model" --num-envs 16 \
  --iterations 1000 --difficulty auto --headless \
  --run-dir "ppo_assembly/runs/force_resume_cmd_$(date +%Y%m%d_%H%M%S)"
```

### 7.8 将最新 v2 训练曲线导出为 PNG

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
ppo_model=$(/home/cjh/anaconda3/envs/tpkmp/bin/python -B -c \
  'from pathlib import Path; r=Path("ppo_assembly/runs"); p=[*r.glob("force_*/latest.pt"), *r.glob("position_force_*/latest.pt")]; print(max(p, key=lambda f: f.stat().st_mtime))')
./ppo_assembly/run.sh curves "$(dirname "$ppo_model")"
```

<a id="scene"></a>

## 8. 场景、相机与帮助

### 8.1 Xbox 加载之前的 0928 抓取场景

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode dual --control xbox \
  --scene-usd assets/scenes/0928_grasp.usd
```

### 8.2 保留相机，但不打开相机窗口

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --mode dual --control xbox \
  --scene-usd assets/scenes/keel_assembly.usda --no-camera-views
```

`--no-camera-views` 只隐藏相机视图；`--no-wrist-cameras` 不创建配置里的相机。这两个参数也可用于 `teach-sync`。相机配置是 `assets/cameras/wrist_cameras.json`。

对照机械臂摆放物体时，把物体放在 `/World/UserScene/Environment` 下，点击面板 **Save Environment USD** 保存环境；输出是 `assets/scenes/workcell_*.usd`，下一次通过 `--scene-usd` 加载。窗口布局使用面板的布局保存按钮，与场景 USD 分开保存。

### 8.3 查看实际支持的参数

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
./scripts/sim.sh --help
./scripts/learn.sh --help
./scripts/assembly.sh --help
./ppo_assembly/run.sh train --help
```

详细说明：[仿真与 TP-KMP](README.md)、[硬件检查](hardware_checks/README.md)、[双臂装配](assembly_collection/README.md)、[PPO 位置与力](ppo_assembly/README.md)。

<a id="hand-eye"></a>

## 9. D435i + ChArUco 手眼标定

独立目录 [hand_eye_calibration](hand_eye_calibration/README.md)，已按 Calib.io **Rows=5、Columns=7** 配置 **7 列×5 行方格**（`squares_x=7, squares_y=5`）、40mm 方格、30mm marker。用户已确认 marker 编码为 **6×6**，当前配置使用标准 `DICT_6X6_50` 和编号 0–16；这些编号与标准 6X6_100/250/1000 的码字相同，采集命令直接读取配置。若实际编号或字典不同，修改 `hand_eye_calibration/board_config.json`。**2026-10-12 已确认眼在手上相机装在从臂，主从共享 can0 总线；当前应读取 can0/measured。**历史挂板会话接口从 can1 改名 can0，标签为 nero_can1。`--arm-id` 只是身份标签，不改变读取通道或来源；脚本只接收 CAN，不发送机器人命令。

### 9.1 检查相机与机械臂

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py
bash hand_eye_calibration/run.sh doctor.py --camera-profiles
bash hand_eye_calibration/run.sh doctor.py --can can0 --joint-source both
bash hand_eye_calibration/run.sh inspect_board.py
```

若提示“位姿与已采集数据太接近”，另开终端运行下面的只读监测，并在 20 秒内让装相机或标定板的实体臂运动。观察 `q(deg)` 是否随动作变化；报告含各关节变化范围和法兰变化。接口 DOWN 会提示，需要先确认接口对应关系。主从联动时两种来源都可能变化；从臂上的相机/板必须使用从臂的 measured 实际反馈，不能因 leader 也变化就选它。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 can1 --joint-source both --seconds 20
```

若当前对应的接口实际为 DOWN，再启用后重跑上面的监测；以下以当前 can0 为例：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

早期 00:01、00:53 的 can0 会话选到了另一只臂；01:43 的 can0 会话是在挂板臂接口改名后采集。接口确认不等同于主从位姿来源确认；历史 leader 会话只有在板装于主臂本体时才对应正确实体。不要仅凭 can0/can1、目录名称或主从同时运动判定来源，也不要混合两只臂的数据。

若接口已经 UP 但仍无完整反馈，先检查原始 CAN 帧数：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
/home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/probe_can.py --seconds 5
```

0 帧时先检查该路适配器的 CAN 端接线、机械臂供电和广播；更换 `--joint-source` 不能让没有到达的报文出现。新版 `monitor_motion.py` 也显示并保存原始帧数。通道与实体臂不确定时，可暂时只保留挂板那只臂的 USB-CAN 数据连接，再确认剩余接口名和 UP 状态。只接一路时，原始检查提示另一路不存在或 DOWN 是预期现象。详见[完整标定说明](hand_eye_calibration/README.md)。

新版采集窗口显示实际收到的关节角、法兰 XYZ 和位姿差；要加载代码更新，先退出原窗口，用新时间戳目录重启。重复判断的默认条件仍为至少 5mm 或 3°法兰变化；只改变七轴冗余姿态、保持末端不变时仍会拒绝。失败图像和差值保存在会话的 `rejections/`，不参与标定。

### 9.2 眼在手上：板固定，相机装于末端

当前相机装在从臂，主从共用 can0 总线；以下使用 **can0/measured** 读取从臂实际反馈，标签记为 nero_follower，并指定相机 262522071128 及 22:26 已通过的 RGB 内参。此代码块使用原来的 **7 列×5 行、40/30 mm、6×6 编码板**；若手眼也换成新 9 行×14 列板，使用 9.8 的完整命令。2026-10-12 USB 3.2、1280×720、30 fps 及 measured 七轴反馈均已实测正常。相机固定在末端，板固定在环境中，可以继续操控主臂，采样前等待从臂停稳。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
intr_file="hand_eye_calibration/datasets/intrinsics_9x14_20261012_222636/intrinsics.json"
calib_data="hand_eye_calibration/datasets/eye_in_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --board-config hand_eye_calibration/board_config.json --intrinsics "$intr_file" \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --target-samples 30 --out "$calib_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$calib_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json"
```

旧 `eye_in_can0_20261012_202718` 使用 leader，而相机实际装在从臂，不作为从臂手眼结果使用。新采集必须另建目录；修改旧 metadata 或重跑 solve 不能恢复当时未保存的从臂反馈。共享总线上的来源说明见 9.6。

### 9.3 眼在手外：相机固定，板装于末端

历史挂板会话重新插拔后对应 can0，实体标签为 nero_can1。先运行监测，并确认板实际安装在哪个实体臂。下面的 can0/leader 示例仅适用于板装于主臂本体；板在从臂时使用 `--joint-source measured --arm-id nero_follower`。主从同步变化不能证明两组角度完全一致。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 --joint-source both --seconds 20
```

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
calib_data="hand_eye_calibration/datasets/eye_out_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-to-hand \
  --can can0 --arm-id nero_can1 --joint-source leader \
  --target-samples 30 --out "$calib_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$calib_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json"
```

停稳后在预览窗口按空格保存；改变位置并绕不同轴转动，重复 30 次。Q/ESC 提前退出。标定另一只臂时，按其真实对应通道修改 `--can` 和 `--arm-id`，并另建数据目录；多相机同时接入时加入对应 `--serial`。

若报“关节反馈不完整”，先在原窗口按 Q/ESC 退出，用 `doctor.py --can can0 --joint-source both` 检查当前挂板臂。原始帧数为 0 时先恢复通信；有报文后验证对应来源随关节动作变化，再使用该来源重新采集。实际跟随臂需要其自身 measured 反馈，不能仅因 measured 缺帧就直接改用 leader。

若求解已保存但“阈值通过=False”，检查结果中的 `limits` 和 `holdout_report`。01:43 这组留出平移 3.228 mm、旋转 0.778°已通过，闭环重投影 6.794 px 超过 5 px，失败原因是图像闭环一致性。先检查板的平整刚性、末端连接、实际方格尺寸及停稳状态；改变安装后重新采集一组。详细误差见[数据诊断](hand_eye_calibration/diagnostics/dataset_014300_20261011/README.md)，RGB 内参检查命令见[完整标定说明](hand_eye_calibration/README.md)。

眼在手上输出 `T_flange_color.matrix`，眼在手外输出 `T_base_color.matrix`。均是 RGB 光学系到对应机器人坐标系、平移单位米。结果包含留出验证，失败仍保存报告并退出 2。RGB 内参、独立验证、手动位姿输入和板打印的命令，见 [完整标定说明](hand_eye_calibration/README.md)。

### 9.4 当前眼在手外结果的独立验证

最新结果为 `eye_out_can0_20261011_165821/result.json`：留出平移 RMS=3.243 mm、旋转 RMS=0.461°已通过；重投影 RMS=5.076 px，比 5 px 高约 1.52%，严格判定仍为 False。该阈值是当前程序的检查设置，接近阈值不代表实际装配精度已经确认。训练样本 0009 的平移闭环误差约 13.5 mm，验证时也应覆盖相似姿态附近。

保持本次相机、机械臂基座和板与末端的安装不变，另采 10 个新姿态。下面保留历史 leader 验证命令，只适用于板装在主臂本体；若历史采集时板在从臂，需改用 measured 重新标定，不能把以下命令当作有效从臂验证。验证直接使用已保存外参和板偏置，不重新求解。采集参数与本次 metadata 一致；USB 重新插拔后若接口改名，应先恢复并确认原对应关系。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
calib_result="hand_eye_calibration/datasets/eye_out_can0_20261011_165821/result.json"
check_data="hand_eye_calibration/datasets/check_out_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-to-hand \
  --can can0 --arm-id nero_can1 --joint-source leader \
  --serial 262522072061 --target-samples 10 --out "$check_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_result" \
  --dataset "$check_data" --out "$check_data/validation.json"
```

独立验证仍不等同于实物绝对定位精度，应在实际工作区域测量几个已知点的定位偏差，与螺栓和孔的配合容差比较。若板安装改变，旧结果中的固定板偏置不再适用于上述检查，需要重新采集并求解。

2026-10-11 已完成 10 个新姿态的独立验证，用户确认相机、基座和板安装保持不变：平移 RMS=4.338 mm（最大 7.786 mm）、旋转 RMS=0.918°（最大 1.385°）、闭环重投影 RMS=6.171 px（最大 11.979 px），严格判定仍为 False。10 个姿态中 4 个的图像闭环误差超过 5 px，单图 PnP 为 0.176–0.411 px。现有结果可保留作为粗定位候选，精确插孔前需核实板实际尺寸、平整度和夹持刚性，并用工作区域实测点验证。尺寸与固定条件确认后仍有偏差，再检查 RGB 内参和 FK。详细结果见[独立验证诊断](hand_eye_calibration/diagnostics/independent_171048_20261011/README.md)。

### 9.5 相机启动报 Couldn't resolve requests

CAN 检查通过后出现此错误，说明失败在 RealSense 流配置/启动。先枚举实际设备、USB 类型和支持的 RGB BGR8 模式；此检查不启动图像流：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py --camera-profiles
```

2026-10-12 相机 262522071128 最初协商为 USB 2.1，1280×720 仅支持 6/10/15 fps，默认 30 fps 无法解析；后续连接变为 USB 3.2，30 和 15 fps 均已实际取图通过。当前可使用 9.2 的 30 fps 指令。若再次连接为 USB2，可保留 1280×720 分辨率并明确选择 15 fps：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
calib_data="hand_eye_calibration/datasets/eye_in_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 15 \
  --target-samples 30 --out "$calib_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$calib_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json"
```

板预览也支持流参数：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh inspect_board.py \
  --serial 262522071128 --width 1280 --height 720 --fps 15
```

相机流模式检查会提前说明不支持的请求和可用参数。若模式支持但启动仍失败，关闭占用相机的 Viewer/其他采集程序，检查 USB3 接口、数据线及 RGB/深度组合。当前相机序列号与之前眼在手外的 262522072061 不同，内参和手眼结果分别生成；独立验证保持与对应采集相同的相机、分辨率和帧率。

### 9.6 主从共用 CAN：相机/板在哪只臂，就读取哪只臂的实际位姿

当前主从共享 can0。`leader` 读取主臂角度/发给从臂的关节指令，`measured` 读取实际关节反馈；切换来源通过 `--joint-source`，不要求换到 can1。2026-10-12 现场两种来源均完整新鲜，一组近同时快照的 FK 位置差约 3.85 mm、旋转差约 1.18°，说明跟随运动不能代替从臂实测位姿。这只是一次对照，不是旧数据集的误差归因统计。

从臂采集前的只读检查：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py --can can0 --joint-source measured
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 --joint-source both --seconds 20
```

确认实际反馈来自装相机/板的从臂后，眼在手上使用 9.2；眼在手外若板也装在从臂，则将 9.3 改成 measured 来源和从臂标签。若 measured 不完整，先恢复从臂的实际反馈，不能改读 leader 绕过检查。现场只读检查报告位于 [follower_feedback_20261012](hand_eye_calibration/diagnostics/follower_feedback_20261012/README.md)。

### 9.7 当前从臂结果仍未通过：独立检查 RGB 内参

`eye_in_can0_20261012_210908` 已正确使用 measured，留出平移 RMS=9.567 mm、旋转 RMS=1.497°、图像闭环 RMS=9.197 px。板尺寸已实测确认，当前 SDK／控制器位姿对照和静止噪声检查未显示明显软件换算或静止波动问题；训练图片内参对照有所改善但仍未通过，尚不能确定唯一根因。详见 [21:09 诊断](hand_eye_calibration/diagnostics/dataset_210908_20261012/README.md)。

**当前相机内参使用新板：Rows=9、Columns=14，20 mm 方格、15 mm marker、标准 5×5 编码，编号从 0 开始。**配置使用 DICT_5X5_100，完整板有 63 个 marker、104 个内角点；5X5_50 容量不足，当前 100/250/1000 的前 63 个码字一致。棋盘区域为 280×180 mm，连续四格应实测为 80 mm。旧手眼板配置保留。新板配置与参考 PDF 见 [board_configs](hand_eye_calibration/board_configs/README.md)。

先检查新板，让完整板入镜后按空格；退出检查后再采集：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh inspect_board.py \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --serial 262522071128 --width 1280 --height 720 --fps 30
```

以下只采图片，不读取 CAN。相机仍用 262522071128、1280×720@30 fps；35 张需包含不同距离、不同方向的倾斜，覆盖图像中央和四周。保持板自身平整，停稳后按空格。本次每张至少要求 40 个角点，允许部分板入镜以覆盖图像边缘。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
intr_data="hand_eye_calibration/datasets/intrinsics_9x14_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode intrinsics \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --min-corners 40 --target-samples 35 --out "$intr_data"
bash hand_eye_calibration/run.sh calibrate_intrinsics.py --dataset "$intr_data"
```

`intrinsics.json` 的 `passed=true` 只表示内参图片检查通过。之后使用该内参重新做 9.2 的手眼采集时，在 collect 命令加 `--intrinsics "$intr_data/intrinsics.json"`；完整手眼仍须独立通过验证。若内参确认后跨姿态误差仍明显，继续用返回相同关节姿态的重复检查排查结构／支架受力，再核查机械臂零位、几何和名义 FK 的实物精度，避免只重复同样条件的采集。

内参可以用新板、手眼再用旧板。手眼拍摄的实物若是旧 7 列×5 行、40/30 mm 板，则保持 9.2 默认板配置；若手眼也实际换成新板，collect 命令还要加新板的 `--board-config`。calibrate_intrinsics.py、solve.py 和 validate.py 会从各自数据集读取板参数，无需额外指定板，也不会改写旧会话。

<a id="hand-eye-new-intrinsics"></a>

### 9.8 使用已通过的新内参和新 9×14 板做眼在手上标定

35 张新板图片的内参结果已通过，28 张训练 RMS≈0.150 px、7 张留出 RMS≈0.139 px。文件是 `hand_eye_calibration/datasets/intrinsics_9x14_20261012_222636/intrinsics.json`，对应相机 262522071128、1280×720@30 fps。

已用该内参离线重算旧 21:09 的 30 张 measured 照片，保存为 `datasets/eye_in_can0_20261012_210908_intr_222636`。同一 24/6 划分下，留出平移从 9.567 降到 7.502 mm、旋转从 1.497 降到 1.049°，图像闭环为 9.065 px，仍未通过。该副本保留原图、机械臂位姿和拍摄时间，仍是同一次采集，不能当作独立验证。对照见 [22:26 内参与重算说明](hand_eye_calibration/diagnostics/intrinsics_222636_20261012/README.md)。

下面是新的实物采集：固定新 9 行×14 列板，相机仍刚性安装在从臂；通过主臂操控可以继续，位姿必须读取从臂 measured。停稳后按空格，改变位置、距离并绕不同方向倾斜；每张至少 40 个角点。终端应显示加载的内参路径以及 fx≈899.940、fy≈900.556。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
intr_file="hand_eye_calibration/datasets/intrinsics_9x14_20261012_222636/intrinsics.json"
calib_data="hand_eye_calibration/datasets/eye_in_9x14_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --intrinsics "$intr_file" \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --min-corners 40 --target-samples 30 --out "$calib_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$calib_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json"
```

这组通过后，保持相机支架、机械臂基座和固定板的位置不变，在同一终端另采 10 个新姿态做独立验证，不重新求解：

```bash
check_data="hand_eye_calibration/datasets/check_in_9x14_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --intrinsics "$intr_file" \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --min-corners 40 --target-samples 10 --out "$check_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json" \
  --dataset "$check_data" --out "$check_data/validation.json"
```

以后更换相机内参时，可以先离线重算保存过原始图片的手眼数据。下面不读取硬件，自动使用原数据集的板配置，保留所有样本和原角点编号集合；原始数据不改动，输出需要新目录：

```bash
recomputed_data="hand_eye_calibration/datasets/eye_in_recomputed_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh reprocess.py \
  --dataset hand_eye_calibration/datasets/eye_in_can0_20261012_210908 \
  --intrinsics hand_eye_calibration/datasets/intrinsics_9x14_20261012_222636/intrinsics.json \
  --out "$recomputed_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$recomputed_data"
bash hand_eye_calibration/run.sh validate.py --result "$recomputed_data/result.json"
```

重算保留原机械臂位姿来源，不能修复旧 leader 数据选错实体臂的问题，也不能代替重新采集的独立验证。

<a id="real-tpkmp"></a>

## 10. 真机 TP-KMP：单只从臂抓取与搬运

独立目录 `real_tpkmp/`，完整按键、TCP/工作空间配置、新位置参考点和独立验证见 [真机 TP-KMP 说明](real_tpkmp/README.md)。先审核该目录的 `config.json`；默认工具参考点与法兰重合。以下命令在同一终端分步执行，采集时直接拖动主臂，记录共享 can0 上从臂 measured 反馈。

统一编辑 [camera_config.json](real_tpkmp/vision/camera_config.json)：直接填写相机序号、RGB 分辨率/帧率、内参 K/畸变和手眼矩阵等参数，不需要填写参数文件路径。`object_id` 选择物体名称，`bind-grasp --update-config` 可把抓取参数自动填入 `grasp`。修改后重启。`collect.sh` 默认读取此文件，使用 [V 定位示教流程](real_tpkmp/README.md#视觉定位后示教推荐)：每轮 V 定位并预览接受 → 回起点 R → N 六阶段 → S 保存；固定放置点首次 L 记录后跨轮复用。首次需完成有效手眼与抓取绑定。下面保留人工 P/L 示例，已加 `--manual`。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
task_id="pick_place_$(date +%Y%m%d_%H%M%S)"
dataset="real_tpkmp/datasets/$task_id"
model_dir="real_tpkmp/models/$task_id"
bash real_tpkmp/collect.sh --manual --can can0 --demos 8 --out "$dataset"
```

每组先 P 记录抓取 TCP，L 记录放置 TCP，回到统一起点后 R 开始；N 依次进入夹取、抬升、搬运、松开、撤离阶段，成功后 S 保存，X 丢弃，Q 退出。键盘只标记采集阶段。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/train.sh --dataset "$dataset" --out "$model_dir"
```

先生成离线预览，不连接 CAN：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
preview_dir="real_tpkmp/runs/${task_id}_preview_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/test_real.sh --model "$model_dir/model.npz" \
  --frames "$dataset/demo_001_frames.json" --out "$preview_dir"
```

审核 `preview.png` 和 `preview.json`。先通过主臂把从臂移到所列起点、匹配夹爪开度，随后停止主臂继续发令，保持电脑到已使能从臂的 CAN 通路；主臂静止仍可能发令，脚本会拒绝争用总线。检查仅包含关节限位、TCP 工作空间和禁入盒，没有整臂碰撞规划。真机执行命令：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/test_real.sh --plan "$preview_dir/plan.npz" --execute --can can0 \
  --out "real_tpkmp/runs/${task_id}_real_$(date +%Y%m%d_%H%M%S)"
```

留出失败/合成模型不能真机执行。Ctrl+C 或反馈/跟踪错误触发 SDK 阻尼急停；脚本不切主从模式、不使能、不回零。实际跟踪与误差输出到运行目录，抓取和放置是否成功需观察实物。

<a id="real-vision"></a>

### 10.1 真机视觉：D435i 识别 keel 并生成抓取目标

视觉内容位于 `real_tpkmp/vision/`，完整安装、拍照、识别、一次抓取关系绑定和新位置泛化命令见 [视觉使用说明](real_tpkmp/vision/README.md)。先检查 GPU 环境与权重：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/doctor.sh
bash real_tpkmp/vision.sh self-test
```

当前采用 SAM3 文本分割和 FoundationPose RGB-D CAD 配准，支持眼在手上/眼在手外。需要与当前相机安装、内参对应的手眼 `result.json`；曾经失败的手眼结果只能诊断，不能驱动真机。相机没有自动得到物体到夹爪的抓取关系，需按视觉文档示教一次。定位输出的 `frames.json` 可直接交给本节 `test_real.sh` 预览；只有通过检查且仍新鲜的真实定位结果才能执行。

<a id="meshy"></a>

### 10.2 Meshy：通过物体照片生成 3D 模型

详细拍照、单图/多图 API 和模型接入说明见 [Meshy 文档](real_tpkmp/vision/meshy/README.md)。**下载模型放哪里、导入后如何采集/训练/测试**见 [完整操作流程](real_tpkmp/vision/meshy/WORKFLOW.md)。先在 Meshy 网页生成并下载 STL/GLB 到 `real_tpkmp/vision/meshy/downloads/keel/`，再运行离线导入。这里的尺寸使用已确认的 keel 150×98×23 mm；其他物体需填写其实际尺寸。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh import \
  --mesh real_tpkmp/vision/meshy/downloads/keel/model.glb \
  --object-id keel_meshy --measured-extents-mm 150 98 23 \
  --prompt 'blue metal bracket'
```

想用脚本调用 API，先将同一实物不同角度的照片放在下列路径。以下只进行本地检查，不上传、不扣积分；单图时只保留一个图片路径：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
meshy_run="real_tpkmp/vision/meshy/jobs/keel_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/meshy.sh generate --dry-run \
  --images real_tpkmp/vision/meshy/images/keel/front.jpg \
           real_tpkmp/vision/meshy/images/keel/side.jpg \
           real_tpkmp/vision/meshy/images/keel/back.jpg \
  --out "$meshy_run"
```

在同一终端设置你自己的 key 后提交，会上传照片并消耗 Meshy 账户积分。此处使用交互输入，密钥不写入命令历史：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
read -r -s -p 'Meshy API key: ' MESHY_API_KEY
echo
export MESHY_API_KEY
bash real_tpkmp/vision/meshy.sh submit --run "$meshy_run"
bash real_tpkmp/vision/meshy.sh import --mesh "$meshy_run/model.stl" \
  --object-id keel_meshy_api --measured-extents-mm 150 98 23 \
  --prompt 'blue metal bracket'
unset MESHY_API_KEY
```

若 submit 超时云端未完成，先 `bash real_tpkmp/vision/meshy.sh resume --run "$meshy_run"`，成功下载后再 import。新配置位于 `real_tpkmp/vision/models/keel_meshy_api/object.json`，传给 `estimate.sh --object`；需要重新绑定抓取关系。新模型不覆盖原 CAD，均匀缩放不能修复生成的孔径或薄壁误差。

<a id="vision-tracking"></a>

### 10.3 单独测试视觉跟踪：不连接 CAN

不需要通过手眼标定。默认使用鼠标框选而非文字描述，固定 D435i，查看模型轮廓、坐标轴和延迟。停止占用相机的其他标定/采集程序后，在桌面终端运行：

启动时默认读取 [camera_config.json](real_tpkmp/vision/camera_config.json)。`intrinsics` 直接填写内参矩阵 K、畸变模型/系数和对应设备/分辨率信息；当前已填实际运行记录中的出厂值。设为 null 或临时加 `--factory-intrinsics` 则从当前相机读取出厂参数。跟踪不使用手眼结果，未填写手眼矩阵也可运行。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/track.sh \
  --object real_tpkmp/vision/objects/keel.json \
  --mask-method box --expected-distance-mm 50 \
  --out "real_tpkmp/vision/runs/track_keel_$(date +%Y%m%d_%H%M%S)"
```

模型加载后拖框选完整物体，框尽量贴近轮廓、少含背景，Enter/空格确认，保持物体不动直到配准完成。F 重新框选，S 采 10 个静止有效帧看抖动；A 采平移起点，等完成后让物体保持朝向平移尺测 50 mm，停稳按 B；相机始终固定。R 重选并定位，P 保存快照，Q 退出。左侧是当前画面，右侧是已处理帧及模型叠加；绿色为掩膜、紫色为提示框。Meshy 模型把 `--object` 换成其导入后的 `object.json`。

静态 `estimate.sh` 同样支持 `--mask-method box`。框选模式不需要 JSON 中的 `text_prompt`；若要继续文字自动分割，显式使用 `--mask-method sam3`。

小抖动不代表绝对准确；keel 的近对称结构也可能有 180°姿态歧义。完整测量条件、曲线、回放和诊断见 [独立视觉测试说明](real_tpkmp/vision/TRACKING.md)。
