# 抓取物体模型

每个物体放在 `assets/objects/<名称>/`，启动时使用 `--mode grasp --object <名称>`。

## keel

- `keel/keel.STL`、`keel/keel.usd`：从 `/home/cjh/Downloads/new_urdf/` 复制的原文件，未修改。
- `keel/keel_grasp.usda`：仿真使用的派生模型，单位为米、Z 轴向上，包含真实网格的凸分解碰撞。
- `keel/object.json`：质量、夹持宽度、物体原点高度、平台尺寸以及源文件哈希。

用户确认外形为 **150×98×23 mm**。原始 USD 标注为厘米，直接加载会得到 1.5 m 长的物体。
派生模型使用 STL 三角网格，按 0.001 缩放到米，并减去原始坐标中的 `[75, 49, 22]`。
这样物体原点就是两条凸壁之间的抓取参考点。凸壁外侧名义间距约 38 mm，抓取点靠近上沿，以避开底座。
源 USD 的 OmniPBR 材质依赖未带入派生文件，派生文件使用独立的蓝色网格显示颜色。

仿真质量暂设为 **80 g**，这不是实物称重结果。平台为 240×220 mm。
夹爪通过碰撞和摩擦夹持，不设置吸附约束。仅在每次试验初始化时复位物体和机械臂。

```bash
./scripts/sim.sh --mode grasp --object keel --grasp-spread 0 \
  --demos 2 --duration 10 --output-dir recordings/keel_view_01
```

首次测试固定位置，两次均通过，报告在 `recordings/keel_check_01/summary.json`。
这只验证了固定姿态、固定位置和当前仿真质量；不能代表任意摆放、所有物体或真实抓取效果。

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

## 新增其他物体

后续新增物体时，保留原始文件，另建归一化的 USD 和 `object.json`。需要分别确定单位、支撑姿态、
夹持参考点、夹持宽度和碰撞近似，不能仅换模型文件就假定原抓取动作适用。
当前模板仍限定 25–40 mm 的名义夹持宽度以及不超过 100 g 的质量。
不同物体的数据应分开训练；技能文件保存物体名称和抓取参数，回放会加载对应模型。
