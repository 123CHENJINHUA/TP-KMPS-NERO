# 从下载模型到真机采集、训练和测试

适用于当前单只从臂、主从共享 `can0`、拖动主臂示教的抓取/搬运任务。以下以 **keel 实物 150×98×23 mm** 为例；其他物体必须改成实际尺寸、名称和分割描述。下载的模型要对应桌面上的真实物体。

这里有两类模型：3D 网格用于相机定位；TP-KMP 的 `model.npz` 用于学习示教轨迹。生成或下载 3D 网格不会自动生成真机抓取示教。当前示教采集保存从臂 measured 关节、FK TCP、夹爪实际值/控制目标、阶段和任务参考系，不连续录制相机视频。

先完成“人工参考点采集 → 训练 → 原位置预览/回放”，再加入“相机识别新位置”。前一段不依赖视觉或手眼标定。

## 1. 下载模型放哪里

原始下载文件放到：

```text
/home/cjh/Downloads/TP-KMPs-nero-isaacsim/real_tpkmp/vision/meshy/downloads/keel/model.glb
```

也可使用 `model.stl`。压缩包先解压；OBJ 需要保留相邻的 MTL 和纹理文件。每个物体使用自己的子目录。

先创建文件夹，再将下载文件复制进去：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
mkdir -p real_tpkmp/vision/meshy/downloads/keel
```

导入时校正尺度，并生成现有视觉程序能读取的配置：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh import \
  --mesh real_tpkmp/vision/meshy/downloads/keel/model.glb \
  --object-id keel_meshy \
  --measured-extents-mm 150 98 23 \
  --prompt 'blue metal bracket'
```

输出：`real_tpkmp/vision/models/keel_meshy/object.json`、`model.stl`、`model.glb`、`preview.png` 和尺寸报告。先看预览是否与实物相符。模型比例不符会拒绝生成配置；网格斜放可用 `--rotate-xyz-deg RX RY RZ` 调整或在编辑器中摆正。重复导入需用新名字或新 `--out`，不会覆盖旧模型。

导入不发机器人指令，也不调用 Meshy API。更换模型之后，视觉抓取关系需要重新绑定。

## 2. 采集真机示教

检查 `real_tpkmp/config.json`：实体臂为 `nero_follower`，通道 `can0`，TCP 定义在采集、训练、测试中保持一致；当前默认工具参考点与法兰重合。若要改成实测夹爪中心，先修改配置再采集。工作空间和 TCP 禁入盒按实际桌面设置。

在同一终端依次执行后续命令，保留变量：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
task_id="keel_pick_place_$(date +%Y%m%d_%H%M%S)"
dataset="real_tpkmp/datasets/$task_id"
model_dir="real_tpkmp/models/$task_id"
object_cfg="real_tpkmp/vision/models/keel_meshy/object.json"
bash real_tpkmp/collect.sh --manual --can can0 --demos 8 --out "$dataset"
```

这里显式使用 `--manual`，可先采集人工 P/L 示教；若要每轮相机自动定位，编辑 [camera_config.json](../camera_config.json) 中的 `object_id`、`hand_eye`、`grasp` 等直接参数，再使用 [V 定位示教流程](../../README.md#视觉定位后示教推荐)，启动时省略 `--manual`。

保留主从跟随，直接拖动主臂和操作试教夹爪；程序只读取从臂实际反馈，不主动运动。先确认屏幕上的关节角确实随对应从臂变化。

每组按以下顺序操作：

| 按键 | 在什么时候按 |
| --- | --- |
| P | TCP 到真实抓取位姿并停稳，记录抓取参考 |
| L | TCP 到物体放稳、尚未松爪的位姿并停稳，记录放置参考；随后恢复物体到抓取位置 |
| R | 回到统一起点、夹爪打开，开始接近 |
| N（第 1 次） | 到抓取点，闭爪之前；进入夹取阶段，然后闭爪并等待夹稳 |
| N（第 2 次） | 开始抬升之前 |
| N（第 3 次） | 开始搬运之前；搬运到放置参考位姿，让物体得到支撑 |
| N（第 4 次） | 松爪之前；然后打开夹爪并等待放稳 |
| N（第 5 次） | 开始撤离之前 |
| S | 撤离结束，确认本次确实抓起、搬运并放稳，保存成功示教 |
| X / Q | 丢弃当前尝试 / 结束采集 |

每阶段至少 0.2 秒。每组重新 P/L/R；抓取点是夹爪/TCP 的目标位姿，不是物体中心。建议采 8–15 组，先在 2–5 cm 范围内改变摆放位置，使用一致的抓取方向、起点和阶段顺序。位置要覆盖后续测试范围。

保存 `demo_001.npz` 等数据，以及各自的 `demo_001_frames.json`。只有 operator 确认成功并按 S 的示教进入训练。这个步骤可以在手眼标定尚未通过时先进行。

## 3. 训练及独立验证

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/train.sh --dataset "$dataset" --out "$model_dir"
```

查看 `validation.json`、`holdout_01.png` 等轨迹对比，以及终端的“留出阈值通过”。训练输出的 `model.npz` 与 3D 网格是不同文件。训练/留出都由真实示教得到，不需要重新训练 SAM3 或 FoundationPose。

想单独检验泛化，再摆几个没参与原训练的位置，按相同按键流程另采 **至少 4 组**。下面只验证已有 TP-KMP，不重新训练：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
check_data="real_tpkmp/datasets/${task_id}_check_$(date +%Y%m%d_%H%M%S)"
check_out="real_tpkmp/runs/${task_id}_validation_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/collect.sh --manual --can can0 --demos 4 --out "$check_data"
bash real_tpkmp/validate.sh --model "$model_dir/model.npz" \
  --dataset "$check_data" --out "$check_out"
```

这里比较预测轨迹与新示教，实际抓取成功还要通过真机执行检查。

## 4. 先在一个示教位置预览和测试

将物体和放置目标恢复为第一组示教的位置：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
preview_dir="real_tpkmp/runs/${task_id}_preview_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/test_real.sh --model "$model_dir/model.npz" \
  --frames "$dataset/demo_001_frames.json" --out "$preview_dir"
```

这条只计算轨迹、IK 和限速，不连接 CAN。看 `preview.png` 和 `preview.json`，确认能执行及起点关节/TCP/夹爪状态。

通过主臂把从臂移到预览要求的起点、匹配夹爪开度，然后停止主臂继续发送控制指令，同时保留电脑到已使能从臂的 CAN 通路。主臂静止通常仍在发令，不能仅靠松手；脚本会拒绝总线争用。它不切主从角色、不使能、不回零。当前检查包括关节限位和 TCP 范围/禁入盒，没有整臂和场景的碰撞规划。

条件满足后，这条才会实际运动：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/test_real.sh --plan "$preview_dir/plan.npz" --execute --can can0 \
  --out "real_tpkmp/runs/${task_id}_execution_$(date +%Y%m%d_%H%M%S)"
```

查看 `execution.json`、`execution.png` 和实物。跟踪通过、夹起物体、搬运不滑落、放置稳定分别检查；轨迹误差通过不等于实际任务成功。Ctrl+C 或反馈/跟踪异常会尝试 SDK 阻尼急停。

没有视觉也可以测试新位置：`bash real_tpkmp/run.sh frames.py --can can0 --out real_tpkmp/runs/新的参考文件.json`，停稳后 S/P/L 分别记录起点/抓取/放置，W 保存；再把该文件交给 `test_real.sh --frames` 预览。此处的 S 是记录起点，与采集脚本的 S 保存示教含义不同。

## 5. 加入视觉：为新模型绑定一次抓取关系

需要与当前相机安装和内参一致、通过验证的手眼结果。2026-10-21 核对当前目录：现有两份 `eye_in_can0_20261012_210908*/result.json` 均为 `passed=false`，只能诊断；先完成新的合格标定，再进行视觉驱动的真机测试。不要直接把文件中的 passed 改成 true。

在 [camera_config.json](../camera_config.json) 中把 `object_id` 改为导入后的名称（本例为 `keel_meshy`），直接填写实际内参和通过验证的手眼矩阵及对应标定信息。程序启动时自动读取，不需要内参/手眼路径。其余变量继续使用前面的终端：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
reference_run="real_tpkmp/vision/runs/${task_id}_reference_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/capture.sh --can can0 \
  --out "$reference_run/capture"
bash real_tpkmp/vision/estimate.sh --capture "$reference_run/capture" \
  --object "$object_cfg" --mask-method box --out "$reference_run/pose"
```

拍照时物体和从臂停稳，目标无遮挡。定位时在弹出的冻结图像上框住完整物体，框贴近轮廓、少包含背景，Enter 确认。检查 `segmentation.png`、`overlay.png`、几何质量和 `observation.json` 的 `usable_for_robot=true`。眼在手上使用每帧 measured 法兰位姿转换，不用主臂 leader 指令。只想看相机定位时加 `--camera-only`，但结果不能驱动真机。

保持物体没有移动，运行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision.sh bind-grasp --can can0 \
  --observation "$reference_run/pose/observation.json" \
  --out "$reference_run/grasp.json" --update-config
```

拖动主臂，让从臂 TCP 到达正确抓取位姿，停稳后按空格记录；先不要把物体抓走。这一步只读取关节反馈，保存 `T_object_tcp`，并将完整绑定参数直接写入相机配置的 `grasp`，无需填写文件路径。

## 6. 换位置，视觉定位后生成 TP-KMP 测试轨迹

让物体在训练覆盖范围附近换个位置，先用 1–2 cm 的变化测试。恢复主从跟随用于摆位/拍照，保持同一模型、TCP 和抓取方式：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
new_run="real_tpkmp/vision/runs/${task_id}_new_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/capture.sh --can can0 \
  --out "$new_run/capture"
bash real_tpkmp/vision/estimate.sh --capture "$new_run/capture" \
  --object "$object_cfg" --mask-method box --out "$new_run/pose"
bash real_tpkmp/vision.sh frames \
  --observation "$new_run/pose/observation.json" \
  --template-frames "$dataset/demo_001_frames.json" --out "$new_run/frames.json"
bash real_tpkmp/test_real.sh --model "$model_dir/model.npz" \
  --frames "$new_run/frames.json" --out "$new_run/preview"
```

视觉更新抓取点，起点和放置点沿用模板；不自动识别新的放置目标。**每次 estimate 都带 `--object "$object_cfg"`**，否则默认会定位原来的 keel CAD。

看过新的 overlay/preview，将从臂移到要求的起点并释放主臂控制权后，再执行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/test_real.sh --plan "$new_run/preview/plan.npz" --execute --can can0 \
  --out "$new_run/execution"
```

观察默认 300 秒有效，超时或物体再次移动需重新定位和规划。模型/物体坐标系、夹爪 TCP 或手眼安装变更后需重新绑定抓取关系。Meshy 网格的孔径和薄壁误差不能靠统一缩放修复，这条流程先用于抓取搬运。

新终端不会保留上述变量；重新打开时将 dataset/model_dir/object_cfg/reference_run 等设置为已保存的实际路径。每次输出用新目录，不覆盖旧采集或模型。
