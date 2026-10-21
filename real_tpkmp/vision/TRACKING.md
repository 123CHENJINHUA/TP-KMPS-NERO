# 先单独验证 D435i 视觉定位和连续跟踪

入口 `track.sh` **只打开相机，在相机 COLOR optical 坐标系中工作**。不读 CAN、不使用手眼结果、不启动 Isaac Sim，也不发机械臂运动指令。输出始终是诊断记录，不生成可直接执行的机器人目标，因此可以先验证视觉，之后再处理手眼和抓取。

默认采用**鼠标框选 + SAM3 单实例分割**，不需要填写文字 prompt。首次用分割掩膜和 FoundationPose `register` 配准；后续使用官方 [`track_one`](https://github.com/NVlabs/FoundationPose/blob/main/estimater.py)。模型常驻两个独立 GPU worker；当前帧重新分割并检查 RGB-D 几何。提示框通过目标区域的前后向光流更新，保持用户选框的范围，避免少量掩膜噪点使框不断膨胀。

框选使用 SAM3 的 `predict_inst` 单实例接口，输入是框的 XYXY 像素坐标，不把矩形直接当作物体掩膜。框内分割丢失/深度不足时等待用户重新框选；位姿质量失败但分割仍有效时继续用同一目标重新配准。文字 `--mask-method sam3` 和颜色 `hsv` 仍可显式选择。

## 1. 运行

先结束占用 D435i 的标定/采集程序。相机序号、分辨率、帧率和内参从 [camera_config.json](camera_config.json) 读取；在正常桌面终端运行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
track_run="real_tpkmp/vision/runs/track_keel_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/track.sh \
  --object real_tpkmp/vision/objects/keel.json \
  --mask-method box --expected-distance-mm 50 \
  --out "$track_run"
```

当前 JSON 已填相机 `213522070575`、RGB `1280×720@30`，原生深度为 `640×480@30`。以后更换相机直接修改该文件，无需修改 Python。先做 RGB 去畸变和深度点云到 COLOR 的投影，再将配准后的 RGB/深度一起缩到最多 640 像素宽，并同步缩放 K。深度始终为光轴 Z、单位米。

JSON 顶层填写一次 serial/width/height/fps；`intrinsics` 只填 K、畸变模型和系数，当前已填该相机实际运行记录中的出厂值。自行标定后直接替换这些数值；把整个 `intrinsics` 设为 null 或临时加 `--factory-intrinsics` 可在启动时读取出厂参数。旧版 `--intrinsics 路径/intrinsics.json` 仍兼容。回放使用原采集记录中的内参。跟踪不使用手眼结果或抓取绑定，未填写手眼矩阵也可运行。

模型加载完成后会弹出**冻结的当前帧**：鼠标拖框，**框住完整物体并贴近外轮廓，少包含背景**，按 Enter/空格确认。先保持物体不动，直到显示配准结果再缓慢移动。C/取消后回到预览，可按 F 再选。框的松紧会影响掩膜质量；右侧绿色是分割掩膜，紫色是本帧提示框，黄色轮廓是 CAD 投影。

使用 Meshy 导入后的新模型时，只把 `--object` 换成 `real_tpkmp/vision/models/keel_meshy/object.json`。框选模式忽略 JSON 的 `text_prompt`，选框坐标和对应图像保存在本次运行的 `selections/`，不写回物体配置。若显式改为 `--mask-method sam3`，才读取 JSON 的 `text_prompt` 或 `--prompt` 描述。

若实物确实为蓝色且场景只有一个蓝色目标，原始 keel 配置也可改用 `--mask-method hsv`，减少分割开销。新 Meshy 配置没有默认 HSV 范围，先使用框选。文字模式的 `--instance` 是按当前置信度排序的编号，框选模式使用空间位置指定一个实例，不使用该排名参数。

## 2. 窗口怎么看

- 左侧：当前 RGB，相机持续采集。
- 右侧：**实际送入网络的那一帧**及模型轮廓、包围盒、坐标轴，不把旧位姿叠加到新的左侧画面。
- 红 X、绿 Y、蓝 Z 是模型的原始坐标轴；窗口显示的 CENTER XYZ 是模型包围盒中心在相机中的位置，单位 mm。相机 X 向右、Y 向下、Z 向前。
- `REGISTER` 首次/重新配准；`TRACK` 连续跟踪；`LOST / REJECTED` 表示本帧没有通过检查。被拒绝的候选用红色轮廓表示。框选目标丢失后，按 F 在新位置重新选框；不自动改用文字去寻找其他相似物体。
- `Processing` 是本帧分割、网络和质检的耗时；`submitted-to-result` 是提交处理后的耗时，不包括相机曝光时间。实时窗口的 `age` 是该已处理帧距现在的主机接收时间差。
- IoU 是模型投影轮廓与目标 mask 的重合度；depth median/p90 是模型渲染深度与相机深度的误差。它们是质检指标，不能直接当作 6D 绝对精度。

相机采集 30 fps，网络处理速率取决于实际耗时。最多只有一个处理任务在途，空闲时取最新帧；先用缓慢移动测试。首次配准/重新配准通常比持续跟踪慢，模型加载和 CUDA 初始化期间仍可在窗口中退出。

## 3. 按键和三项实物检查

先固定相机。如果是眼在手上，把机械臂停住；本入口不补偿机械臂/相机运动。

| 按键 | 作用 |
| --- | --- |
| F | 暂停并在新的冻结帧上重新框选；取消后等待再次按 F |
| S | 开始一个静止窗口，默认累计 10 个有效处理帧，输出位置/旋转抖动 |
| A | 相机和物体停稳，平均记录平移检查的起点 |
| B | 平移后停稳，平均记录终点，比较 A/B 的模型中心位移 |
| R | 框选模式重新选框和配准；其他模式重新配准。清除当前测量锚点 |
| P | 保存当前已处理帧的叠加图、RGB-D/K/mask 和位姿 |
| Q / Esc | 退出并保存统计和曲线；Ctrl+C 也可退出 |

**静止检查**：相机和物体都不动，跟踪有效后按 S，保持至窗口完成。查看位置 RMS 抖动和旋转 RMS 抖动。窗口内发生丢失/重新配准会取消测量；明显移动会标为不稳。小抖动只表示稳定，可能仍存在固定偏差。

**尺测平移检查**：在静止位置按 A，等提示 `A saved`；保持相机固定和物体朝向不变，用尺子/限位工装将物体平移 **50 mm**，停稳后按 B。启动命令已经设置 `--expected-distance-mm 50`，程序报告测得距离及距离误差。若你的实际位移为其他值，启动时改成尺测值。两窗口之间跟踪丢失或重新配准会使 A 失效，需要重做；旋转变化超过 3°时不输出有效距离误差。单次方向的距离检查不能代表完整位置和姿态绝对精度。

**旋转和遮挡检查**：缓慢旋转物体，观察模型边缘、孔轮廓和坐标轴是否跟随；短暂遮挡或移出视野，观察是否报告丢失，然后按 F 在恢复的目标上重新框选。keel 接近对称，可能存在相差约 180°的姿态歧义；轮廓/深度匹配和稳定性检查不能证明首尾方向唯一。要评估旋转角度的绝对误差，需要独立角度基准/夹具。

当前过滤阈值沿用物体配置：原始 keel 的 IoU≥0.45、深度误差中位数≤6 mm、p90≤15 mm；另外默认连续两帧模型中心跳变≤80 mm、旋转≤45°、帧间隔≤10 s。这些是拒绝明显失败的初始门限，不是抓取或插孔精度承诺。

## 4. 输出文件

| 文件 | 内容 |
| --- | --- |
| `session.json` | 物体、相机、内参、模型校验值和无 CAN 的诊断标记 |
| `frames.jsonl` | 每个处理帧的相机位姿、处理时间、K、质量和丢失原因 |
| `latest.png` | 最近一次处理帧的模型叠加图 |
| `mask.png`、`segmentation.png` | 当前掩膜及绿色分割预览；没有分割输出时显示 NO VALID MASK。掩膜显示不代表通过深度/位姿质检 |
| `selections/` | 每次框选的归一化坐标、对应冻结帧及相机 K |
| `tracking.png` | 退出时生成相对首次有效位置的变化、轮廓 IoU、深度误差曲线；红线为拒绝帧 |
| `measurements.jsonl` | S/A/B 的测量结果 |
| `snapshots/` | P 或每 30 个处理帧保存的图像、RGB-D 和位姿；`--save-every 0` 关闭定期快照 |
| `pose.log`、`sam3.log` | 常驻网络的初始化和运行日志 |
| `summary.json` | 有效/拒绝帧计数和错误信息；始终 `usable_for_robot=false` |

无桌面环境的框选模式需要 `--headless --box x0 y0 x1 y1`，坐标为去畸变图像中的**归一化 0–1 XYXY**；例如 `--box 0.29 0.27 0.72 0.72` 只适用于后面的合成例子，不是实物通用框。可加 `--seconds 30 --measure-static`，时长含模型加载。实时无窗口模式丢失后会退出，需提供新框重新启动；有窗口时按 F 即可。CUDA 不可见时在正常主机终端检查 `nvidia-smi` / `doctor.sh`。GPU 内存不足且实物适合蓝色分割时，可使用 HSV 路径。

## 5. 不打开硬件的回放与验证

已有 `capture.sh` 输出的 RGB-D 目录可以逐帧回放。这里会真正运行分割与跟踪网络，不打开相机或 CAN：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/track.sh \
  --replay real_tpkmp/vision/runs/tracking_gpu_01_capture \
  --headless --mask-method sam3 --prompt 'blue object' \
  --out "real_tpkmp/vision/runs/tracking_replay_$(date +%Y%m%d_%H%M%S)"
```

以上内置回放是明确标记的合成序列：静止、8/16 mm 平移、目标消失和恢复；验证过 `register → track → lost → register → track`。网络没有接收真值位姿。合成输入及重复静止图像不包含真实相机噪声，不能据此评估实物精度，详见 [CHECKS.md](CHECKS.md)。

框选路径的回放可加 `--mask-method box --box 0.29 0.27 0.72 0.72 --max-frames 4`，替换上面命令的文字分割参数；框只在起点给定，后续随光流变化。目标消失后框选模式会等待新框。

纯离线测试不加载网络或访问 GPU：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision.sh tracking-self-test
```

## 6. 静态三帧定位也可以鼠标框选

对已拍摄的静态 RGB-D 目录，运行下面命令会先弹出第一帧供框选，再分别分割三帧。物体/相机在静态采集期间应保持不动：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
capture_dir="填写已有RGB-D采集目录"
bash real_tpkmp/vision/estimate.sh --capture "$capture_dir" \
  --object real_tpkmp/vision/objects/keel.json --mask-method box \
  --out "real_tpkmp/vision/runs/box_estimate_$(date +%Y%m%d_%H%M%S)"
```

没有传手眼结果时只输出相机位姿；正式抓取仍需同组内参对应的有效手眼结果。也支持 `--box` 归一化坐标跳过弹窗。

现有 `capture.sh` / `estimate.sh` 的静态三帧、手眼转换和抓取流程保持独立。先完成这里的实物视觉检查，再使用经过验证的手眼结果生成基座抓取目标。
