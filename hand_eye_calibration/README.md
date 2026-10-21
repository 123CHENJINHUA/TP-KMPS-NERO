# D435i + ChArUco 手眼标定

独立于原来的仿真、试教、装配和 PPO 程序。支持眼在手上、眼在手外；使用已有 `tpkmp` Python，新增依赖只安装在本目录的 `.vendor/`。真实采集只接收 CAN，不向机械臂发送运动、使能或模式切换命令。手动移动到不同姿态，停稳后按空格采样。

## 1. 已配置的板与坐标系

| 参数 | 配置 |
| --- | --- |
| 相机 | RealSense D435i 的 **RGB** 流，默认 1280×720、30 fps |
| ChArUco | Calib.io **Rows=5、Columns=7**，即 **7 列 × 5 行方格** |
| 方格边长 | 40 mm = 0.040 m |
| marker 边长 | 30 mm = 0.030 m，不含周围白色棋盘区域 |
| 棋盘区域尺寸 | 宽 280 × 高 200 mm，24 个 ChArUco 内角点 |
| ArUco 字典 | 已确认 6×6；默认使用标准 `DICT_6X6_50`，说明见第 3 节 |
| marker 编号 | 默认 0–16；若现有板不同，修改 `marker_ids` |

板参数在 [board_config.json](board_config.json)。用户已确认 Calib.io 的 Rows=5、Columns=7，因此 `squares_x=7` 对应 Columns、`squares_y=5` 对应 Rows。这里填写方格数，不是内角点数。旧 OpenCV 的 legacy 开关主要影响偶数行数；当前 5 行板开关不会改变布局。板应固定在平整、刚性的底板上，打印后用尺测量方格。

**相机内参改用新板：Rows=9、Columns=14，20 mm 方格、15 mm marker，标准 5×5 编码、编号从 0 开始。**独立配置为 [charuco_rows9_cols14_5x5_20mm.json](board_configs/charuco_rows9_cols14_5x5_20mm.json)：14 列×9 行、棋盘区域 280×180 mm、104 个内角点、63 个 marker（0–62）。使用 `DICT_5X5_100`；当前 OpenCV 的 100/250/1000 前 63 个码字相同，`5X5_50` 容量不足。新板采集必须显式传 `--board-config`，第 6 节已经使用该配置。旧默认板保留，用于之前的手眼流程。新板说明和参考打印文件见 [board_configs/README.md](board_configs/README.md)。

变换统一记为 `T_parent_child`：将 **child 系的点变换到 parent 系**，是按行保存的 4×4 齐次矩阵，平移单位米。机械臂坐标为 SDK Nero 基座系和**法兰系**，没有套用夹爪 TCP 偏移。SDK RPY 使用 `Rz(yaw) Ry(pitch) Rx(roll)`；手动输入角度单位 rad。输出四元数顺序为 **xyzw**。

相机系是 RGB 光学系：X 向右、Y 向下、Z 向前；不等于 D435i 外壳、深度相机或 Isaac Sim 的 Camera 坐标系。板坐标使用 OpenCV `getChessboardCorners()` 的实际定义，原点不在板中心。[OpenCV 板与角点说明](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html)、[RealSense 各流坐标系说明](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)。

## 2. 安装与设备检查

下面命令可整块复制。新目录依赖已安装；换机器或缺包时再运行安装脚本，不需要改现有 conda 环境。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/install_deps.sh
bash hand_eye_calibration/run.sh doctor.py
bash hand_eye_calibration/run.sh doctor.py --camera-profiles
```

`doctor.py` 显示依赖路径、相机序列号和 USB 连接类型；`--camera-profiles` 还列出当前支持的 RGB BGR8 分辨率/帧率，只枚举、不启动图像流。多个 D435i 同时连接时，采集或预览加 `--serial 实际序列号`。在图形桌面的终端采集；不要让 RealSense Viewer 同时占用相机。

**`Couldn't resolve requests` 是 RealSense 流启动失败。**脚本现先检查所选相机实际支持的分辨率/帧率和 RGB/深度组合，失败会显示序列号、USB 类型及可用模式。默认请求 RGB 1280×720、30 fps；USB2 连接可能无法提供这个模式。USB 限制说明见 [RealSense USB2 支持](https://www.realsenseai.com/news-insights/usb2-support-for-intel-realsense-technology/)。

2026-10-12 现场：眼在手上使用的相机是 **262522071128**。最初检测为 USB 2.1，仅提供 1280×720 的 6/10/15 fps；后续连接变为 USB 3.2，1280×720 的 30 fps 和 15 fps 都已实际取图通过。当前第 4 节命令明确指定该相机和 30 fps。若再次协商为 USB2，可明确选择 15 fps：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
calib_data="hand_eye_calibration/datasets/eye_in_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 15 \
  --target-samples 30 --out "$calib_data"
```

检查板的实时预览也支持同样参数：`bash hand_eye_calibration/run.sh inspect_board.py --serial 262522071128 --width 1280 --height 720 --fps 15`。需要 30 fps 时检查 USB3 接口及数据线，再用 `doctor.py --camera-profiles` 确认。内参和手眼结果必须对应本次相机；之前眼在手外相机 **262522072061** 的内参不能套用到这只相机。设备 USB 信息另存于会话 `camera_device`，采集使用实际启动的 RGB 分辨率和帧率。

**2026-10-12 已确认：眼在手上相机装在从臂，主从共用 can0 总线，电脑接在该总线上。**实际检查 can0 的 `measured`、`leader` 都有完整新鲜七轴数据；从臂相机应使用 `--joint-source measured`，实体标签使用 `nero_follower`。`--can` 选择总线接口，不能在共享总线上单凭接口名判断是哪只臂；`--arm-id` 只是记录标签，不改变读取来源。

`leader` 读取主臂角度/发给从臂的关节指令，`measured` 读取实际关节反馈。主从跟随不保证角度严格相等；2026-10-12 一组近同时快照的 FK 位置差约 3.85 mm、旋转差约 1.18°，这是当时姿态的对照，不是整组误差统计。相机或板装在从臂时，必须用该从臂实际反馈计算其基座到法兰位姿；共用总线不需要为了切换来源而更换 `can0`。

之前眼在手外挂板臂的接口重插后从 can1 改名 can0，历史标签为 nero_can1。那些会话使用 can0/leader，但仅观察关节随联动变化不能证明相机/板装在主臂本体；若当时板也在从臂，旧结果同样需要改用 measured 重新采集。历史数据保持原样供诊断。

检查当前挂板臂 can0 的两种反馈来源和离线 FK：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py --can can0 --joint-source both
```

`both` 仅用于诊断，不会替采集选择来源。报告分别列出 `measured`、`leader` 的有效样本数和最近位姿；确认板/相机安装在对应实体臂上，再在采集时明确选择。

接口重新插拔后名称或 UP 状态可能变化，须重新确认实体臂对应关系。早期 00:01、00:53 的 can0 会话选到了另一只臂；01:43 新会话是在接口改名后采集，不能仅凭目录名含 can0 就判定选错。历史报告见 [VALIDATION.md](VALIDATION.md)。从臂应读取它自己的实际反馈；主从联动时两种来源都可能随动作变化，不能据此把主臂指令当成从臂实测位姿。

**反馈完整且时间戳更新，还不足以确认它跟随这只实体臂运动。**标定前再做一次运动对应检查：另开终端运行以下命令，在 20 秒监听期间直接移动挂板的实体臂，观察哪条 CAN、哪种来源的七个 `q(deg)` 随动作变化。脚本只接收 CAN，可与采集窗口同时运行，不使用相机。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 can1 --joint-source both --seconds 20
```

报告自动保存在 `diagnostics/motion_<时间戳>/report.json`，包含每路 `transport` 原始 CAN 帧数、ID 计数和接收错误。无完整反馈时，终端同时显示原始累计帧数：0 帧表示尚未收到报文；有原始帧但没有完整反馈，才需要核对报文类型、分包、新鲜度和固件解析。若实体臂移动时 q 不变，先检查通道对应、连接与广播模式，不能用重复收到的固定目标角代替真实动作。接口 DOWN 会明确提示，不自动启用接口或切换机械臂模式。

只检查某一种来源，或检查另一条 CAN：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py --can can0 --joint-source measured
bash hand_eye_calibration/run.sh doctor.py --can can1 --joint-source measured
```

CAN 必须已经 UP。可先查看：

```bash
ip -details link show can0
ip -details link show can1
```

若接口 DOWN，使用之前 [硬件检查文档](../hardware_checks/README.md) 的配置命令。当前默认 SDK 固件配置 `v121`，可通过 `--firmware` 更换。

只有接口实际为 DOWN 时，启用当前对应的通道，然后重新监测，期间只移动挂板的实体臂。以下以当前 can0 为例：

```bash
sudo ip link set can0 up type can bitrate 1000000
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 can1 --joint-source both --seconds 20
```

哪个通道、来源的关节角和法兰位姿随动作变化，就选择对应反馈。若两路都不随动作变化，需要继续检查反馈来源或硬件广播，不能仅凭接收样本数认定通道正确。

若接口已 UP 但原始报文仍为 0，问题发生在 SDK 解析之前，单纯换 `leader/measured` 或固件解析不能解决。先确认该适配器的 CAN 端实际接到已上电的机械臂，核对 CAN_H、CAN_L、公共地和总线终端配置；如果连接、供电均已确认，继续核查该实体臂是否在广播。

不知道 USB-CAN 对应哪只实体臂时，可以暂时只保留挂板那只臂的 USB-CAN 数据连接，用 `ip -details link show type can` 确认剩余接口名，再对该接口做原始接收和运动监测。USB 重新插拔后接口名或 UP 状态可能变化，需要重新检查。无第三方依赖的原始接收命令：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
/home/cjh/anaconda3/envs/tpkmp/bin/python -B hardware_checks/probe_can.py --seconds 5
```

该工具会检查 can0/can1；只连接一路时，另一路报不存在或 DOWN 是预期现象。先让挂板那只臂所对应通道收到报文，并确认拖动关节时角度变化，再重启采集。

**选择正确的实体机械臂位姿来源：**默认 `measured` 读取 `get_joint_angles()`，缺帧或过期就拒绝，不会自动改读 leader。若你采集的是**装有板/相机的试教主臂本体**，且该臂只广播主臂角度，则先用以下命令确认，再在采集命令中明确加入 `--joint-source leader`：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh doctor.py --can can0 --joint-source leader
bash hand_eye_calibration/run.sh doctor.py --can can1 --joint-source leader
```

不能把主臂角度当成另一台跟随臂的实际位姿。`can0/can1` 是通道名；用 `--arm-id` 标记实体臂身份，移动或交换线缆后仍需确认对应关系。采集启动会先等待所选来源的完整新鲜反馈，通过后再打开相机和创建数据目录；若只收到另一种来源，报错会说明，但不会自动切换。

如果已经开着使用错误通道或来源的采集窗口，先按 **Q/ESC** 退出；确认正确实体臂的通信和来源后，用新的时间戳目录重新启动。不要在已有会话中更改位姿来源或改写其元数据。早期 00:01、00:53 两个 can0 会话保留供诊断，不能用于当前挂板实体臂的标定。

## 3. 确认现有板的字典，或打印程序生成的板

用 D435i 实时预览，板完整入镜后按空格检查：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh inspect_board.py
```

也可检查照片：`bash hand_eye_calibration/run.sh inspect_board.py --image /实际照片路径.png`。

检查报告列出识别数量和与当前板布局匹配的角点数。某些字典共用同一批码字，可能多个候选都识别 17 个 marker、24 个角点；这时查原板生成配置，脚本不会伪装成已经唯一识别。识别不到全部角点也可能是遮挡、字典、编号、板大小或布局不匹配。

**17 个 marker、0 个 ChArUco 角点**说明编号识别成功，但不能证明棋盘布局正确。Calib.io 的 Rows/Columns 顺序与 OpenCV 的 `(squares_x, squares_y)` 顺序不同；这次零角点已确认为之前把行列设反。修正后在 D435i 实拍帧上识别到 17 个 marker、24 个 ChArUco 角点。[Calib.io 生成器](https://calib.io/pages/camera-calibration-pattern-generator)、[OpenCV 检测与板定义](https://docs.opencv.org/4.11.0/df/d4a/tutorial_charuco_detection.html)。

检查脚本现在自动保存 `diagnostics/<时间戳>/raw.png`、`annotated.png`、`report.json`，比较当前行列、互换行列、旧版布局和几种编号排列。几何检查使用所有匹配 marker 的四个角点，不通过丢弃 marker 来掩盖错误；OpenCV 异常会报告，不再默默显示成 0。只有一个完整板候选通过时另存 `suggested_board_config.json`，不会自动修改主配置。建议配置基于图像布局，不能证明打印后的毫米尺寸。

你的板已确认使用 **6×6 marker**，这指每个 marker 的编码格数；整块板为 **5 行×7 列方格**。目前默认 `DICT_6X6_50`，使用编号 0–16。已对当前 OpenCV 的 `DICT_6X6_50/100/250/1000` 实际比较：这 17 个编号的码字相同，生成的板图也逐像素一致。因此标准板使用默认编号 0–16 时，不必为了确认容量后缀而停止采集。如果实际板用了其他编号，或是自定义 6×6 字典，仍必须取得对应的编号/字典配置。完整标准字典名称明确后，可修改 `board_config.json` 的 `dictionary`，或用 `--dictionary` 覆盖。

后续采集命令从 `board_config.json` 读取字典。如果要打印程序生成的 6×6 板，已经提供 [宽 280×高 200 mm 的 A4 横向 PDF](boards/charuco_rows5_cols7_6x6/board_A4_100percent.pdf) 和 [配套配置](boards/charuco_rows5_cols7_6x6/board_config.json)。打印时选择 **实际大小 100%**，关闭“适合页面”，打印后测量 40 mm 方格和 30 mm marker；不要根据 PNG 在浏览器里显示的大小判断实物尺寸。旧 `boards/charuco_5x7_6x6/` 为 5 列×7 行，和这块 Calib.io 板的编号布局不同；请使用上述新目录。

需要另生成一份：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh generate_board.py --dictionary DICT_6X6_50 \
  --out "hand_eye_calibration/boards/charuco_$(date +%Y%m%d_%H%M%S)"
```

## 4. 眼在手上：相机装在末端

**固定板，相机刚性固定在末端。**标定过程不移动板，也不改变相机支架。相机需看到完整或足够多的板角点。

以下代码块用于原 **7 列×5 行、40/30 mm、6×6 编码板**，指定相机 262522071128、1280×720、30 fps，加载 22:26 已通过的 RGB 内参，使用共享 can0 总线上的 **measured 从臂实际反馈**。实物若也换成新 9 行×14 列板，使用第 6.1 节。可以继续通过主臂控制从臂；采样时等待装相机的从臂停稳。把臂摆到一个姿态并停稳，点击预览窗口后按空格；改变位置、距离及不同方向的倾角约 15–45°，总共采集 30 张。采满自动退出，Q/ESC 可以提前结束。只平移、只绕一个轴、或姿态几乎相同都无法可靠标定。

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

结果字段 **`T_flange_color.matrix`** 是相机 RGB 光学系到机械臂法兰系的变换：

```python
# 均为齐次列向量；T_base_flange 是机械臂当前实际法兰位姿。
p_base = T_base_flange @ T_flange_color @ p_color
```

第二只末端相机请按其实体臂的实际 CAN 通道指定 `--can` 和 `--arm-id`，另建数据目录；有多个相机时同时指定对应的 `--serial`。相机支架、安装位置、焦距/流分辨率变化后重新标定。

旧会话 `datasets/eye_in_can0_20261012_202718/` 只保存了 leader 位姿，而相机在从臂，不作为从臂相机的有效标定结果。不能通过改 metadata、改 `arm_id` 或重新运行 solve 把主臂位姿变成从臂反馈；需要上述新采集。原文件保留不改。

## 5. 眼在手外：相机固定在外部

**固定相机，板刚性固定在机械臂末端。**手握着板随臂大致移动不满足刚性安装要求。法兰到板的偏移不需要手工测量，程序一起估计，但在整组采集和验证中不能改变安装。

之前挂板会话读取 can0/leader，实体标签为 nero_can1。以下历史示例仅适用于板装在主臂本体；若板装在从臂，必须改用其 `--joint-source measured`，并记录从臂身份，例如 `--arm-id nero_follower`。先用下列监测检查完整新鲜反馈；主从同时运动时，单凭 q 随动作变化不能区分实际反馈与指令：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh monitor_motion.py \
  --channels can0 --joint-source both --seconds 20
```

以下采集命令保留历史 can0/leader 和实体标签 nero_can1，限板确实装在主臂本体时使用；换线或改变模式后应重新监测。板在从臂时，将 `--joint-source leader` 改成 `--joint-source measured`，并使用从臂标签。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
calib_data="hand_eye_calibration/datasets/eye_out_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-to-hand \
  --can can0 --arm-id nero_can1 --joint-source leader \
  --target-samples 30 --out "$calib_data"
bash hand_eye_calibration/run.sh solve.py --dataset "$calib_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_data/result.json"
```

结果字段 **`T_base_color.matrix`** 是固定相机 RGB 光学系到这只机械臂基座系的变换：

```python
p_base = T_base_color @ p_color
```

固定相机需要与两只臂配合时，板分别安装到 can0、can1 的末端，各采一组，分别求解各自基座到同一个相机的关系。不要混合两只臂的数据；两个基座并不是同一个坐标系。

## 6. RGB 内参：可选，但畸变模型必须正确

默认使用 D435i 当前 RGB 流的工厂内参；全零畸变和普通 Brown 可以直接使用。非零 `modified_brown_conrady`、`inverse_brown_conrady` 不会直接冒充 OpenCV Brown；这种情况程序提示先做 RGB 内参标定。也可主动重标内参，提高与实际板、当前成像条件的一致性。

内参采集不需要 CAN，板或相机均可以移动。当前使用新板：9 行×14 列、20 mm 方格、15 mm marker、DICT_5X5_100。建议 30–40 张，包含不同距离和倾角，覆盖图像四周以及中央，停稳再按空格。可以先检查实物板；完整板应检测到 63 个 marker、104 个内角点：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh inspect_board.py \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --serial 262522071128 --width 1280 --height 720 --fps 30
```

采集并求解。每张至少要求 40 个角点，板不必每张完整入镜：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
intr_data="hand_eye_calibration/datasets/intrinsics_9x14_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode intrinsics \
  --board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --min-corners 40 --target-samples 35 --out "$intr_data"
bash hand_eye_calibration/run.sh calibrate_intrinsics.py --dataset "$intr_data"
```

确认 `passed=true` 后，手眼采集命令加入 `--intrinsics "$intr_data/intrinsics.json"`。相机序列号和分辨率必须相同。相机内参标定不会修改 D435i 固件。

用新板标定内参不要求手眼也换板：手眼镜头前若仍是旧 7 列×5 行、40/30 mm 板，保持原默认板配置；若也换成新 9 行×14 列板，手眼 collect 命令同时加入上述 `--board-config`。新板连续四格应为 80 mm，旧板为 160 mm。求解和验证读取各自会话中的板参数，不使用今天的默认配置改写旧数据。

2026-10-12 21:09 的从臂 measured 会话仍未通过：留出平移 RMS=9.567 mm、旋转 RMS=1.497°、图像闭环 RMS=9.197 px。已确认板横纵连续 4 格约 160 mm，当前静止图像／位姿波动较小；训练图片内参对照可降低部分误差，但完整链仍不通过。不能认定继续采同样条件的 30 张就能解决；应进一步验证内参、相机和机械臂的姿态相关刚性及名义 FK 的实物精度。完整证据见 [21:09 会话诊断](diagnostics/dataset_210908_20261012/README.md)。这里的 35 张内参图片应覆盖图像四周和中央、不同距离、不同方向的明显倾角，不能只有画面内转动。无需 CAN，相机和板的相对位置可以变化，但板自身必须保持平整。

如果以后使用深度点，在手眼采集时加入 `--with-depth`：`metadata.json` 的 `camera_raw.T_color_depth` 保存 SDK 的深度到 RGB 外参。先 `p_color = T_color_depth @ p_depth`，再使用相应手眼变换。RGB/深度的对齐图与原始深度光学系的点不能混淆。

### 6.1 当前内参已通过：使用新板做从臂眼在手上标定

`datasets/intrinsics_9x14_20261012_222636/intrinsics.json` 已通过：35 张实际图片，28 训练／7 留出，训练 RMS≈0.150 px、留出≈0.139 px，每张 90–104 个角点。相机为 262522071128、1280×720@30 fps，fx≈899.940、fy≈900.556；结果有非零 OpenCV Brown 畸变参数。

已用此内参重算原 21:09 的 30 张 measured 手眼照片，保存为 `datasets/eye_in_can0_20261012_210908_intr_222636`。相同 24/6 划分下，留出平移从 9.567 降到 7.502 mm、旋转从 1.497 降到 1.049°，图像闭环 9.065 px，仍未通过。原始图片、CAN 位姿、采集时间和角点编号集合均保留；这不是新的独立采集。详见 [内参与重算对照](diagnostics/intrinsics_222636_20261012/README.md)。

新的实物采集同时使用新 9 行×14 列板和这组内参。固定板，相机固定在从臂支架上；可由主臂操控，记录从臂 measured，等待从臂停稳后按空格。终端会显示实际加载的内参文件和 fx/fy。

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

通过后保持安装不变，独立采集 10 个新姿态、不重新求解的复制命令见 [COMMANDS.md 9.8](../COMMANDS.md#hand-eye-new-intrinsics)。该验证必须使用同样的新板配置与内参；第 7 节中的旧板示例仅对应第 4 节的采集。

新增 `reprocess.py --dataset 原手眼目录 --intrinsics 内参文件 --out 新目录`，只读取已保存的照片和位姿，自动使用拍摄时保存的板配置，重新检测和求解板位姿。相机／分辨率／帧率／光学系必须匹配，内参必须 passed=true；原样本与角点编号集合保留，任何一张不通过就取消整次输出。派生数据不是独立验证，也不能修复记录了错误实体机械臂位姿的旧 leader 数据。完整命令同见 9.8。

## 7. 验证与结果判断

`solve.py` 默认 OpenCV **PARK** 初始化，再用同时优化外参和固定板偏置的稳健非线性细化。至少 12 个训练位姿和 5 个留出位姿；默认留出 20%，不足 5 张时留出 5 张。留出数据**没有参与**外参和板偏置拟合。检查旋转轴激励，纯平移、单轴旋转或转角太小会拒绝求解。[OpenCV 手眼接口](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)。

程序的两种约束分别为：

```text
P = T_base_flange，Q = T_color_board（ChArUco PnP）
眼在手上：P_i X Q_i = Y，X = T_flange_color，Y = T_base_board（固定）
眼在手外：P_i C = X Q_i，X = T_base_color，C = T_flange_board（固定）
```

眼在手外会先反转**每个输入机械臂位姿**，再送入 OpenCV；不是把眼在手上求出的结果简单反转。

输出包括：外参及其逆矩阵、法兰/板偏置、训练与留出 ID、平移闭环误差（mm）、旋转闭环误差（deg）、图像闭环重投影误差（px）、每张 PnP 误差。默认通过条件是留出平移 RMS ≤ 5 mm、旋转 RMS ≤ 2°、闭环重投影 RMS ≤ 5 px，且单张平移/旋转不超过各 RMS 阈值的 3 倍。失败仍保存报告，退出码为 2。

这些是开始检查的阈值，**不代表满足螺栓插孔的精度要求**。真实误差还取决于机械臂 FK、板尺寸/平整度、支架刚度、角点检测和数据配对。按任务公差收紧 `--max-translation-mm`、`--max-rotation-deg`、`--max-loop-reprojection-px`，并用实物位置验证。没有自动删除误差大的样本来美化结果。

本机 01:43 会话的留出平移 RMS=3.228 mm、旋转 RMS=0.778°，均通过；图像闭环重投影 RMS=6.794 px 超过 5 px，故总验证失败。单张 PnP RMS 为 0.175–0.616 px，训练姿态激励通过。详见[数据诊断](diagnostics/dataset_014300_20261011/README.md)及[误差图](diagnostics/dataset_014300_20261011/quality.png)。板的纸板底座和单侧夹持可能引入弯曲或滑移，属于待验证因素。先保证平整刚性底板及固定连接，实测方格为 40 mm、连续四格为 160 mm，每次停稳后等待约 2 秒再按空格；改变安装后另建会话重新采集。板和相机安装原则可参考 [Calib.io 标定实践](https://calib.io/blogs/knowledge-base/calibration-best-practices)。若这些已确认仍不通过，再检查 RGB 内参与机械臂 FK。

采样在按空格后读取新帧，使用图像前后的完整 CAN 关节反馈，约 0.5 秒前置静止窗口及 0.2 秒后置检查。保存设备时间戳和主机接收时间，但**没有实现硬件触发同步或估计不同设备时钟的偏移**。因此采样必须静止，不能将此脚本用于高速连续轨迹下的同步标定。

采集窗口现会显示七个 CAN 关节角（deg）、法兰 XYZ（m）、与上次保存的法兰位置/转角差、视觉板的转角差。按空格被拒绝时，会显示具体参考样本、位移、转角和关节角变化，而不是只有“太接近”。

重复判断比较的是**法兰末端位姿**，门槛为至少 5 mm 位移或 3°转角，并与全部已保存样本比较。七轴机械臂可能在中间关节变化时保持末端位姿相近；这种动作不增加手眼标定需要的位姿激励，不能仅因为关节角变化大就保存。

两种手眼模式中，机器人与视觉板的相对旋转角应一致，与未知的相机/板安装偏移无关；平移幅度可以不同。程序默认将相对转角差超过 8°的样本拒绝，提示核对实体臂对应关系和固定安装。重复或旋转不一致的失败采样会保存到该会话的 `rejections/`（图像、机器人位姿、视觉位姿和差值），不会放进 `samples/` 参与求解。阈值通过 `--max-motion-angle-error-deg` 配置；不能用放宽重复位姿条件来掩盖不随动作更新的 CAN 反馈。

更可靠的检查是另采 10 个新姿态，固定相机/板安装不变，不重新求解外参或偏置：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
# 在完成第 4 节的新 measured 标定的同一终端运行，calib_data 指向该次新会话。
intr_file="hand_eye_calibration/datasets/intrinsics_9x14_20261012_222636/intrinsics.json"
calib_result="$calib_data/result.json"
check_data="hand_eye_calibration/datasets/check_in_can0_$(date +%Y%m%d_%H%M%S)"
bash hand_eye_calibration/run.sh collect.py --mode eye-in-hand \
  --board-config hand_eye_calibration/board_config.json --intrinsics "$intr_file" \
  --can can0 --arm-id nero_follower --joint-source measured \
  --serial 262522071128 --width 1280 --height 720 --fps 30 \
  --target-samples 10 --out "$check_data"
bash hand_eye_calibration/run.sh validate.py --result "$calib_result" \
  --dataset "$check_data" --out "$check_data/validation.json"
```

如果训练时使用自定义内参、相机序列号、分辨率/帧率、leader 来源或其他字典，此独立采集必须使用同样配置；例如原采集为 `--fps 15`，独立验证也指定 15。验证不重新估计固定板偏置，否则会掩盖错误。

当前眼在手外 `datasets/eye_out_can0_20261011_165821/result.json` 的独立验证命令，见[指令说明 9.4](../COMMANDS.md#94-当前眼在手外结果的独立验证)。该结果留出平移 RMS=3.243 mm、旋转 RMS=0.461°，均通过；重投影 RMS=5.076 px，比 5 px 高约 1.52%，严格判定仍为 False。训练样本 0009 的平移闭环误差约 13.5 mm，需要在其附近姿态额外核查；这些闭环指标不等于实物绝对定位误差。

10 个新姿态的独立验证已完成，用户确认安装保持不变：平移 RMS=4.338 mm、旋转 RMS=0.918°、重投影 RMS=6.171 px，严格判定仍为 False，未重新拟合原外参或板偏置。详见[独立验证诊断](diagnostics/independent_171048_20261011/README.md)。先核实实际 40 mm 方格、平整刚性底板和末端固定，再检查 RGB 内参与 FK；插孔可用性需要实物定位偏差和配合容差验证。

## 8. 手动输入位姿与数据格式

如果暂时不能读取 CAN，可在 `collect.py` 使用 `--pose-source manual`。每次在图像窗口按空格后，终端输入 `x y z roll pitch yaw`（m、rad）。这是机械臂**实际法兰相对于其基座**的位姿，不是目标指令、关节角或仿真跟随臂位姿。输入时保持机械臂/板静止，脚本收到输入后才取新图像。

也可加 `--pose-file hand_eye_calibration/你的位姿.json`，每次采样重新读取它；格式参考 [examples/pose_base_flange.json](examples/pose_base_flange.json)。示例矩阵是假的，必须替换。manual 来源没有自动静止/新鲜度检查，结果会记录这一事实。

```text
datasets/<会话>/
  metadata.json        模式、板、RGB 内参、相机序列号、臂身份及来源
  images/0000.png      原始 RGB 图像
  samples/0000.json    与图像配对的 T_base_flange、T_color_board、角点及时间戳
  result.json          求解结果与留出验证
```

所有输出限制在本目录内，拒绝覆盖已有会话/报告。同一数据需要比较求解方法时用 `--method TSAI --out <数据目录>/result_tsai.json`，其他方法见 `--help`；同一个留出划分默认固定 seed=42。

## 9. 程序自检与已有验证记录

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash hand_eye_calibration/run.sh self_test.py
```

40 项测试覆盖两种模式已知外参的恢复、带噪声角点 PnP、错误模式和退化动作、板图识别、CAN 分帧完整性与过期、静止检查、SDK FK 单位/旋转、RealSense 畸变模型及 SDK 外参存储顺序；布局测试覆盖行列互换、按列编号、偶数行 legacy 和错误编号排列的拒绝；来源测试确认诊断不自动切换采集、过期的备用来源不被建议。新增测试验证实际 SDK 的 leader 报文内容变化会更新关节角/FK、机器人与视觉转角的一致性、冗余关节动作不能绕过重复判断、真实位移可正常采样，以及原始接收计数不向机械臂发送报文。相机测试覆盖 USB2 不支持默认帧率的提前拒绝、只采用所选设备的模式、可选深度组合、配置解析失败和 SDK 启动错误说明。新板测试覆盖 63 个 marker／104 个角点的生成与检测、5×5 字典前 63 个码字兼容性、容量不足拒绝、采集角点阈值随所选板变化，以及用 104 个角点完成内参求解并恢复已知合成相机。重算测试覆盖相机匹配、保留原图／机器人位姿及原数据、角点集合变化时取消整次输出。已运行的合成数据位于 `results/synthetic_check_01/`，全部标为 `synthetic=true`，**不可用于实际相机或机械臂**。详细记录见 [VALIDATION.md](VALIDATION.md)。

脚本入口：`doctor.py`、`monitor_motion.py`、`inspect_board.py`、`generate_board.py`、`collect.py`、`calibrate_intrinsics.py`、`reprocess.py`、`solve.py`、`validate.py`、`self_test.py`；均可 `bash hand_eye_calibration/run.sh <脚本名> --help`。
