# D435i + CAD 物体位姿 → 真机 TP-KMP 抓取目标

所有视觉代码、物体模型、权重、新环境、编译和运行输出都放在 `real_tpkmp/vision/`。视觉入口不发机械臂运动指令。支持 **眼在手上**和**眼在手外**，从手眼标定结果的 `mode` 判断；眼在手上使用拍照时配对的从臂 measured 位姿。

默认 `capture.sh` / `estimate.sh` 实现**物体停稳后的三帧独立 6D 配准**。新位置先拍照、定位、生成参考系，再经过现有 TP-KMP 预览/IK 检查后执行。不是运动中追踪抓取，也没有绕障、抓取候选搜索或插入力控制。

想先单独看定位和连续跟踪，使用 [相机独立跟踪测试](TRACKING.md)：`track.sh` 默认鼠标框选、用 SAM3 单实例分割，显示模型轮廓/坐标轴、处理延迟、静止抖动和尺测平移比较。它不读 CAN、不依赖手眼标定，所有结果仅诊断，不能直接交给机械臂执行。

## 方法选择与参考

| 方法 | 参考结论与本次使用方式 |
| --- | --- |
| [FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus) | 主要在 FoundationPose 上增加 2D 跟踪、滤波等组件，针对高动态场景。本次参考其 CAD/RGB-D 接口，不启用 Cutie/Qwen/Kalman 跟踪链路。 |
| [官方 FoundationPose](https://github.com/NVlabs/FoundationPose) | 本次默认后端。已知 CAD、RGB、深度、内参和目标 mask，使用预训练模型估计物体相对相机的完整位姿；不对 keel 重新训练。 |
| [FoundationPose--](https://github.com/ziqin-h/FoundationPose--) | 较新的配准加速工程。作者报告的速度提升需要本机复测；当前不启用其减枝或 RGB-only 分支。 |
| [SAM3](https://github.com/facebookresearch/sam3) | 使用文本提示自动分割目标，作为配准的 mask 输入。源码和已有权重已放入本目录。 |

`--foundationpose-root` 支持官方目录，也能指向 `FoundationPose++/FoundationPose` 或 `FoundationPose--/FoundationPose`。这表示调用其中相同的配准 API，**不等于启用了这些扩展项目的全部算法**。

## 1. 环境与模型

先检查：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/doctor.sh
```

默认父进程仍用 `tpkmp`，复用已有 OpenCV/RealSense vendor；GPU 分割和配准各用一个独立 worker 环境，避免与 TP-KMP 的 NumPy、OpenCV 冲突。

从头安装/修复的入口：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/setup.sh sources
bash real_tpkmp/vision/setup.sh sam3-env
bash real_tpkmp/vision/setup.sh pose-env
bash real_tpkmp/vision/setup.sh weights
bash real_tpkmp/vision/doctor.sh
```

本机安装脚本在 `vision/envs/` 创建 venv，默认**只读复用**已有 `/home/cjh/anaconda3/envs/qwen-sam3` 中的 CUDA PyTorch；补充依赖安装到新 venv，不改原环境。CUDA 编译工具链通过 conda 安装在 `vision/envs/cuda_toolchain/`，conda/pip 缓存也在视觉目录，关闭环境全局注册。默认工具链 12.8 与本机 PyTorch `2.9.1+cu128` 匹配。

其他机器可用 `VISION_BASE_PYTHON` 指定已有 CUDA PyTorch+torchvision 环境，`VISION_CUDA_VERSION` 指定匹配版本，或提供已有 `CUDA_HOME`。`SAM3_PYTHON` / `TPKMP_POSE_PYTHON` 可分别覆盖运行解释器，CLI 也提供 `--sam3-python` / `--pose-python`。本机主机 GPU 为 RTX 4070 Ti SUPER 16GB；受限沙箱可能看不到 GPU，实际运行应在正常主机终端。

默认 FoundationPose 源码位于 `third_party/FoundationPose/`。其官方权重需包含：

```text
third_party/FoundationPose/weights/
  2023-10-28-18-33-37/config.yml
  2023-10-28-18-33-37/model_best.pth
  2024-01-11-20-02-45/config.yml
  2024-01-11-20-02-45/model_best.pth
```

下载来自 [官方发布目录](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)。若 Google Drive 包含 `no_diffusion/` 外层，安装脚本会在同目录中建立对应链接。SAM3 权重位于 `weights/sam3.pt`；当前使用工作区已有权重的副本，无需重新申请或下载。

本次已找到并复制 `/home/cjh/FoundationPose/weights/` 中的两组官方权重，以及本机兼容的 Open3D 等大型依赖到新 worker 环境，避免重复下载。原目录与已有 conda 环境未改动。权重齐全时无需再运行 `setup.sh weights`。

## 2. 物体与相机配置

当前物体配置：[objects/keel.json](objects/keel.json)。STL 已复制到 `models/keel/keel.STL`，使用用户确认的 **150×98×23 mm**，内部转换为米。载入时检查模型 SHA256 和尺寸，防止 1000 倍缩放错误。

添加其他物体时，新建对应 JSON 和 STL，填写真实单位、三轴尺寸、SHA256、分割提示及质量阈值，运行时用 `--object` 选择。当前独立几何检查读取 STL；其他网格格式先转换为保留原坐标与尺度的 STL。

没有 CAD 时也可以用照片生成近似模型。新增 [Meshy 使用说明](meshy/README.md) 提供单图/1–4 张多视图 API、网页模型离线导入、实测尺寸均匀缩放和 `object.json` 生成。入口为 `bash real_tpkmp/vision/meshy.sh --help`。生成模型另存为新物体，重新绑定抓取关系；孔径、薄壁等细节仍需实测确认。

物体坐标系始终是**原始 STL 坐标系缩放为米**，不会换成包围盒中心。FoundationPose 内部居中后的位姿会还原为这个原始坐标系，几何检查和抓取关系使用同一坐标系。

默认 `text_prompt` 是蓝色金属支架示例。如果实际 keel 是银色或其他颜色，用 `--prompt 'metal bracket'` 等提示修改。`--mask-method hsv` 仅用于实际为蓝色、场景中可区分的物体，不是通用检测器。

统一编辑 [camera_config.json](camera_config.json)：相机序号、RGB 分辨率/帧率、内参 K/畸变、完整手眼参数、物体名称和抓取绑定参数都直接放在该文件中，不需要填写参数文件路径。`capture.sh`、实时 `track.sh`、`estimate.sh`、`vision.sh frames` 及默认的 `collect.sh` 启动时读取它；换一套配置可加 `--camera-config 路径.json`。各命令显式提供的参数优先于 JSON。

serial/width/height/fps 只在顶层填写一次；`intrinsics` 只填 3×3 `K`、`distortion_model` 和 `distortion_coefficients`，COLOR 光学系约定由程序补齐。当前已从该相机的实际运行记录填入出厂值。单独拍照/跟踪时，把整个 `intrinsics` 设为 null 或用 `--factory-intrinsics`，可在启动时读取出厂参数。旧版 `--intrinsics 路径/intrinsics.json` 仍可临时覆盖。回放使用录制时的相机几何。

转换为机械臂基座目标时，需要与拍摄**同组内参**求解的手眼结果。`hand_eye` 只填 `mode`、4×4 `matrix` 和实际验证结果 `passed`。眼在手上 matrix 是 COLOR 光学系到法兰，眼在手外是 COLOR 光学系到基座，平移单位米。相机、内参和机器人信息由共享配置自动生成；程序内部保存标定关联记录，设备/内参变化后不能继续复用原已验证矩阵。仍兼容旧版完整标定 JSON 对象，无需在简化配置中维护 signature 或 transform_name。

## 3. 初次：拍摄参考物体位姿

先在 `camera_config.json` 中填入当前安装对应的有效手眼参数。下面的定位命令自动读取，不再需要指定手眼路径。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
vision_id="keel_$(date +%Y%m%d_%H%M%S)"
vision_run="real_tpkmp/vision/runs/$vision_id"

bash real_tpkmp/vision/capture.sh --can can0 \
  --out "$vision_run/reference_capture"

bash real_tpkmp/vision/estimate.sh --capture "$vision_run/reference_capture" \
  --mask-method sam3 --prompt 'metal bracket' \
  --out "$vision_run/reference_pose"
```

拍照时保持物体、相机和从臂停稳，避免夹爪挡住目标。当前配置为相机 `213522070575`、RGB `1280×720@30`、原生深度 `640×480@30`；实际参数以 `camera_config.json` 为准。多相机时修改 JSON 的 `serial`，也可用 `--serial` 临时指定。

程序使用 RealSense 原生深度点云和 depth→color 外参，把深度投影到**去畸变后的彩色相机像素**。不是直接 resize 深度。RGB 和深度一起使用同一个零畸变 K，深度为 COLOR 光轴 Z、单位米。拍照前后检查从臂关节稳定性，并把每帧法兰位姿写入数据。

只检查图像、不读 CAN 时可给 `capture.sh` 加 `--no-robot`；这种数据能用于眼在手外。眼在手上会拒绝缺少拍照时实际法兰位姿的数据。

输出包括：

- `reference_capture/rgb.png`、`depth_mm.png`、`rgbd.npz` 和相机/机器人元数据。
- `reference_pose/segmentation.png`、`mask.png`：分割结果。
- `reference_pose/overlay.png`：CAD 包围盒及原始 CAD 坐标轴叠加，红 X、绿 Y、蓝 Z。
- `reference_pose/observation.json`：`T_color_object`、`T_base_object`、逐帧质检和能否用于真机。

若有多个符合提示的候选，默认拒绝，避免抓错对象；可缩小场景目标范围或显式指定 `--instance 0` 等编号。编号按分割候选排序，不保证跨场景身份不变。

如果实际物体是蓝色，可以先用不需要 SAM3 的自动颜色分割验证 RGB-D 配准和 FoundationPose：

```bash
bash real_tpkmp/vision/estimate.sh --capture "$vision_run/reference_capture" \
  --mask-method hsv --out "$vision_run/reference_pose_hsv"
```

也支持 `--mask-method file --mask-file ...`，但 mask 必须对应保存的**去畸变图像**，不是原始畸变图。

文字分割不稳定时，给 `estimate.sh` 加 `--mask-method box`，在冻结的第一帧上框选完整物体，Enter 确认。框尽量贴近物体外轮廓，减少背景；SAM3 生成实际掩膜，保留同样的深度和几何质检。无窗口可给 `--box x0 y0 x1 y1`（归一化 0–1）。静态采集的三帧使用同一选框，物体和相机应保持静止。

## 4. 只需示教一次：物体到抓取 TCP 的关系

视觉得到物体位姿后，还要知道夹爪应该在物体的哪个位置、以什么朝向夹取。本程序用一次实际示教记录该关系，不猜测 STL 中心就是抓取点。

保持物体与 `reference_pose` 拍摄时的位置不变，通过主臂把从臂夹爪移到实际抓取位姿、停稳，在下面程序中按空格记录；记录前不要把物体抓走：

```bash
bash real_tpkmp/vision.sh bind-grasp \
  --observation "$vision_run/reference_pose/observation.json" --can can0 \
  --out "$vision_run/keel_grasp.json" --update-config
```

这一步只读取 measured。`--update-config` 在通过检查后把抓取矩阵及绑定信息直接写入 `camera_config.json` 的 `grasp` 对象，保留其余参数，无需填写抓取文件路径；仍保留输出文件供追溯。手眼或物体不匹配时不会更新。也可以给 `bind-grasp` 加 `--demo real_tpkmp/datasets/.../demo_001.npz`，使用**同一次物体摆放**下已示教的 pick 参考 TCP，不能把无关摆放的旧示教直接绑定。

公式：

```text
T_object_tcp = inverse(T_base_object_reference) @ T_base_tcp_grasp
```

抓取关系保存为 `keel_grasp.json`。在夹爪/TCP、CAD 原点或手眼标定改变后需要重新绑定；更换物体也需要自己的 CAD 与抓取关系。

## 5. 物体换位置：自动生成抓取参考系并预览

训练仍用已有真机 `collect.sh` / `train.sh`。假设模型与示教目录如下，替换成你已有的目录名：

```bash
dataset="real_tpkmp/datasets/pick_place_01"
model_dir="real_tpkmp/models/pick_place_01"
new_run="real_tpkmp/vision/runs/keel_new_$(date +%Y%m%d_%H%M%S)"

bash real_tpkmp/vision/capture.sh --can can0 \
  --out "$new_run/capture"
bash real_tpkmp/vision/estimate.sh --capture "$new_run/capture" \
  --prompt 'metal bracket' --out "$new_run/pose"
bash real_tpkmp/vision.sh frames \
  --observation "$new_run/pose/observation.json" \
  --template-frames "$dataset/demo_001_frames.json" --out "$new_run/frames.json"
bash real_tpkmp/test_real.sh --model "$model_dir/model.npz" \
  --frames "$new_run/frames.json" --out "$new_run/preview"
```

新的抓取参考为 `T_base_object_new @ T_object_tcp`；起点和放置 TCP 沿用模板。当前自动更新的是抓取目标。若要视觉识别放置工装，需要为工装另配 CAD 和目标关系。

确认定位叠加和轨迹预览后，按已有真机流程把从臂移到计划起点，停止主臂发令、保持从臂已使能，再执行：

```bash
bash real_tpkmp/test_real.sh --plan "$new_run/preview/plan.npz" --execute --can can0 \
  --out "$new_run/execution"
```

单次观察默认 300 秒有效；物体一旦再次移动，应重新拍照定位，不能继续用旧计划。视觉时保留主从跟随；只有运动执行阶段才需要释放主臂的 CAN 控制权。

推荐使用 [集成视觉示教](../README.md#视觉定位后示教推荐)：`collect.sh` 默认读取 [camera_config.json](camera_config.json)，在同一采集终端按 V 拍照定位并预览抓取 TCP，按 L 记录一次固定放置点，回统一起点按 R 开始六阶段示教。每次 R 消耗本轮观察，保存/丢弃后重新 V；放置点跨轮复用。已有 `bind-grasp` 结果可直接使用，不依赖已有 TP-KMP 示教模板。人工 P/L 模式需显式加 `--manual`。

原有文件接口仍可使用：给 `collect.sh` 加 `--task-frames "$new_run/frames.json"`，R 时从该文件读取 pick/place，起点仍记录当前实际 TCP。这个文件只适用于该次物体摆放；换位置需要重新定位生成文件。`--task-frames`、`--manual`、`--camera-config`（或 `--vision-config`）互斥。

## 6. 检查、限制与诊断

相机中的位姿不等于已验证的真机抓取目标。程序独立检查：目标有效深度、CAD 轮廓与分割 IoU、渲染深度与观测深度误差、三帧位姿离散程度，以及手眼/工具/物体来源和观察时间。

默认几何门限在 `objects/keel.json`：深度误差中位数 ≤6 mm、90 分位 ≤15 mm，三帧最大偏离 ≤5 mm/5°。这些是初始过滤阈值，**不是所保证的定位精度**；对小间隙插孔还需实测及接触控制。

未通过手眼验证、使用 leader 标定从臂、内参不一致、深度不足、定位跳变、过期观察、诊断导入位姿或合成数据都会阻止真机执行。定位失败不会静默换成 RGB 中心点或固定高度。已修改的来源文件也要求重新生成计划。

对称物体可能存在相差 180°但几何同样匹配的姿态。三帧检查不能证明所有对称歧义都已消除；应结合实际 grasp 对称性或可辨识标记验证。TP-KMP 仍有示教范围、IK 和限速约束，也仍未包含完整整臂/场景碰撞规划。

只需要相机坐标输出时，使用 `estimate --camera-only` 忽略配置中的手眼参数；结果不能用于基座目标。外参转换为：

```text
眼在手外：T_base_object = T_base_color @ T_color_object
眼在手上：T_base_object = T_base_flange_at_capture @ T_flange_color @ T_color_object
```

离线检查：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision.sh self-test
smoke="real_tpkmp/vision/runs/offline_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision.sh smoke --out "$smoke"
```

`smoke` 用实际 keel STL 渲染合成 RGB-D，使用明确标记的已知位姿验证“分割/深度→手眼转换→抓取关系→参考系→TP-KMP→IK预览→拒绝真机执行”的接口。它不加载位姿网络，不能当作网络或真机精度评测。`--diagnostic-poses` 是外部位姿/离线夹具的导入入口，始终强制诊断标记。

本机已完成实际 GPU 网络联调，记录见 [CHECKS.md](CHECKS.md)。要自行重跑“参考位置→物体移动→SAM3+FoundationPose→抓取目标→TP-KMP预览”，可以执行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision.sh gpu-check \
  --out "real_tpkmp/vision/runs/gpu_$(date +%Y%m%d_%H%M%S)"
```

此测试使用实际神经网络，不向网络提供已知位姿；输入仍是合成图像，物体在相机 X 方向移动 8 mm，使用同一手眼变换，最终只生成不可执行的合成计划。每次摆放的三帧为重复渲染，不是传感器噪声或真实相机精度测试。已验证软件链路，尚未用真机相机和实物完成抓取验收。
