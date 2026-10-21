# 真机单臂 TP-KMP：抓取与搬运

适用当前硬件：拖动主臂示教，**一只从臂与主臂共享 can0**，AGX 夹爪。采集读取从臂实际 measured 七轴反馈，不用主臂 leader 指令代替。所有程序、数据、模型、图表都位于 `real_tpkmp/`，不改变 Isaac Sim 控制程序。

第一版任务是“接近 → 夹取 → 抬升 → 搬运 → 松开 → 撤离”。TP-KMP 学习相对于起点、抓取点和放置点的 TCP 位姿轨迹；夹爪原生开合目标与力目标另外学习和同步。运动入口不包含螺栓插入、自动绕障或强化学习。

现已增加独立视觉链路：D435i RGB-D、SAM3 自动分割、FoundationPose CAD 位姿、两种手眼转换及抓取参考生成。入口为 `vision.sh`，见 [视觉使用说明](vision/README.md)。下面原有流程仍可人工记录 P/L；视觉不会自动启动真机运动。

下载或生成新物体的 3D 模型后，完整示例见 [模型导入 → 真机采集 → 训练 → 视觉泛化测试](vision/meshy/WORKFLOW.md)。

## 入口与环境

| 文件 | 用途 | 是否发运动指令 |
| --- | --- | --- |
| `collect.sh` | 拖动主臂采集从臂成功示教 | 否 |
| `train.sh` | 训练、整条留出验证、输出误差图 | 否 |
| `validate.sh` | 用新示教独立验证冻结模型 | 否 |
| `run.sh frames.py` | 记录或偏移新的任务参考 TCP | 否 |
| `test_real.sh --model ... --frames ...` | 离线预测、IK、限速与预览 | 否，不连接 CAN |
| `test_real.sh --plan ... --execute` | 执行已生成计划并记录实际跟踪 | **是** |

默认用 `/home/cjh/anaconda3/envs/tpkmp/bin/python`，无需重新安装环境。复用本项目 pyAgxArm、TP-KMP 算法和 `hand_eye_calibration/.vendor` 的 python-can。换环境时可在命令前设置 `TPKMP_PYTHON=/绝对路径/bin/python`。使用下面的 `.sh` 入口，它会配置模块路径与缓存目录。

## 先检查配置

编辑 [config.json](config.json)，再开始采集：

- `channel` 默认 `can0`；USB-CAN 插拔可能导致通道重命名，采集和执行也可显式加 `--can can1`。通道内只能有一组使用这些反馈 ID 的从臂。
- `arm_id` 是实体臂的身份标签，不会改变 CAN 读取来源。换实体臂必须区分标签和数据。
- `T_flange_tcp_pose_wxyz` 是**法兰到工具参考 TCP**的安装位姿 `[x,y,z,qw,qx,qy,qz]`，平移单位米。默认 `[0,0,0,1,0,0,0]`，即工具参考点暂时与法兰重合。若使用夹爪中心作为 TCP，先实测该变换并填写；修改后重新采集和训练，不要猜一个工具长度。
- `workspace_min_m` / `workspace_max_m` 是**从臂基座坐标系**内的 TCP 范围。按当前桌面和安装位置调整。默认范围只是初始配置。
- `keepout_aabbs_m` 可设置 TCP 禁入盒，例如 `[[[0.2,-0.1,0.0],[0.3,0.1,0.2]]]`。
- 关节限位来自本地 Nero SDK；速度、加速度、跟踪误差和夹爪范围/力上限在这里配置。夹爪 width 使用米，angle 使用度，自动保留采集时的模式。

**本程序只检查关节限位、TCP 范围和 TCP 禁入盒，没有整个机械臂、夹爪、桌面、其他机械臂的碰撞模型。**先在空旷工作区用小物体短距离验证。留出误差通过只能说明轨迹拟合达到配置阈值，不能证明抓取成功或无碰撞。

## 1. 采集真机示教

相机引导的采集使用下方的 [V 定位示教流程](#视觉定位后示教推荐)。原有人工 P/L 操作继续可用。

### 视觉定位后示教（推荐）

首次为当前物体完成一次 [抓取关系绑定](vision/README.md#4-只需示教一次物体到抓取-tcp-的关系)，保存 `keel_grasp.json`。相机、内参和安装位置需要与已通过验证的真实手眼结果一致；首次绑定不需要先采集 TP-KMP 数据集。之后相机每轮确定物体位姿，通过完整的物体到 TCP 变换计算本轮抓取目标。

直接编辑已经创建的 [camera_config.json](vision/camera_config.json)。示教采集、拍照、定位、跟踪和视觉任务参考系入口启动时默认读取这一份文件，无需每次传配置路径。

| 字段 | 填写内容 |
| --- | --- |
| `serial` | 当前相机序号，已填 `213522070575` |
| `width`、`height`、`fps` | RGB 分辨率和帧率，已填 `1280`、`720`、`30` |
| `intrinsics.K` | 3×3 内参矩阵 `[[fx,0,cx],[0,fy,cy],[0,0,1]]`，已填写该相机运行记录中的出厂值 |
| `intrinsics.distortion_model` / `distortion_coefficients` | 畸变模型及系数；当前为 `none` 和五个 0，Brown 模型使用 `opencv_brown` |
| `hand_eye.mode` | `eye-in-hand` 表示眼在手上，`eye-to-hand` 表示眼在手外 |
| `hand_eye.matrix` | 4×4 手眼矩阵，平移单位米；未知项暂为 null |
| `hand_eye.passed` | 实际标定验证结果；目前为 false，完成验证后填写真实结果 |
| `object_id` | 物体名称，当前为 `keel`；从 `objects/<名称>.json` 或 `models/<名称>/object.json` 选择，无需填路径 |
| `grasp` | 完整抓取绑定参数对象，包含 `T_object_tcp` 矩阵；首次绑定后可用 `--update-config` 自动填入 |
| `frames`、`mask_method`、`iterations` | 静态采集帧数、分割方式和位姿配准迭代数 |

参数直接写在这一份 JSON 中，每项只填一次。serial/width/height/fps 只在顶层填写，`intrinsics` 只填 K 和畸变。内参数字来自 `track_20261021_164401/session.json` 的 `source_camera`，对应 `213522070575 / 1280×720@30` 的出厂内参，没有冒充新的内参标定结果。更换相机或分辨率时，需要填写对应参数。

`hand_eye` 只填 `mode`、`matrix` 和 `passed`。眼在手上的 matrix 把 COLOR 光学系坐标转换到法兰系；眼在手外则转换到基座系。矩阵为 `[R,t;0,0,0,1]`，t 单位米，切换安装方式只改 mode 和对应矩阵。相机信息从顶层和 intrinsics 读取；机器人身份在示教时从当前机器人配置读取，独立定位时从配对的拍照记录读取。格式版本、变换名称和校验信息由程序补齐，不需要重复填写。

通过验证的手眼矩阵与相机/内参、机器人之间的关联会自动保存到内部记录。以后改变设备或内参却继续沿用原矩阵时仍会拒绝，不需要用户在 JSON 中维护另一份 signature。旧版完整标定对象和文件接口继续兼容。

手眼矩阵尚未填写或未通过验证、grasp 尚未绑定时，视觉示教会在连接 CAN 前提示补齐；单独跟踪和拍照不使用手眼或 grasp。只检查相机位姿可用 `estimate.sh --camera-only`。单独跟踪/拍照可把整个 `intrinsics` 改为 null，或临时使用 `--factory-intrinsics`，在启动时读取出厂内参；简化手眼模式需要明确填写对应的 K 和畸变参数。

默认 `mask_method: "box"`，每次 V 后在冻结图上框住完整物体，SAM3 生成掩膜；也支持 `"sam3"` 文字分割或 `"hsv"` 颜色分割。模型环境/权重沿用现有视觉安装。修改 JSON 后重启程序。

需要切换整套配置时，各入口可用 `--camera-config 另一份.json`；采集保留 `--vision-config` 作为同义参数。命令行显式参数优先于 JSON，旧版内参/手眼文件参数仍兼容。直接参数在运行时保存为不可覆盖的快照，沿用原来的来源校验，用户无需填写快照路径。

在同一终端设置并保留变量：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
task_id="pick_place_vision_$(date +%Y%m%d_%H%M%S)"
dataset="real_tpkmp/datasets/$task_id"
model_dir="real_tpkmp/models/$task_id"
bash real_tpkmp/collect.sh --can can0 --demos 8 --out "$dataset"
```

保留主从跟随，通过主臂拖动从臂。首次停在放置 TCP（物体被支撑但尚未松爪的位姿），在终端按 **L**；该点跨轮复用，放置位置改变时再按 L 更新。也可启动时加 `--place-frames "$已有数据集/demo_001_frames.json"`，只读取其中的真实 place 参考。

每组操作：

1. 摆好物体，在能完整看见目标的观察位停稳，**在终端按 V**。采集至少三帧 RGB-D 和配对的从臂 measured 位姿，执行定位及几何/手眼检查。V 处理期间保持物体与从臂不动。
2. 在冻结图中框选完整物体并 Enter；等待分割、配准完成。检查抓取预览中的物体模型、黄色 `PICK TCP` 和抓取坐标轴，**在预览窗口按 Enter 接受**，C/Esc 取消。本轮抓取位姿自动计算，不需要 P。每次 V 会启动一次静态定位，首次模型加载可能较慢。
3. 接受后保持物体不动，通过主臂回统一、安全、夹爪打开的起点；**返回终端按 R**。实际起点自动记录为 start；R 消耗本轮目标，只能用于这一次尝试。
4. 继续执行下方人工流程的第 4–8 步：闭爪前 N → 抬升前 N → 搬运前 N → 松爪前 N → 撤离前 N。
5. 成功按 **S** 保存；失败按 **X** 丢弃。下一轮无论成功或失败，都重新摆好物体、V 定位、R 录制。L 的固定放置点保留，Q/Ctrl+C 退出。

定位失败、取消、目标过期或来源文件发生改变时，R 不会沿用旧目标。录制中 V 不会触发定位。定位后若物体被挪动，先 X 清除本轮目标，再 V 重新定位；程序无法从静态照片自动知道物体后来被移动。

数据集仍保存相同格式的 `demo_*.npz` 和三参考系 JSON，可直接使用后面的训练/留出验证命令。另保存 `vision_config.json`、`vision_robot_config.json`、`place_reference_*.json` 和逐条视觉来源；`session.json` 的 `visual_runs` 指向本次 RGB-D、模型叠加、`pick_preview.png`、观察和任务参考文件。首次手眼/抓取关系、原始 RGB-D 和这份数据集应一起保留。

### 人工记录 P/L 的示教

使用 `--manual` 选择原有人工 P/L 模式，该模式不读取相机配置。在一个终端中执行，保留变量供后续命令使用：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
task_id="pick_place_$(date +%Y%m%d_%H%M%S)"
dataset="real_tpkmp/datasets/$task_id"
model_dir="real_tpkmp/models/$task_id"
bash real_tpkmp/collect.sh --manual --can can0 --demos 8 --out "$dataset"
```

此时保留主从跟随功能，直接拖动主臂，使用试教夹爪操作。脚本只接收反馈，按键仅用于记录标记，不用于控制机械臂。

每组操作：

1. 把从臂 TCP 移到将要抓住物体时的位姿，停稳后按 **P**。这是本组的 `pick` 参考 TCP，不是物体几何中心。
2. 移到物体将要放稳、尚未松爪时的位姿，停稳后按 **L**，记录 `place` 参考 TCP。P/L 可以先空手定位；随后把物体恢复到抓取位置。
3. 回到统一、安全、夹爪打开的示教起点，按 **R**，开始 `approach` 接近阶段。起点自动成为 `start` 参考 TCP。
4. 到抓取点、**闭爪之前按 N**，进入 `grasp`，闭爪并停留至夹稳。
5. **开始抬升之前按 N**，进入 `lift`；抬升至安全高度。
6. **开始搬运之前按 N**，进入 `transfer`；移到放置参考 TCP，使物体被支撑。
7. **松爪之前按 N**，进入 `release`；打开夹爪，停留至物体放稳。
8. **开始撤离之前按 N**，进入 `retreat`；撤到安全位置。
9. 确认本次确实抓起、搬运、放稳后按 **S** 保存成功示教。失败按 **X** 丢弃重来，**Q** 退出。

每阶段至少 0.2 秒且至少 3 个样本。不同示教按相同顺序标记阶段，保持相近的夹取、放置姿态。脚本会显示实际七轴角度；先确认拖动的是该 measured 反馈对应的从臂。

建议 8–15 组，先做 2–5 cm 范围内的抓取位置和放置位置变化，覆盖日后测试区域。每组重新 P/L，不要移动物体后沿用旧参考点。起点和任务顺序尽量一致。训练最少 4 组。

反馈缺失、过期、夹爪模式改变会丢弃当前尝试，已保存示教保留。需要从臂关节反馈 `0x2A5/2A6/2A7/2A9`、夹爪实际反馈 `0x2A8` 和主臂发出的正常夹爪目标 `0x159`。只有 leader 指令时不能进行此采集。

保存内容：`demo_001.npz` 等包含时间、阶段、实际七轴关节、FK TCP、夹爪实际开度、夹爪控制目标及力；对应 `demo_001_frames.json` 包含三个参考 TCP。实际开度与控制目标分开保存：物体夹住后实际开度可能仍为 20 mm，回放仍需保持原来的闭爪目标和力。

## 2. 训练和留出验证

接着在同一终端执行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/train.sh --dataset "$dataset" --out "$model_dir"
```

默认对每条示教按六阶段重采样为 96 点，使用本项目原始 TP-KMP，GMM 分量数 8。按固定种子 42 划分整条示教，约 80% 训练、20% 留出；留出示教不会参与拟合。

输出：

- `model.npz`：训练数据、任务参考系、夹爪目标、阶段时长及重建模型所需配置。加载时复建同一个 TP-KMP，不使用 pickle。
- `validation.json`：训练/留出平移 RMS、旋转 RMS、最大误差，以及留出参考系是否不同于训练参考系。
- `holdout_01.png` 等：留出示教与预测轨迹对比。

默认留出阈值为平移 RMS ≤10 mm、旋转 RMS ≤0.12 rad（约 6.9°），是初始粗抓取轨迹阈值，**不适合据此判断精密插孔精度**。失败模型可以查看离线预览，但不能真机执行。重复同一摆放的留出测试是复现测试；需要不同摆放的新示教来检查位置泛化。

新终端需要重新设置已有路径，例如：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
dataset="real_tpkmp/datasets/填写已有数据目录名称"
model_dir="real_tpkmp/models/填写已有模型目录名称"
```

## 3. 默认先生成离线测试轨迹

先在一个已采集位置复现，命令仍用同一终端的变量：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
preview_dir="real_tpkmp/runs/${task_id}_preview_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/test_real.sh \
  --model "$model_dir/model.npz" \
  --frames "$dataset/demo_001_frames.json" \
  --out "$preview_dir"
```

这一条**不打开 CAN，不运动**。查看 `preview.png` 和 `preview.json`，包括起点七轴角度、TCP 起点、起始夹爪开度、时长、最大速度/加速度和是否可执行。`plan.npz` 是下一步执行的已检查计划。

规划过程：TP-KMP 融合三个任务参考系 → 小范围修正抓取/放置阶段到所记录参考 TCP（偏差过大即拒绝）→ 夹取和松开阶段保持 TCP，前 60% 阶段内平滑改变夹爪目标、后 40% 停留等待 → SDK MDH 模型下有界数值 IK，沿用附近示教的冗余关节分支 → 关节插值 → 根据示教阶段时长及关节/TCP/夹爪速度、加速度限制整体放慢。密集采样后再次检查 FK 偏差。不是全局最优轨迹，也不根据接触力闭环调整轨迹；夹爪发送的是有上限的学习力目标。

## 4. 真机执行已预览计划

先通过主臂把从臂停在 `preview.json` 给出的起点附近，匹配起始夹爪开度。随后必须**停止主臂继续发送 CAN 控制指令**，并关闭其他运动控制程序。主臂静止时通常仍会持续发送，单纯松手不等于释放控制权。

使用已有硬件方式停止主臂发令，或隔离主臂 CAN 支路并保留电脑到从臂的通路和正确的总线终端。保持从臂已处于可接受位置控制的跟随状态且已使能。本脚本不切换主从角色、不使能、不回零；如果停主臂同时导致从臂无法保持使能，需要先解决硬件控制权切换方式，不能绕过独占检查。

确认工作区及预览后执行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/test_real.sh \
  --plan "$preview_dir/plan.npz" --execute --can can0 \
  --out "real_tpkmp/runs/${task_id}_real_$(date +%Y%m%d_%H%M%S)"
```

脚本先等待从臂新鲜反馈、检查 CAN 至少 1 秒没有其他控制源、七轴已使能、起点和夹爪匹配，然后以默认 50 Hz 下发 SDK `move_j` 位置目标和夹爪目标。使用 SDK 默认平滑位置控制，不使用 MIT/JS 快速伺服。固件内部平滑可能产生跟踪滞后，实际误差超过阈值会停止。

Ctrl+C、反馈过期、循环超时、误差过大或发现新控制源会尝试发送 SDK 阻尼急停；不会自动打开夹爪、失能或回原点。正常完成保留最后的关节目标。报告为 `execution.json`，实际反馈为 `feedback.npz`，跟踪图为 `execution.png`。`tracking_passed` 检查实际运动跟踪；物体是否被抓起、是否放稳仍需观察，脚本没有物体位姿传感器。

## 5. 新位置泛化

推荐被动记录新位置的实际参考 TCP：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
new_frames="real_tpkmp/runs/${task_id}_new_frames_$(date +%Y%m%d_%H%M%S).json"
bash real_tpkmp/run.sh frames.py --can can0 --out "$new_frames"
```

主从跟随期间停稳后，S 记录起点，P 记录抓取位姿，L 记录放置位姿，W 保存。此程序只接收 CAN。抓取/放置参考点定义与采集一致；本入口使用人工参考点。要从 D435i 自动更新抓取目标，使用独立的 [视觉流程](vision/meshy/WORKFLOW.md)。

若物体只沿基座 X 平移 10 mm，也可从已有参考系生成对应文件：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
new_frames="real_tpkmp/runs/${task_id}_shift_$(date +%Y%m%d_%H%M%S).json"
bash real_tpkmp/run.sh frames.py \
  --from-frames "$dataset/demo_001_frames.json" --pick-offset-m 0.01 0 0 \
  --out "$new_frames"
```

必须将实际物体也移到对应位置；JSON 偏移不会移动真机或物体。随后重新预览，审核并按上一节执行：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
preview_dir="real_tpkmp/runs/${task_id}_generalize_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/test_real.sh --model "$model_dir/model.npz" \
  --frames "$new_frames" --out "$preview_dir"
```

默认新参考系需要在至少一组训练示教的对应参考系附近：每个参考点平移变化 ≤80 mm、转角 ≤0.35 rad。这个范围限制不能保证物体可抓或避免碰撞。

更可靠的离线泛化评价是重新采集一组不同位置示教，不加入原模型训练，再验证：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
check_data="real_tpkmp/datasets/${task_id}_check_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/collect.sh --manual --can can0 --demos 4 --out "$check_data"
bash real_tpkmp/validate.sh --model "$model_dir/model.npz" \
  --dataset "$check_data" \
  --out "real_tpkmp/runs/${task_id}_validation_$(date +%Y%m%d_%H%M%S)"
```

验证只重建原训练模型，不用新数据重估参数，也不会改变原模型的留出通过标记。

## 6. 离线自检与数据格式

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/run.sh self_test.py
```

测试覆盖 measured/leader 区分、分帧与过期反馈、夹爪 width/angle 和零位命令拒绝、SDK FK/发包一致性、控制权冲突、IK 不可达、跟踪错误，以及反馈中断后的停止和报告。不会打开 SocketCAN。

`offline_check/` 保留开发时合成数据的训练、留出、独立验证和规划结果供查看，全部明确标记 `synthetic=true`，禁止用于真机。要自己重跑完整离线流程：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
smoke="real_tpkmp/offline_check/check_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/run.sh make_synthetic.py --out "$smoke/dataset"
bash real_tpkmp/train.sh --dataset "$smoke/dataset" --out "$smoke/model"
bash real_tpkmp/test_real.sh --model "$smoke/model/model.npz" \
  --frames "$smoke/dataset/demo_001_frames.json" --out "$smoke/preview"
```

输出均排他创建，不覆盖已有文件；换时间戳/目录重跑。数据格式版本为 1，位姿始终为米和 `[qw,qx,qy,qz]`，关节为弧度；独立目录中的 NPZ 格式不直接混入旧仿真训练入口。代码与离线流程已验证，真机运动、实际抓取效果仍需在你的硬件上按预览流程检查。
