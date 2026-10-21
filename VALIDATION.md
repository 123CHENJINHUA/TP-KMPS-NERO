# 仿真控制修复与验证（2026-09-17）

## keel 实体模型抓取（2026-09-23）

- 原始 `keel.STL` 和 `keel.usd` 已复制到 `assets/objects/keel/`，源文件 SHA256 保存在 `object.json`。
- 用户确认单位为毫米；派生 `keel_grasp.usda` 统一为米、Z 向上，使用原始三角网格的凸分解碰撞。
- 平台上固定位置 `[0.30, 0, 0.172]` m（夹持参考点），仿真质量 80 g，两次各 10 秒。
- 在本机 RTX 4070 Ti SUPER、Isaac Sim 5.1 中运行 `--mode grasp --object keel --grasp-spread 0 --demos 2 --duration 10 --headless`。
- 两次均通过：终点抬升 59.043 mm，最终保持 1 秒；最大夹持参考点误差 1.321 mm，相对滑动约 0.025 mm。
- 原始记录：`recordings/keel_check_01/`。仅验证固定位置/姿态和给定仿真质量，未验证任意物体或真实抓取。


## 扩展位置与多组采集（2026-09-22）

- 查看用户回传的 `recordings/grasp_check_02/summary.json`：基础抓取 2/2 成功，最后保持阶段抬升约 59 mm。
  这是用户运行后的实测记录，并非当前开发环境重新执行所得。
- 新增 X/Y 独立范围、分层随机抽样、单进程多组采集。每组独立种子、位置清单和结果报告。
- 支持 X `[0.28, 0.34]` m、Y `[-0.05, 0.05]` m。URDF 有界数值 IK 检查覆盖原区域及扩展区域的角点和边缘中点。
- 离线回归 21 项通过，新增每轴分层覆盖、固定种子复现、保留旧随机分布和范围校验测试。
- `output/grasp_positions_preview/` 是种子 42/43/44、每组 12 个计划位置的预览，不是已采集的示教。
- 扩展区域尚未在当前环境完成物理采集；实际执行时仍需通过逐条 Lula IK 预检查和抓取成功检查。

## 抓取打滑修正（2026-09-22）

- 分析用户实测 `recordings/grasp_check_01/failed_001.npz` 和 `failed_002.npz`：
  抬升前两指间距均约 30 mm；方块最大抬升分别为 8.1895 mm 和 2.0950 mm，随后落回平台，夹爪最终闭合到约 0。
  数据支持“接触方块后在抬升时丢失夹持”，而不是没有执行闭合或初始位置明显偏离。
- 仅对两个夹爪滑动关节显式设置 USD `force` 驱动；刚度 600 N/m、阻尼 10 N·s/m、每指力上限 10 N。
  原数据未记录驱动类型，不能仅凭记录确认当时是否为 acceleration；新输出 `physics.json` 保留此信息。
- 静摩擦系数从 1.0 调至 1.2，动摩擦保持 1.0；启用 solveArticulationContactLast，增加位置求解迭代数至 64。
- 失败结果增加抬升前指距、最高抬升和原因；正常的失败汇总以状态码 2 退出，不再打印 Python 异常堆栈。
- 离线回归：18 项通过。使用本机 Isaac Sim 自带 USD 库的内存 Stage，验证两个 acceleration 测试夹爪被改为 force、增益和力上限正确、机械臂测试关节未改变。
- 本次尚未进行修改后的 PhysX 抓取实测，不能据此声称抓取已经成功。原始失败数据保持不变。

## 自动抓取模式离线检查（2026-09-22）

- 新增 `--mode grasp`：动态方块、碰撞平台、分阶段抓取与抬升、成功和失败数据分开保存。
- 夹爪命令相位保留到技能和预测文件，抓取回放启用相同物理场景。
- `/usr/bin/python3 -m unittest discover -s tests -v`：16 项通过；Python 编译检查通过。
- 包含独立 URDF 正运动学和有界数值 IK 检查：默认中心及 XY ±20 mm 四个角落，下降和抬升路径可达。
- 包含失败判定回归：仅机械臂抬升、瞬间弹起、物体偏离、夹爪空夹、保持不足和相对滑落均不能作为成功。
- **未完成物理抓取实测**：当前环境没有 `/dev/nvidia*`，`nvidia-smi` 无法连接驱动。
  当前环境也缺少原训练 Python 环境；本次验证覆盖数据格式与夹爪时序传递，
  训练/预测文件衔接测试替换了数值模型调用，未重新运行 TP-KMP 数值训练。
- 以下 2026-09-17 的物理验证只适用于此前的自由空间模式，不代表新增抓取模式已经通过。

环境：Isaac Sim 5.1、本机 RTX 3060，世界重力开启。未操作真实机械臂，也未关闭用户正在录制的仿真窗口。

## 修复

1. 关节目标限速从“实际关节位置加一个有限步长”改为“上一帧已下发目标加一个有限步长”。旧实现会把下垂持续带入目标，限制位置驱动能够形成的纠偏误差。
2. 每个物理步使用 PhysX 原生重力补偿力矩，再推进仿真。没有关闭世界或机械臂重力，也没有通过逐帧瞬移关节生成轨迹。
3. 恢复 Isaac Sim 默认快速退出，避免逐个卸载扩展时发生退出挂起/原生崩溃；异常时显式请求非零退出码。

## 结果

- 离线回归：4 项通过，包含“实际关节滞后时，目标仍应按限速持续推进”的回归测试。
- 只修正目标更新、不补偿重力时：20 mm 上抬检查仍有 5.15 mm 终点误差。
- 加入补偿后：20 mm 上抬检查通过，终点位置误差 1.619 mm，姿态误差 0.00264 rad；阈值为 2 mm / 0.02 rad。
- 实际 PhysX 自动采集：3 条示教，每条 121 个样本，4 秒。
- 实际 TP-KMP 训练、目标向上平移 20 mm 的泛化、Isaac Sim 回放均完成。
- 泛化轨迹回放：121 个样本，位置 RMSE 0.543 mm、最大位置误差 0.792 mm、终点误差 0.441 mm；最大姿态误差 0.0102 rad，检查通过。

原始报告和回放数据放在 `validation/gravity_fix/`。这些结果只覆盖 HOME 附近的小范围自由空间轨迹，不能代表所有姿态、接触任务或真机精度。

旧窗口继续运行旧代码。保存数据并关闭它，重新启动 `scripts/sim.sh` 后修复生效。之前明显下垂或跟踪失真的示教应重新录制，不要混入新模型。

## Xbox input (2026-09-23)

- Ported joy_pkg TCP-frame stick/trigger/A/B mapping and gripper closure-percent semantics to native Linux joydev input. Interactive Xbox mode does not subscribe to keyboard control events.
- `python -m unittest discover -s tests -v` in tpkmp: 30 tests passed, including six Xbox tests covering frame/sign conventions, deadzone, analog scaling, initialization, disconnect, rearming, gripper repeat/clamping, USB/Bluetooth axis discovery, and permission errors.
- `bash scripts/sim.sh --mode interactive --control xbox --headless --steps 30 --output-dir output/xbox_smoke_01`: exit 0; Isaac Sim reached NERO_READY and completed the interactive loop.
- No Xbox device was connected during verification. This is not an end-to-end physical-controller test.

## 双 Nero 固定支架与 Y 键切换（2026-09-24）

- 新增 `--mode dual --control xbox`：两台独立的固定基座 Nero，默认间距 0.9 m、支架高度 0.5 m，面向中央工作区。
- 每台机械臂独立保存 IK、关节控制、末端目标和夹爪状态；Y 键仅在手柄可用、时间轴播放时按键沿切换，并要求重新回中。
- 本机 Isaac Sim 5.1 无窗口 30 步验证成功：`/World/nero` 和 `/World/nero2` 是独立层级；基座分别在 Y=-0.45/+0.45 m、Z=0.5 m，两臂各自通过 1 mm 末端 IK 预检查。
- tpkmp 环境 `python -m unittest discover -s tests -q`：33 项通过。测试覆盖 Y 键沿触发、双臂基座朝向及第二 URDF 资源独立性。
- 本机未连接 Xbox，硬件按键与实际切换尚未经过手柄端到端测试；尚未做双臂相互避障或真机控制。

## 手柄 X/Y 按键修正（2026-09-24）

当前连接的 `/dev/input/js0` 按钮映射：index 2 为 0x133（X），index 3 为 0x134（Y）。
将双臂切换键改为 0x134，并加入 X 不切换、Y 切换的回归测试。
