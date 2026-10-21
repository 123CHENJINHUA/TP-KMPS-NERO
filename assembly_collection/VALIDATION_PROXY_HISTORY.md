# 已停用：固定约束模型的历史验证记录

**下述结果来自旧版固定约束模型，不能证明无约束的真实重力装配成功，也不能作为当前接触版本的成功示范。** 当前程序已删除所有零件装配约束，数据检查器默认排除这些 v1 数据。当前验证见 [VALIDATION.md](VALIDATION.md)。

2026-10-07，在本机 Isaac Sim 5.1 中使用 `--headless` 实际执行；物理频率 120 Hz，记录频率 30 Hz。全部使用独立运行舞台，原场景和原控制程序未修改。

当前路径与接触配置已完成四次四颗螺丝装配。最终脚本的运行时 keel 贯穿检查由 `full_10` 完整验证；其他三次保存的轨迹也通过了更新后的数据审计和贯穿检查。

| 试验目录 | 结果 | keel 初始变化 X/Y（mm） | 采样数 | 两臂松手后 keel 位置误差（mm） |
| --- | --- | --- | --- | --- |
| `validation/full_09/episode_001` | 4/4 颗安装通过 | 0 / 0 | 10574 | 0.489 |
| `validation/random_01/episode_001` | 4/4 颗安装通过 | +1.644 / -0.367 | 10660 | 0.488 |
| `validation/random_01/episode_002` | 4/4 颗安装通过；同进程自动复位 | +2.152 / +1.184 | 10667 | 0.488 |
| `validation/full_10/episode_001` | 最终脚本 4/4 颗安装通过 | 0 / 0 | 10574 | 0.489 |

最终脚本 `full_10` 在两臂松手后测得：

| 螺丝 | 入墙深度（mm） | 墙孔轴线最大横向误差（mm） | 实际 keel 底板两面最大轴线偏差（mm） | keel 孔剩余径向间隙（mm） |
| --- | --- | --- | --- | --- |
| 4 | 24.306 | 0.064 | 0.070 | 1.250 |
| 1 | 24.300 | 0.001 | 0.967 | 2.000 |
| 3 | 24.301 | 0.029 | 0.471 | 1.250 |
| 2 | 24.300 | 0.002 | 0.848 | 2.000 |

四枚轴线均贯穿实际 keel 底板；偏差加 0.15 mm 后仍在各孔间隙内。两臂最后实际开度均为 70 mm。左臂实际抬起 keel 约 115 mm，右臂每颗实际抬升约 80 mm。keel 在首颗固定后有小幅面内转动，最终角误差约 0.0068 rad，已在上述实际孔检查中计入。

另外通过：

- `validation/preflight_final`：两臂抓取和四孔装配端点均可达。
- `python -B -m unittest assembly_collection.test_geometry`：4 项位姿、对孔、入墙深度和实际 keel 孔验证测试。
- `check_dataset.py`：全部四次成功示范的数据形状、时间递增、四元数、阶段标记、松爪和最终孔位检查通过。
- Python 源文件语法与 `scripts/assembly.sh` 的 Bash 语法检查。

参数：墙沿 X 前移 120 mm、keel 起点沿 X 前移 55 mm、料架沿 Y 移开 40 mm，螺丝夹持头高度 40 mm。模型的物理假设、固定约束与训练格式限制见 [README.md](README.md)。这四次验证覆盖当前布局和小范围初始位置变化，不代表任意摆放或其他几何都可装配成功。

最终运行报告：[report.json](validation/full_10/episode_001/report.json)；轨迹图：[validation.png](validation/full_10/episode_001/validation.png)；连续随机位置批次：[summary.json](validation/random_01/summary.json)。

此目录也保留了开发中的失败试验及日志，均标为 `success=false`，用于排查抓取、避碰和插入问题。采集训练数据时应使用独立的 `recordings/assembly_*` 输出目录，并筛选成功示范。
