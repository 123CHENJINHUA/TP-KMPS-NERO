# 标定板配置

新相机内参板使用 [charuco_rows9_cols14_5x5_20mm.json](charuco_rows9_cols14_5x5_20mm.json)。用户确认 Rows=9、Columns=14，标准 5×5 字典，编号从 0 开始。

| 参数 | 新板 |
| --- | --- |
| 方格数 | 14 列 × 9 行（不是内角点数） |
| 方格边长 | 20 mm |
| marker 边长 | 15 mm |
| 棋盘区域 | 280 × 180 mm |
| ChArUco 内角点 | 104，编号 0–103 |
| marker | 63，编号 0–62 |
| 配置字典 | DICT_5X5_100 |
| 连续四格实测长度 | 横纵均应为 80 mm |

5×5 是单个 marker 的编码格数。当前 OpenCV 中 DICT_5X5_100/250/1000 的前 63 个码字相同，已逐项核对，配置使用 DICT_5X5_100。DICT_5X5_50 只有 0–49，不能容纳本板全部 marker；程序会提前拒绝容量不足的配置。

参考 [PNG](../boards/charuco_rows9_cols14_5x5_20mm/board.png) 和 [A4 横向 PDF](../boards/charuco_rows9_cols14_5x5_20mm/board_A4_100percent.pdf) 已生成。打印选择实际大小 100%，打印后实测 20 mm 方格和 15 mm marker；棋盘区域尺寸不含页边距。

采集或检查板时，指定 `--board-config hand_eye_calibration/board_configs/charuco_rows9_cols14_5x5_20mm.json`。内参求解从采集会话 metadata 读取板几何，运行 calibrate_intrinsics.py 时不另指定板。新板完整有 104 个角点；内参采集示例使用 `--min-corners 40`，允许部分板仍在画面中时采集图像边缘，但每张必须有足够清晰的角点。

旧手眼板配置 [board_config.json](../board_config.json) 保持 7 列×5 行、40/30 mm、DICT_6X6_50。相机内参可以用新板求出，手眼再用旧板：手眼采集使用默认旧板配置并加新内参。若手眼采集也实际使用新板，就同时指定新板配置；始终按镜头前的实物选择，不能混合不同板的采集会话。完整复制命令见 [COMMANDS.md 9.7](../../COMMANDS.md#97-当前从臂结果仍未通过独立检查-rgb-内参)。
