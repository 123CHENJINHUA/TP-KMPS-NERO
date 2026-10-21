# Meshy 照片生成模型 → D435i 物体定位

单图和多图都可以。建议先对同一实物拍 3–4 张不同角度照片；Meshy 生成模型后，在本地校正尺寸，再交给已有 SAM3 + FoundationPose 定位。所有新增脚本、照片副本、模型和输出都在 `real_tpkmp/vision/`，不控制机械臂。

已下载模型、准备采集真机示教时，按 [从模型到采集、训练和真机测试](WORKFLOW.md) 操作。原始模型统一建议放在 `meshy/downloads/<物体名>/`，图片放在 `meshy/images/`，导入后的米制模型在 `vision/models/<物体名>/`。

这是**生成式的近似模型**。只有图片时，遮挡面和细节可能由模型补全；即便渲染看起来相似，也不能据此确认孔径、孔距、薄壁或接触面精度。已有精确 keel CAD 应继续保留。初期先用于粗抓取/搬运验证；螺栓插孔仍应使用实测几何及接触校正。

## 拍摄要求

- 同一个物体，分别拍正面、侧面、背面、斜上方，最多 4 张；单图时用能看清最多结构的斜视图。
- 第一张作为主视图。物体完整入镜，背景干净、光照均匀，避免夹爪/手遮挡。几何结构在不同照片中保持一致。
- 使用真实拍摄的不同视角；不要重复同一张图，也不要用 AI 补出来的视角冒充测量。
- 推荐照片放在 `real_tpkmp/vision/meshy/images/keel/`。手机照片或 D435i RGB 均可。已有 `capture.sh --no-robot` 的 `rgb.png` 也可用；每次改变视角后单独拍摄，三帧静态连拍不等于三个不同视角。
- 用尺测量实物长、宽、高，保留毫米值。单靠照片或 Meshy `auto_size` 的估计不能作为真机的尺度来源。

官方 [单图 API](https://docs.meshy.ai/en/api/image-to-3d) 和 [多图 API](https://docs.meshy.ai/en/api/multi-image-to-3d) 支持 JPG/JPEG、PNG、WebP，每张最多 20 MB；多图为 1–4 张。当前固定使用 `meshy-7.1`，不跟随 `latest` 的自动版本变化。核对日期：2026-10-20。

## A. 在 Meshy 网页生成，再离线导入

打开 [Meshy Image to 3D](https://www.meshy.ai/features/image-to-3d)，上传单图或选择 Multi-view 上传同一物体的多视图，生成后下载 **GLB 或 STL**。GLB 可保留纹理和材质；导入时会另生成米制 STL，供当前配准程序读取。

例如将下载文件保存为 `real_tpkmp/vision/meshy/downloads/keel/model.glb`，实物确认为 150×98×23 mm：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh import \
  --mesh real_tpkmp/vision/meshy/downloads/keel/model.glb \
  --object-id keel_meshy \
  --measured-extents-mm 150 98 23 \
  --prompt 'blue metal bracket'
```

输出在 `real_tpkmp/vision/models/keel_meshy/`：

| 文件 | 用途 |
| --- | --- |
| `source.glb`（或原格式） | 原始模型副本 |
| `model.stl` | 已均匀缩放为米、以包围盒中心为原点的新物体模型 |
| `model.glb` | 相同坐标系/尺寸的展示模型，保留可读取的材质 |
| `preview.png` | 四视图和 XYZ 坐标轴；红 X、绿 Y、蓝 Z |
| `geometry.json`、`import.json` | 实测尺寸、缩放、三轴误差、变换和 SHA256 |
| `object.json` | 可直接传给 `estimate.sh --object` 的配置；只在尺寸检查通过后生成 |

导入使用已有 `vision/envs/foundationpose` 的 trimesh，**只进行 CPU 网格处理，不运行 GPU 模型**，不需要 Meshy API key。也支持 OBJ/PLY 三角网格；OBJ 的 MTL/纹理要与原文件保持可解析的相对路径。

只做**一次等比例缩放**，让最长包围盒边与实测最长边一致；另外两边按大小排序比较，门限为各实测边长的 10% + 1 mm。这仅筛除明显的比例错误，不能证明毫米级精度。不会分别拉伸 XYZ 三轴去掩盖生成的形状误差。

尺寸比较假定物体长宽高已与模型坐标轴对齐。如果导出的网格斜放，先看预览，用 `--rotate-xyz-deg RX RY RZ` 调整，或在 Blender 中摆正后重新导入。新尝试使用新 `--object-id` / `--out`，不覆盖之前结果。尺寸失败时保留模型和预览供检查，不输出可用配置。

## B. 在脚本中调用 Meshy API

API 会上传照片到 Meshy，并按账户规则消耗积分；运行下面的本地检查不联网、不扣费，也不要求 key。输入文件名对应你实际准备的照片。

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
meshy_run="real_tpkmp/vision/meshy/jobs/keel_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/meshy.sh generate --dry-run \
  --images real_tpkmp/vision/meshy/images/keel/front.jpg \
           real_tpkmp/vision/meshy/images/keel/side.jpg \
           real_tpkmp/vision/meshy/images/keel/back.jpg \
           real_tpkmp/vision/meshy/images/keel/oblique.jpg \
  --out "$meshy_run"
```

只留一个 `--images` 后的路径就是单图；程序自动选择 `image-to-3d` 或 `multi-image-to-3d` API。本地保存快照与 SHA256，参数中不写 API key 或大段 base64。默认不生成纹理、不重拓扑、关闭 `auto_size` 和图像风格增强。可在首次 `generate` 时加 `--texture`、`--geometry-resolution 2k` 或 `--polycount 50000`；这些选项可能改变积分和处理时间，不应把加面数当成精度保证。

在同一终端中从 [Meshy 开发者平台](https://www.meshy.ai/api) 获取你自己的 key，并提交准备好的任务：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
read -r -s -p 'Meshy API key: ' MESHY_API_KEY
echo
export MESHY_API_KEY
bash real_tpkmp/vision/meshy.sh submit --run "$meshy_run"
```

不要把 key 写进项目文件或发到聊天中。也可以省略 dry-run，在设置好 key 后直接 `generate --images ... --out ...`。任务成功会保存原始 `model.stl` / `model.glb`、`task.json` 和下载校验值；这些原始模型还没有实测尺度校正。

终端断开、等待超时或下载中断后，继续已有任务，不创建新任务：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh resume --run "$meshy_run"
```

创建任务的 POST 不自动重试。若刚好在提交后丢失了响应，工具会阻止盲目重提；先在 Meshy 网页找到对应任务，再 `resume --run "$meshy_run" --task-id 真实任务编号`。`resume` 只查询和下载，不创建任务。达到本地等待时间不代表取消云端生成。下载时不给资源服务器发送 API key，已有文件校验不一致时不覆盖。

下载完成后导入；实测值以下仍以原 keel 为例：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh import \
  --mesh "$meshy_run/model.stl" --object-id keel_meshy_api \
  --measured-extents-mm 150 98 23 --prompt 'blue metal bracket'
unset MESHY_API_KEY
```

原始文件保存在 jobs，标准化模型在 models，当前物体配置仍默认原有 CAD，**不会自动替换**。如希望保留展示纹理，可导入 `$meshy_run/model.glb`；当前 FoundationPose 仍使用导入输出的 STL 几何，不读取展示 GLB 的纹理。

## C. 接回现有视觉和 TP-KMP

先用真实 D435i RGB-D 检查模型的轮廓和深度是否匹配，可暂时不带手眼结果，仅检查相机中的物体位姿。默认读取当前 RGB 流的出厂内参；仅在需要自行标定内参时显式加 `--intrinsics 路径/intrinsics.json`：

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
meshy_check="real_tpkmp/vision/runs/meshy_check_$(date +%Y%m%d_%H%M%S)"
bash real_tpkmp/vision/capture.sh --no-robot \
  --out "$meshy_check/capture"
bash real_tpkmp/vision/estimate.sh --capture "$meshy_check/capture" \
  --object real_tpkmp/vision/models/keel_meshy/object.json \
  --out "$meshy_check/pose"
```

这里的 `keel_meshy` 对应网页例子；API 例子改成 `keel_meshy_api`。检查 `overlay.png` 和 `observation.json`。省略手眼且没有机器人反馈，结果只能诊断；眼在手上正式使用时，按[原视觉流程](../README.md)拍照并读取 measured，传入当前有效手眼结果。

后续 `estimate.sh` **每次都要带新模型的 `--object`**，否则默认使用原有 keel CAD。新模型有新的原点/朝向与 SHA256，必须为它重新 `bind-grasp`，不能使用原 CAD 的 grasp 文件。更换模型不会自动改 TP-KMP 的示教或训练数据；抓取关系绑定后，新位置定位→frames→离线预览沿用原有流程和限制。

## 验证范围

```bash
cd /home/cjh/Downloads/TP-KMPs-nero-isaacsim
bash real_tpkmp/vision/meshy.sh self-test
```

测试覆盖单/多图负载、图片快照校验、dry-run 不联网、提交失败不重复 POST、已有任务下载和文件校验、真实 STL 的单位/坐标变换与配置兼容，以及比例错误拒绝。API 使用 mock；本次没有上传照片、调用实际 Meshy 付费任务或验证其生成质量，也没有打开相机/CAN 或移动真机。
