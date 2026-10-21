# 外部代码与模型来源

- `third_party/FoundationPose`：NVIDIA 官方 FoundationPose，下载时提交 `a1b694b83e633c2cb6115b9063d940a687759392`。保留上游 LICENSE 与源文件版权声明；本项目不改动其算法源码。https://github.com/NVlabs/FoundationPose
- `third_party/sam3`：从本机已有 SAM3 源码复制，保留 LICENSE；图像 API 按官方仓库及本机源码核对。https://github.com/facebookresearch/sam3
- `weights/sam3.pt`：本机已有权重的独立副本，不包含任何账号凭据。
- FoundationPose 两组权重从 `/home/cjh/FoundationPose/weights/` 复制到本目录，与官方发布的版本目录一致。原始文件未改动。
- `models/keel/keel.STL`：本项目现有、用户提供并确认毫米尺寸的 CAD 模型副本，来源和 SHA256 在 `objects/keel.json`。
- FoundationPose++ 作为跟踪工程参考：https://github.com/teal024/FoundationPose-plus-plus
- FoundationPose-- 作为后续加速候选参考：https://github.com/ziqin-h/FoundationPose--
- PyTorch3D 编译依赖提交 `88e182f989c80836f4bd744e0d9cb1852762ce01`，源代码与许可证保留在 `third_party/pytorch3d/`。
- nvdiffrast 编译依赖提交 `253ac4fcea7de5f396371124af597e6cc957bfae`，保留在 `third_party/nvdiffrast/`。

新增封装代码直接调用公开 API，不把上游研究用途条款替换为本项目许可证。实际测试范围记录在运行输出及 README 中。
