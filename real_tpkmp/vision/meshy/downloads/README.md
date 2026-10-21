# 下载的原始模型

把 Meshy 网页或其他网站下载、与实际物体对应的模型放在这里，建议每个物体一个子目录：

```text
downloads/
  keel/
    model.glb
  other_object/
    model.obj
    model.mtl
    textures/
```

压缩包先解压。推荐 GLB 或 STL；OBJ 要保留 MTL 和纹理的相对目录。这里存原始文件，不手动覆盖 `vision/models/` 中已导入和校正的模型。

下一步见 [从模型到真机采集和测试](../WORKFLOW.md)。
