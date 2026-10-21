将同一个物体的 1–4 张真实照片放在此目录，例如 `keel/front.jpg`、`keel/side.jpg`、`keel/back.jpg`、`keel/oblique.jpg`。不同角度分别保存为图片，不拼成一张图。也可以在命令中指定其他本地图片路径。

照片只在执行 `meshy.sh generate`（不带 `--dry-run`）时上传 Meshy；网页生成后可直接离线导入模型，无需 API key。不要把密钥写进此目录。
