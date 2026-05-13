# tests/fixtures/

本目录放测试用的小文件 fixture. 入库是**例外**, 仅限严格必需的 e2e 测试使用.

## mini_pingpong_5s.mp4

**用途**: 002 feature (raw-video-feature-bmn) 的 e2e 集成测试 — `pp extract-feat` / `pp infer-rawvideo` / `pp build-feature-pkls`.

**规格**:
- 分辨率: 224×224 (= PP-TSM inference 输入分辨率, 节省预处理环节)
- 时长: 5 秒
- 帧率: 25 fps (= BMN GT fps, 与 ffmpeg 抽帧强制 fps 一致)
- 帧数: 125
- 编码: H.264 yuv420p (与真实手机/相机视频兼容)
- 文件大小: ~245 KB

**合成命令** (需要 ffmpeg 4.x+):

```bash
ffmpeg -y -f lavfi -i "testsrc2=size=224x224:rate=25:duration=5" \
    -c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p \
    tests/fixtures/mini_pingpong_5s.mp4
```

- `testsrc2` 产生一段**合成测试图样** (动态色块 + 移动计数器), 像素值分布丰富但不含任何真实人物/版权内容.
- `-preset ultrafast + -crf 28` 让文件尽量小.
- `-pix_fmt yuv420p` 保证与 macOS/iOS 播放器兼容 (即使只做测试, 也避免奇怪问题).

**为什么不用真实乒乓球视频**:
1. 版权: COS bucket 上的 `PP15pingskills/*.flv` 有教程版权, 不适合入库公开仓库.
2. 文件大小: 真实 5 分钟视频 ~5 MB, 仅 5 秒版本也会让 repo 膨胀.
3. 测试可重现性: 真实视频的随机压缩噪声会让 PP-TSM 抽出的特征在不同机器上轻微不同 (违反 SC-013). `testsrc2` 是确定性合成, 抽出的特征跨机器 bit-wise 一致.

**SC-013 兼容性**:
- 同一机器上多次抽特征结果必须 bit-wise 一致 (static graph 推理是确定性的).
- 跨不同 paddle 版本/ffmpeg 版本, 特征**数值**可能小差异 (ε), 但 cosine 相似度 ≥ 0.999 (SC-013 硬门槛).

**测试用例**:
- `tests/integration/test_extract_feat_e2e.py`: 跑 `pp extract-feat --input mini_pingpong_5s.mp4` → 检查 `.pkl['image_feature'].shape == (125, 2048)` 与 `dtype == float32`.
- `tests/integration/test_infer_rawvideo_e2e.py`: 端到端 `pp infer-rawvideo` → 检查 timeline.json schema.
- `tests/integration/test_build_feature_pkls_e2e.py`: 把该文件复制 3 份, 跑 `pp build-feature-pkls` → 检查 3 份 .pkl + manifest.csv + (可选) label_cls14_<name>.json.

**重新生成**:
如果 ffmpeg 升级到不兼容版本 (例如 pix_fmt 变化), 重新跑上面命令即可. 合成内容是**完全确定性**的 (`testsrc2` 无随机成分), 同一 ffmpeg 版本下多次生成结果 bit-wise 一致.
