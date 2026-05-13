# Quickstart: 原始视频到 BMN 时序定位 (002)

**功能**: 002-raw-video-feature-bmn
**日期**: 2026-05-13
**前置**: 001 quickstart.md (`env-check / data-prepare / train / eval` 已介绍); v0.2.x BMN 训练已完成.

本文件**只**介绍 002 feature 引入的新流程. 其它步骤 (env / 训练 / 评估) 见 001 quickstart 与 README.

---

## 场景 A: 用我的乒乓球视频跑端到端推理 (US1, P1)

**前提**: 你已经有一份本项目 v0.2.x 训过的 BMN 权重 (例如 `experiments/<run>/BMN_epoch_00020.pdparams`).

```bash
# (一次性) 1. 下载 PP-TSM 训练权重 (120MB, 公开)
mkdir -p data/raw/pretrained
curl -fL -o data/raw/pretrained/ppTSM_k400_dense.pdparams \
    https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams

# (一次性) 2. 导出 inference 模型 (30s, 自动转换)
.venv/bin/python scripts/export_pptsm_inference.py

# 3. 端到端推理 (你的视频 → 时间轴 + 可视化)
.venv/bin/pp infer-rawvideo \
    --input my_pingpong_video.mp4 \
    --bmn-checkpoint experiments/<run>/BMN_epoch_00020.pdparams \
    --output-dir outputs/my_run/
```

**预期产物** (`outputs/my_run/`):
- `timeline.json` — 候选时间区间列表, schema `rawvideo-timeline-v1`
- `my_pingpong_video_visualized.mp4` — 在原视频上叠加预测的可视化版
- `feature.pkl` — 中间产物 (12 维 fp32 特征序列), 重跑时可用 `--keep-features=false` 关
- `run.log` — 全流程日志

**预计耗时** (T4 GPU):

| 视频时长 | extract_feat | bmn_forward | bmn_postprocess | visualize | **合计** |
|---------|-------------|-------------|-----------------|-----------|---------|
| 3 分钟 (180s) | 38s | 25s | 18s | 9s | **~90s** |
| 10 分钟 | 125s | 80s | 60s | 28s | ~290s |
| 30 分钟 | 380s | 240s | 180s | 85s | ~890s |

满足 **SC-010 ≤ 5 min** (3 分钟视频).

### 检查输出

```bash
# 看候选区间数量
python -c "
import json
data = json.load(open('outputs/my_run/timeline.json'))
print(f\"schema: {data['schema']}\")
print(f\"input: {data['input_video']}\")
print(f\"n_proposals: {len(data['results'])}\")
print(f\"top 3 by score:\")
for r in sorted(data['results'], key=lambda x: -x['score'])[:3]:
    print(f\"  {r['start_sec']:.1f}-{r['end_sec']:.1f}s {r['label_name']:6s} score={r['score']:.3f}\")
"
```

预期输出 (示例):
```
schema: rawvideo-timeline-v1
input: /abs/path/my_pingpong_video.mp4
n_proposals: 47
top 3 by score:
  12.4-14.2s 侧旋     score=0.842
  98.7-100.5s 拉      score=0.781
  156.3-158.1s 摆短   score=0.733
```

---

## 场景 B: 用我的视频集合扩充训练数据 (US2, P2)

**前提**: 你有一组新拍/新收集的 mp4 + 自己手写的 GT JSON (按 `label_cls14_train.json` schema, 14 类不变).

```bash
# 1. 把视频目录批量转 .pkl + 重写 GT JSON
.venv/bin/pp build-feature-pkls \
    --videos-dir my_videos/ \
    --output-dir data/clips/my_extension/ \
    --gt-json my_label.json \
    --name my_ext

# 2. 转换为 BMN 训练输入 (复用 v0.2.x 已有脚本)
.venv/bin/python scripts/prepare_bmn_inputs.py \
    --label-json data/clips/my_extension/label_cls14_my_ext.json \
    --feature-dir data/clips/my_extension/Features_my_ext/ \
    --output-dir data/bmn_inputs/my_ext/

# 3. (可选) 微调现有 BMN 基线 — US3
#    复制 bmn_pingpong.yaml 为本次微调专用版本 (改 bmn_inputs_dir 指向你的数据)
cp configs/models/bmn_pingpong.yaml configs/models/bmn_my_ext.yaml
# 编辑 bmn_my_ext.yaml: 把 model.bmn_inputs_dir 改成 data/bmn_inputs/my_ext/
# 或者用 sed 一行解决:
sed -i 's|bmn_inputs_dir: null|bmn_inputs_dir: data/bmn_inputs/my_ext/|' configs/models/bmn_my_ext.yaml

.venv/bin/pp train \
    --config configs/models/bmn_my_ext.yaml \
    --resume experiments/<baseline>/BMN_epoch_00020.pdparams \
    --allow-dirty
```

**预期产物** (`data/clips/my_extension/`):
- `Features_my_ext/<32-hex>.pkl` (每个视频一份, 文件名是内容 hash, 抗改名)
- `manifest.csv` (13 列, 含每个视频的元数据)
- `label_cls14_my_ext.json` (与 14 类标签体系兼容, url 字段已重写为 `<32-hex>.mp4`)

**预计耗时** (T4 GPU, batch_size=32, 100 分钟视频合计):
- 抽特征: ~31 分钟 (满足 SC-011 ≥ 80 fps/sec)
- 写 manifest + label: <1 分钟
- 总计 ~32 分钟

满足 **SC-011 视频时长的 0.5 倍以内**.

### 中断后续传

如果命令中途因 OOM / Ctrl-C 中断, **重新运行同一条命令**, 已成功写出的 .pkl 会被自动跳过 (FR-034 幂等性). 检查 manifest.csv 中 `error` 列即可知道哪些跳过, 哪些失败.

---

## 场景 C: 单视频抽特征 (调试用, FR-033)

如果你只想对一个视频抽特征 (例如调试或与上游 .pkl 比对):

```bash
.venv/bin/pp extract-feat --input my_video.mp4

# 默认输出: <my_video同目录>/<sha256(file_bytes)[:32]>.pkl
```

输出 .pkl 与 `Features_competition_train.tar.gz` 内的 .pkl **schema 100% 兼容** (`{'image_feature': ndarray(N, 2048) float32}`), 可直接喂给现有 `prepare_bmn_inputs.py` 流程.

---

## 章程对齐速查 (002 自检)

| 原则 | 落地 |
|------|------|
| **III** 配置驱动 | fps / batch_size / mean / std 全部在 `configs/models/pp_tsm_extractor.yaml`; CLI 选项只是覆盖入口, 不引入新硬编码. |
| **IV** 数据完整性 | clip_id = sha256(file_bytes)[:32], 跨机器一致 (SC-013); manifest.csv 提供完整审计链. |
| **VI** 上游最小侵入 | 不新增 patches; PP-TSM 模型加载完全在本仓库实现 (调用 `paddlevideo.modeling.builder.build_model` 但不改其源码). |
| **VII** 端到端 ≤ 5 命令 | 4 条命令 (curl + export + infer-rawvideo + 可选 build-feature-pkls); 严格符合. |
| **VIII** 隔离 Python | 所有调用都通过 `.venv/bin/`. |

---

## 故障排查

### `pp infer-rawvideo` 报"PP-TSM 训练权重缺失"

执行 stderr 中给出的 curl 命令下载 (~120MB):
```bash
curl -fL -o data/raw/pretrained/ppTSM_k400_dense.pdparams \
    https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams
```

### `scripts/export_pptsm_inference.py` 报 `paddle.jit.to_static` 失败

- 检查 paddle 版本是否 ≥ 2.6 (`.venv/bin/python -c "import paddle; print(paddle.__version__)"`)
- 退化办法: 用上游 `tools/export_model.py` 直接导出 (research.md R10 风险段)
- 报 issue 时附上 paddle.__version__ + 错误 traceback

### 视频是 30 fps, 推理结果时间戳对不上

`pp infer-rawvideo` 默认强制重采样到 25 fps (与 BMN GT 对齐, research.md R11). timeline.json 中的 `start_sec` / `end_sec` 是基于**原视频时间轴**的真实时刻, 不会偏移. 如果你看到偏移, 检查视频元信息中 fps 是否被正确探测 (manifest.csv 中 `fps_original` 字段).

### GPU OOM

降低 batch_size (`--batch-size 16` 或 `8`); 命令会自动二分一次重试 (边界情况已写). T4 (15GB) 默认 batch_size=32 应当够用; 8GB 显存的卡建议 batch_size=16.

---

## 与 v0.2.x 的兼容性

- `Features_competition_train.tar.gz` (43.5GB COS 数据) 仍可直接训练 — **本 feature 不替换它**, 只是**增加了"我自己也能产出兼容的 .pkl"** 的能力.
- 训练命令 `pp train --config configs/models/bmn_pingpong.yaml` 完全不变.
- 评估命令 `pp eval --checkpoint <ckpt> --split val` 完全不变.
- 002 之后, BMN 主线就**完整闭环**: 任意视频 → 训练 / 微调 / 推理 / 可视化.
