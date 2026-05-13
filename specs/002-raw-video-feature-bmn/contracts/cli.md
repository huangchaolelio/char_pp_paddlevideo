# CLI 契约 (002 增量)

**功能**: 002-raw-video-feature-bmn
**日期**: 2026-05-13
**前置**: 001 contracts/cli.md (`pp env-check / data-prepare / train / eval / infer-clip / infer-video / infer-pkl` 已存在)

本文件**只**记录 002 feature 引入的新命令. 与 001 的契约 (返回码、`--allow-dirty` 语义、JSON stdout 格式等) 保持一致, 不重复定义.

---

## 命令清单 (002 新增)

| 命令 / 脚本 | 用途 | 用户故事 | 引入 FR |
|------------|------|---------|--------|
| `scripts/export_pptsm_inference.py` | 训练权重 → inference 模型 (一次性) | US1/US2 前置 | FR-038a |
| `pp extract-feat` | 单视频 → 单 .pkl | US1 / US2 子步骤 | FR-033, FR-035, FR-037, FR-038 |
| `pp build-feature-pkls` | 视频目录 → .pkl 集合 + (可选) 重写 GT JSON | US2 | FR-034, FR-043, FR-044 |
| `pp infer-rawvideo` | 端到端: mp4 → timeline.json + 可视化 mp4 | US1 | FR-039, FR-040, FR-041, FR-042 |

---

## `scripts/export_pptsm_inference.py`

**目的**: 把 BCEBOS 公开的 PP-TSM 训练权重 (`ppTSM_k400_dense.pdparams`, 动态图 state_dict) 转换为
上游 `extract_feat.py` 期望的 inference 双文件 (`ppTSM.pdmodel + ppTSM.pdiparams`, 静态图).

**用法**:
```
python scripts/export_pptsm_inference.py \
    [--src data/raw/pretrained/ppTSM_k400_dense.pdparams] \
    [--out-dir data/raw/pretrained/] \
    [--config configs/models/pp_tsm_extractor.yaml] \
    [--force]
```

**参数**:
- `--src`: 源训练权重路径 (默认从 yaml 读, 见 FR-036).
- `--out-dir`: 输出目录, 写入 `ppTSM.pdmodel + ppTSM.pdiparams + .export_marker.json`.
- `--config`: 业务 yaml, 决定网络结构参数 (`seg_num=8, seglen=1, pretrained=null` 等).
- `--force`: 即使 marker 已存在也重导出.

**幂等性**:
- 检查 `<out-dir>/.export_marker.json`, 若 `derived_from_train_weight_sha256` + `paddle_version` 都匹配, 默认跳过 (返回 0, 输出 "skipped").
- `--force` 跳过此检查.

**退出码**:
- `0` 成功 (含跳过)
- `1` 用户输入错 (`--src` 不存在 / `--config` schema 错)
- `2` 环境问题 (paddle 不可导 / `paddle.jit.to_static` 失败)
- `3` 章程硬约束违反 (例如 yaml 中 `seg_num != 8` — 与上游 inference 接口不兼容)
- `4` 运行时失败 (磁盘空间不足等)

**stdout 输出 (JSON)**:
```json
{
  "schema": "pp-tsm-export-v1",
  "src_train_weight": "/abs/path/ppTSM_k400_dense.pdparams",
  "src_train_weight_sha256": "<64-hex>",
  "out_pdmodel": "/abs/path/ppTSM.pdmodel",
  "out_pdiparams": "/abs/path/ppTSM.pdiparams",
  "combined_sha256": "<64-hex>",
  "paddle_version": "2.6.2",
  "exported_at": "2026-05-13T14:30:00Z",
  "skipped_reason": null  // 或 "marker matches" 当跳过时
}
```

**章程约束**:
- VI (上游最小侵入): 该脚本**不**修改 PaddleVideo submodule, 完全在本仓库实现 (用 `paddlevideo.modeling.builder` 构造网络, 但不改其源码).
- VIII (隔离 3.11): 通过 `.venv/bin/python` 调用.

---

## `pp extract-feat`

**目的**: 接受单个视频文件, 输出一个与上游 `Features_competition_train.tar.gz` 内部 .pkl schema 100% 兼容的特征 pkl.

**用法**:
```
pp extract-feat --input <video>.mp4
                [--output <video>.pkl]
                [--fps 25]
                [--batch-size 32]
                [--config configs/models/pp_tsm_extractor.yaml]
                [--allow-dirty]
                [--keep-frames]
```

**参数**:
- `--input` (必填): 视频文件路径 (mp4/avi/mov/flv/mkv 等任意 ffmpeg 可解格式).
- `--output`: 输出 .pkl 路径; 默认 `<视频同目录>/<sha256(bytes)[:32]>.pkl` (注意: 用内容 hash 命名, 不是源文件名 stem).
- `--fps`: 强制抽帧 fps; 默认从 yaml 读 (`extraction.fps=25`, 章程 III).
- `--batch-size`: PP-TSM forward batch_size; 默认从 yaml 读 (`extraction.batch_size=32`).
- `--config`: 业务 yaml, 决定 PP-TSM 网络结构 + ImageNet mean/std + resize 策略.
- `--allow-dirty`: git 工作区脏时仍允许运行 (与 `pp train` 一致).
- `--keep-frames`: 保留 ffmpeg 抽帧的临时目录 (默认在结束时清理).

**前置依赖** (按调用顺序检查):
1. ffmpeg 命令存在 (`which ffmpeg` 返回 0); 否则退出码 2 + stderr 给出 apt/conda 安装指引.
2. `data/raw/pretrained/ppTSM_k400_dense.pdparams` 存在; 否则退出码 1 + stderr 给出 curl 一行.
3. `data/raw/pretrained/ppTSM.{pdmodel,pdiparams}` 存在 (FR-038a); 否则**自动调用** `scripts/export_pptsm_inference.py` (一次性, ~30s).
4. CUDA 可用 (CPU 模式仅作 env-check 回退); 否则退出码 2.

**输出**:
- `<output>.pkl`: pickle 文件, 内容 `{'image_feature': ndarray(N, 2048) float32}`.
- 在 `<output>.meta.json` (同目录, 同名 + .meta.json 后缀) 写入元数据 (R12 manifest.csv 的所有字段, 单视频版).

**退出码**: 0 / 1 / 2 / 3 / 4 (与 001 一致).

**stdout (JSON)**:
```json
{
  "schema": "extract-feat-v1",
  "input_video": "/abs/path/input.mp4",
  "clip_id": "<32-hex>",
  "n_frames": 7810,
  "fps_used": 25,
  "duration_sec": 312.4,
  "output_pkl": "/abs/path/<32-hex>.pkl",
  "pkl_sha256": "<64-hex>",
  "elapsed_sec": 38.2,
  "fps_throughput": 204.5
}
```

**章程约束**:
- III: fps / batch_size / mean / std 全部在 yaml, 不在源码硬编码.
- VII: 该命令是 quickstart 的 5 条之一 (替换 v0.2.x 中需要从 COS 拉特征那一步).

---

## `pp build-feature-pkls`

**目的**: 批量把视频目录转为 .pkl 集合 + (可选) 重写 GT JSON, 用于 BMN 训练数据扩充 (US2).

**用法**:
```
pp build-feature-pkls --videos-dir <dir>
                      --output-dir <out>/Features_<name>/
                      [--gt-json <input_label.json>]
                      [--name <name>]
                      [--workers 1]
                      [--allow-dirty]
                      [--force]
```

**参数**:
- `--videos-dir` (必填): 含 mp4 文件的目录 (递归扫描).
- `--output-dir` (必填): 输出根目录; 命令内部会建 `<output-dir>/Features_<name>/` 子目录与上游 tar.gz 一致.
- `--gt-json`: 可选 GT JSON (按 `label_cls14_train.json` schema). 若提供, 系统验证每个 url 在 videos-dir 中存在, 然后写 `<output-dir>/label_cls14_<name>.json` 把 url 字段替换为 `<clip_id>.mp4`.
- `--name`: 数据集子集名 (用于命名输出目录与 label JSON); 默认从 `--videos-dir` 的 basename 派生.
- `--workers`: 并发抽特征 workers 数; 默认 1 (避免单卡 OOM). 推荐 ≤ GPU 数.
- `--force`: 忽略已有 .pkl, 全部重抽.

**幂等性** (FR-034):
- 默认对每个视频先算 `clip_id = sha256(file_bytes)[:32]`, 检查 `<output-dir>/Features_<name>/<clip_id>.pkl` 是否存在; 存在则跳过 (manifest.csv 中 `error` 列写 "skipped (already exists)").
- 若同时提供 `--gt-json`, 即使所有 .pkl 都已存在, 仍会重新写 `label_cls14_<name>.json` (因为可能 GT JSON 内容变了).

**输出**:
- `<output-dir>/Features_<name>/<clip_id_1>.pkl, <clip_id_2>.pkl, ...`
- `<output-dir>/manifest.csv` (R12 表中所有列)
- `<output-dir>/label_cls14_<name>.json` (仅当 `--gt-json` 提供时)

**校验阶段** (在跑 GPU 前一次性做):
- 若提供 `--gt-json`: 解析 JSON, 列出所有 url, 在 `--videos-dir` 中查找 `url == basename(file_path)` 或 `url.stem == basename(file_path).stem`. 任何缺失立即抛错 (退出码 1) 列出缺失项, **不抽特征**.
- 检查磁盘空间: 估算输出大小 (每视频 ~50MB pkl), 不足则退出码 2.

**退出码** (FR-047):
- `0` 全部成功 (含跳过)
- `1` 用户输入错 (videos-dir 不存在 / gt-json 引用了不存在的视频)
- `2` 环境问题 (ffmpeg 缺失 / 权重缺失 / 磁盘满)
- `3` 章程硬约束违反 (例如标签数据缺失而又要求训练 — 但本命令不训练, 不会触发此码)
- `4` 运行时失败 (中间 OOM 等)

**stdout (JSON)**:
```json
{
  "schema": "build-feature-pkls-v1",
  "videos_dir": "/abs/path/videos/",
  "output_dir": "/abs/path/out/Features_mybatch/",
  "name": "mybatch",
  "n_videos_total": 100,
  "n_videos_processed": 95,
  "n_videos_skipped": 3,
  "n_videos_failed": 2,
  "manifest_path": "/abs/path/out/manifest.csv",
  "label_json_path": "/abs/path/out/label_cls14_mybatch.json",
  "elapsed_sec": 2742.0,
  "fps_throughput_avg": 198.3
}
```

**章程约束**:
- IV: 不参与 split 划分 (本命令产出 .pkl, 由 `pp data-prepare` 后续做 split, 章程 IV 责任在那里).
- III: 全部超参在 yaml.

---

## `pp infer-rawvideo`

**目的**: 端到端原始视频推理 (US1) — `mp4 → timeline.json + 可视化 mp4`.

**用法**:
```
pp infer-rawvideo --input <video>.mp4
                  --bmn-checkpoint <ckpt>.pdparams
                  --output-dir <out>/
                  [--threshold 0.5]
                  [--min-duration 0.3]
                  [--allow-dirty]
                  [--keep-frames]
                  [--keep-features]
                  [--no-visualize]
```

**参数**:
- `--input` (必填): 视频路径.
- `--bmn-checkpoint` (必填): BMN .pdparams 路径 (本仓库 v0.2.x 训练产物).
- `--output-dir` (必填): 输出根目录, 命令会建.
- `--threshold`: BMN proposal 置信度过滤阈值; 默认从 yaml 读 (FR-014 配置的 `inference.threshold=0.5`).
- `--min-duration`: 最小区间时长秒, 过滤掉太短的候选; 默认 0.3.
- `--keep-frames`: 保留 ffmpeg 抽帧临时目录 (FR-041).
- `--keep-features`: 保留中间产物 `<out>/feature.pkl` (默认 ON).
- `--no-visualize`: 跳过可视化 mp4 渲染 (调试用).

**内部流水线**:
1. `pp extract-feat` 等价: 视频 → `<out>/feature.pkl`.
2. `prepare_bmn_inputs.prepare_bmn_inputs_for_inference(feature.pkl, <out>/bmn_input/)`: 切 8s 滑窗 → `<out>/bmn_input/feature/*.npy + label_fixed.json` (空 GT).
3. `run_upstream_bmn_eval(reuse_existing=False, gt_required=False)`: BMN 前向 + post-processing → `<out>/bmn_eval/results/bmn_results_validation.json`.
4. 解析 ActivityNet 结果 → 写 `<out>/timeline.json` (R12 schema, `rawvideo-timeline-v1`).
5. (可选) `inference/visualize.py`: 在原视频上叠加候选区间文本, 写 `<out>/<input_name>_visualized.mp4`.

**输出 (FR-040)**:
- `<out>/timeline.json` — 主产物, RawVideoTimelineResult schema.
- `<out>/<input_name>_visualized.mp4` — 可视化产物.
- `<out>/feature.pkl` — 中间产物 (默认保留, `--no-keep-features` 关).
- `<out>/bmn_input/` — 中间产物 (8s 滑窗 .npy + label_fixed.json).
- `<out>/bmn_eval/` — 中间产物 (BMNMetric 写入).
- `<out>/run.log` — 全流程日志.

**退出码**: 0 / 1 / 2 / 3 / 4 (与 001 一致).

**stdout (JSON)**:
```json
{
  "schema": "infer-rawvideo-v1",
  "input_video": "/abs/path/input.mp4",
  "clip_id": "<32-hex>",
  "duration_sec": 312.4,
  "n_proposals": 47,
  "n_proposals_after_filter": 23,
  "timeline_json": "/abs/path/out/timeline.json",
  "visualized_mp4": "/abs/path/out/input_visualized.mp4",
  "elapsed_sec": 187.3,
  "elapsed_breakdown": {
    "extract_feat": 38.2,
    "bmn_forward": 102.1,
    "bmn_postprocess": 38.5,
    "visualize": 8.5
  }
}
```

**章程约束**:
- III: 阈值 / window / stride 全部在 yaml.
- IV: **不**消费 splits (推理时无 GT, 也不写入 splits).
- V: **不**输出 top1/top5/AR@AN (推理时无 GT). 输出 `n_proposals` 即可.
- VII: 该命令是 quickstart 的 1 条 (= "用户拿到本项目第一件事").

---

## 与 001 cli.md 的关系

- **不修改**: env-check / data-prepare / train / eval / infer-clip / infer-video / infer-pkl 的契约.
- **扩展**: `pp eval` 已支持 BMN (v0.2.1); 002 不再扩展.
- **新增**: 上述 4 个 (1 脚本 + 3 cli).

## 与 quickstart 的关系

002 之后 quickstart 升级为:

```bash
# 1) 环境自检
.venv/bin/pp env-check --strict

# 2) 一次性下载 PP-TSM 训练权重 (~120MB)
mkdir -p data/raw/pretrained
curl -fL -o data/raw/pretrained/ppTSM_k400_dense.pdparams \
    https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams

# 3) 一次性导出 inference 模型 (~30s, 自动)
.venv/bin/python scripts/export_pptsm_inference.py

# 4) 端到端推理: 任意 mp4 → timeline.json + 可视化 mp4
.venv/bin/pp infer-rawvideo \
    --input <your_video>.mp4 \
    --bmn-checkpoint experiments/<run>/BMN_epoch_00020.pdparams \
    --output-dir outputs/<run_name>/
```

**4 条命令完成端到端原始视频推理** (含权重下载与一次性 inference 模型导出), 严格符合章程 VII (≤ 5 条).
