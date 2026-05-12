# 数据模型: 基于 PaddleVideo 的乒乓球视频动作识别系统

**功能**: `001-pingpong-action-recognition`
**输入**: spec.md 关键实体 + research.md 决策

本文件定义本项目的逻辑数据模型. 这些不是数据库表, 而是配置文件、列表文件、实验目录、推理产物
共同遵守的字段约定; 所有实体最终落地为文件 / 目录, 由 `src/pingpong_av/` 中的代码读写.

---

## 实体一览

| 实体 | 物理形态 | 章程相关 | 谁创建 | 谁消费 |
|------|----------|----------|--------|--------|
| ActionClass | YAML (类别表) + 整数 ID | III | data-prepare | train / eval / infer |
| VideoClip | 文件 + list 行 | IV | data-prepare | train / eval / infer-clip |
| DatasetSplit | 入库的 list 文件 (`data/splits/*.txt`) | IV | data-prepare | train / eval |
| Experiment | `experiments/<run_id>/` 目录 | II, V | train / eval | 报告 / 后续比较 |
| Model (Checkpoint) | `*.pdparams` + 关联配置引用 | II, VI | train | eval / infer |
| PredictionResult | 内存对象 → JSON 文件 | V (结构) | infer-clip / infer-video | 用户 / 可视化 |
| TimelineSegment | JSON 数组的元素 | — | infer-video (后处理) | 可视化 / 下游 |
| RunManifest | `experiments/<run>/manifest.json` | II | train / eval | 复现性追溯 |

---

## ActionClass

代表一种乒乓球技术动作. **不在源码中硬编码**, 由数据集元信息派生.

**字段**:
- `id` (int, 必填, ≥ 0): 类别整数 ID, 必须从 0 开始连续编号到 N-1
- `name` (str, 必填): 类别规范名 (例: `"forehand_attack"`); 仅含小写字母、数字、下划线
- `display_name` (str, 可选): 用于可视化的展示名 (可中文)
- `description` (str, 可选): 简要说明
- 特殊值 `id = -1` / `name = "unknown"` 保留给长视频滑窗的"未知/背景"类 (FR-014, R5),
  **不**进入训练 list, 仅出现在推理产物中.

**约束**:
- 整个项目内类别集合在一次实验内固定 (spec.md 假设).
- 类别表来自 `configs/datasets/<dataset>.yaml` 的 `classes` 列表, 由 `data-prepare` 校验完整性.

**物理表示** (`configs/datasets/pingpong_public.yaml` 片段):
```yaml
classes:
  - { id: 0, name: serve,           display_name: "发球" }
  - { id: 1, name: forehand_attack, display_name: "正手攻" }
  # ...
```

---

## VideoClip

代表一段被标注的乒乓球动作片段. 物理上是一个视频文件 + list 文件中的一行.

**字段**:
- `clip_id` (str, 必填, 唯一): 全局唯一标识 (例: `match01_0003`)
- `source_video_id` (str, 必填): 源视频 ID, **用于划分时去重** (章程 IV 关键约束)
- `path` (str, 必填): 相对 `data/clips/` 的路径
- `start_sec`, `end_sec` (float, 可选): 在源视频中的起止时间; 若 clip 已物理切分则可空
- `label_id` (int, 必填): 引用 ActionClass.id, 不允许为 -1
- `split` (enum: train | val | test, 必填): 所属划分

**list 文件行格式** (`data/splits/<split>.txt`, 与 PaddleVideo 默认兼容):
```
<relative_path>\t<label_id>
```
扩展元信息 (clip_id, source_video_id 等) 单独写入 `data/splits/<split>.meta.jsonl`,
供 splitter 与评估脚本使用; PaddleVideo 训练入口只读 list 文件.

**校验规则** (由 `tests/unit/test_splitter.py` 强制):
- 任意两个不同 split 中, **不能存在**相同的 `source_video_id` (章程 IV 不可妥协).
- 所有 clip 文件存在且可被视频读取库打开.
- 所有 `label_id` 都在当前 ActionClass 集合内.

---

## DatasetSplit

代表训练 / 验证 / 测试三类样本集合的总称. 物理上是 `data/splits/` 下的 6 个文件:

```
data/splits/
├── train.txt          # PaddleVideo list 格式 (path<TAB>label_id)
├── val.txt
├── test.txt
├── train.meta.jsonl   # 每行一个 VideoClip 的完整 JSON
├── val.meta.jsonl
└── test.meta.jsonl
```

**章程要求**:
- 这 6 个文件**必须入库** (章程 IV); 大文件 (raw / clips) 不入库.
- 重新划分必须递增数据集版本 (在 `configs/datasets/pingpong_public.yaml` 中 bump `split_version`),
  重新划分视为新实验 (spec.md 假设).
- `eval` 入口**只**读 `test.txt`; `train` 入口可读 `train.txt` + `val.txt`, **不得**触及 `test.txt`
  (章程 IV).

---

## Experiment

代表一次完整的训练或评估调用. 物理上是 `experiments/<YYYYMMDD-HHMMSS>-<git_sha7>-<slug>/` 目录.

**目录 schema** (由 `src/pingpong_av/experiment/run_manifest.py` 创建):
```
experiments/20260511-203015-a3f1c8e-pp_tsm_pingpong/
├── manifest.json
├── config.yaml          # 该次运行实际使用的合并后配置 snapshot
├── log/
│   ├── train.log
│   └── eval.log
├── checkpoints/
│   ├── epoch_10.pdparams
│   └── best.pdparams
└── metrics.json         # 仅在 eval 完成后写入
```

**RunManifest 字段** (`manifest.json`):
- `run_id` (str): 目录名
- `kind` (enum: train | eval): 本次运行类型
- `commit` (str, full SHA): 启动时的 git HEAD
- `dirty` (bool): 工作区是否有未提交修改 (true 时启动需 `--allow-dirty`)
- `config_hash` (str): 合并后配置的 SHA256 前 16 位
- `seed` (int): 随机种子
- `python_version` (str): 必须为 `3.11.x`
- `cuda_version` (str | null)
- `gpu_model` (str | null)
- `started_at`, `finished_at` (ISO 8601 UTC)
- `status` (enum: running | succeeded | failed | interrupted)
- `dataset_split_version` (str): 来自 `configs/datasets/*.yaml` 的 `split_version`

**约束** (与章程 II 对齐):
- `commit` + `config_hash` + `seed` + `dataset_split_version` 四元组缺一不可, 缺则拒绝启动.
- 失败的 run 也保留目录 (status=failed), 不允许覆盖已有 run_id.

---

## Model (Checkpoint)

代表一个训练得到的或下载的模型权重. 物理上是 PaddleVideo 标准 `.pdparams` + `.pdopt` 文件,
位于 `experiments/<run_id>/checkpoints/`.

**字段** (由文件命名 + manifest.json 隐式承载):
- `path` (str): `.pdparams` 文件路径
- `experiment_run_id` (str): 引用产生该 checkpoint 的 Experiment
- `epoch` (int) 或 `tag` (str, e.g. "best"): 来源 epoch
- `metrics_at_save` (object, 可选): 保存时刻的验证集 top1/top5 (写入 `manifest.json` 的 `checkpoints` 数组)

**约束**:
- `eval` 与 `infer-*` CLI 通过 `--checkpoint <path>` 接收一个 `.pdparams` 路径; 工具自动从同级
  目录查找该 run 的 `config.yaml` 以正确重建模型结构 (避免人工指定 model 配置时的错配).

---

## PredictionResult

代表对一个**单片段**或**长视频**调用模型后的结果. 物理上是 JSON 文件.

### 单片段结果 (`pp infer-clip` 输出)

```json
{
  "schema": "clip-prediction-v1",
  "input": {
    "video_path": "/abs/path/clip.mp4",
    "duration_sec": 2.4
  },
  "model": {
    "checkpoint": "experiments/<run_id>/checkpoints/best.pdparams",
    "config_hash": "a3f1c8e..."
  },
  "topk": [
    { "id": 1, "name": "forehand_attack", "score": 0.83 },
    { "id": 2, "name": "backhand_push",   "score": 0.11 }
  ],
  "produced_at": "2026-05-11T20:35:12Z"
}
```

字段约束:
- `topk` 长度由 `--topk` 参数决定, 默认 5 (FR-013).
- `score` 必须 ∈ [0, 1] 且大致归一化 (来自 softmax).

### 长视频结果 (`pp infer-video` 输出之 JSON 时间轴)

见 **TimelineSegment**; JSON 是 TimelineSegment 数组 + 顶层 metadata.

---

## TimelineSegment

`pp infer-video` 长视频推理后处理产生的时间轴单元. JSON 数组中的一个元素.

**字段**:
- `start` (float, 必填): 起始时间 (秒, 视频时间轴)
- `end` (float, 必填, > start)
- `label` (str, 必填): ActionClass.name 或 `"unknown"` (低于阈值时)
- `label_id` (int, 必填): ActionClass.id, `unknown` 对应 -1
- `confidence` (float, 必填, ∈ [0, 1]): 合并区间内的窗口置信度均值
- `n_windows` (int, 必填): 该段合并自多少个滑窗

**长视频 JSON 总文件结构**:
```json
{
  "schema": "video-timeline-v1",
  "input": {
    "video_path": "/abs/path/match.mp4",
    "duration_sec": 423.5,
    "fps": 30.0
  },
  "model": { "checkpoint": "...", "config_hash": "..." },
  "inference_config": {
    "window_sec": 2.0, "stride_sec": 1.0,
    "conf_threshold": 0.5, "merge_gap_sec": 1.0
  },
  "segments": [
    { "start": 0.0,  "end": 3.0,  "label": "serve",           "label_id": 0, "confidence": 0.78, "n_windows": 3 },
    { "start": 3.0,  "end": 5.0,  "label": "unknown",         "label_id": -1, "confidence": 0.32, "n_windows": 2 },
    { "start": 5.0,  "end": 9.0,  "label": "forehand_attack", "label_id": 1, "confidence": 0.85, "n_windows": 4 }
  ],
  "produced_at": "2026-05-11T21:00:07Z"
}
```

**约束** (与 FR-014, FR-015 对齐):
- `segments[i].end == segments[i+1].start` (无空隙, 也无重叠).
- `confidence` 是该段所有底层窗口的均值, 不是某一个窗口.
- 与之**配对的** MP4 文件的命名约定: 在 JSON 同目录下, 同名 + `.viz.mp4` 后缀.

---

## 数据对象之间的关系

```
ActionClass (N)  ──标签──>  VideoClip (M)
                                │
                                │ split字段
                                ▼
                          DatasetSplit (3)
                                │
                                │ train.txt 输入
                                ▼
                          Experiment(kind=train) ── 产出 ──> Checkpoint
                                                                │
                                                                │ --checkpoint
                                                                ▼
                                                      Experiment(kind=eval) ─> metrics.json
                                                                │
                                                                ├──> PredictionResult (单片段)
                                                                │
                                                                └──> TimelineSegment[] (长视频)
                                                                            │
                                                                            └──> MP4 可视化视频
```

每个 Experiment 通过 `manifest.json` 反向引用所使用的 `dataset_split_version` 与 `config_hash`,
形成可复现实验的闭环 (章程 II).
