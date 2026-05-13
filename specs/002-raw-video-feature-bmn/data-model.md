# 数据模型 (002 增量)

**功能**: 002-raw-video-feature-bmn
**日期**: 2026-05-13
**前置**: 001 data-model.md (ActionClass, VideoClip, DatasetSplit, Experiment, Model, PredictionResult, TimelineSegment 已存在)

本文件**只**记录 002 feature 引入的新实体或对现有实体的字段扩展. 与 001 重复的内容不重复.

---

## RawVideo (新增)

用户拿到的、本仓库还未抽过特征的原始视频文件.

### 字段

| 字段 | 类型 | 必填 | 描述 | 章程对齐 |
|------|------|------|------|----------|
| `path` | str (abs) | 是 | 视频文件绝对路径; 必须本地可读 | — |
| `clip_id` | str (32-hex) | 是 | sha256(file_bytes)[:32] (流式整文件 hash) | IV (跨机器一致性, SC-013) |
| `fps_original` | float | 是 | 视频原 fps (从 ffmpeg `-i` 探测) | — |
| `fps_used` | int | 是 | 抽帧时强制重采样后的 fps; 默认 25 (与 BMN GT 一致) | III (yaml 默认值, 不硬编码) |
| `duration_sec` | float | 是 | 视频时长 = max(eof, n_frames / fps_used) | — |
| `n_frames` | int | 是 | 抽帧后实际帧数 (= 进入 PP-TSM 的帧数) | — |
| `container_format` | str | 否 | mp4 / avi / mov / flv / mkv (ffmpeg 探测) | — |

### 约束

- `clip_id` 必须从文件**全部字节**计算, 不基于文件名/路径 (FR-034 幂等性).
- `clip_id` 长度严格为 32 个 16 进制字符 (与现有 `Features_competition_train/<32-hex>.pkl` 内部布局一致, 便于无缝替换).
- 任何新拍视频与上游公开数据集中的视频 hash **不会冲突** (sha256 抗碰撞), 因此 RawVideo 集合可以**自由合并**, 不需要去重命名.
- 以下情况视为同一视频: 字节级一致 (即使文件名不同). 以下情况视为不同视频: 任何重压缩 / 剪辑 / 帧率转换后保存.

### 关系

- `1:1 → ImageFeaturePkl` (一个 RawVideo 经 PP-TSM 抽特征产生一个 .pkl)
- `1:N → TimelineSegment` (一个 RawVideo 推理后产生 N 个时间区间; 经 BMN)

---

## ImageFeaturePkl (新增, 与上游 `Features_competition_train.tar.gz` 内部 .pkl schema 100% 兼容)

PP-TSM 抽出的视频级特征序列, pickle 文件.

### 字段 (pickle 文件内部)

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `image_feature` | ndarray(N, 2048) float32 | 是 | N 帧的 ResNet50 全局池化特征 |
| `audio_feature` | None / ndarray | 否 | TableTennis 不用; 兼容 FootballAction |
| `pcm_feature` | None / ndarray | 否 | 同上 |

### 字段 (manifest.csv 中的元数据, 不在 pickle 内)

见 research.md R12 表; 关键字段:
- `pkl_sha256`: pickle 文件本身的 sha256 (用于审计 + 去重)
- `pp_tsm_weight_sha256`: 抽特征所用训练权重 sha256
- `pp_tsm_inference_sha256`: 派生的 inference 双文件 sha256
- `pp_tsm_config_hash`: 业务 yaml config_hash (16-hex)
- `extraction_commit`: 命令 git rev-parse HEAD

### 约束

- `image_feature.shape == (N, 2048)`, dtype float32, 不接受 float16 / float64.
- `N == round(duration_sec × fps_used)`, 容差 ±5% (ffmpeg 在某些容器上偏差).
- pickle 协议必须是 `pickle.HIGHEST_PROTOCOL` (与上游一致).

### 关系

- `1:1 ← RawVideo` (反向)
- `1:N → BmnSlidingWindowSlice` (经过 prepare_bmn_inputs.py 切成 8s 滑窗 .npy)

---

## PPTSMTrainWeight (新增)

BCEBOS 公开下载的 PP-TSM 训练权重 (动态图 state_dict).

### 字段

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `path` | str (abs) | 是 | 默认 `data/raw/pretrained/ppTSM_k400_dense.pdparams` |
| `sha256` | str (64-hex) | 是 | 文件 sha256, 校验下载完整性 |
| `source_url` | str | 是 | `https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams` |
| `expected_sha256` | str (64-hex) | 否 | 预期 sha256 (写入 yaml 时填; 缺失时跳过校验) |
| `downloaded_at` | str (ISO8601) | 否 | 下载完成时间 |
| `size_bytes` | int | 是 | 约 120 MB (用于检查下载是否完整) |

### 约束

- 缺失时 `pp extract-feat` 必须退出码 1 + stderr 给出 `curl -fL -o <path> <url>` 一行 (FR-038).
- 如 yaml 中提供 `expected_sha256`, 实际 sha256 不匹配则退出码 2 (环境问题).
- **不入库**: 路径在 `.gitignore` (`data/raw/pretrained/` 已通过 `data/raw/**` 默认忽略).

### 关系

- `1:1 → PPTSMInferenceModel` (经 export_pptsm_inference.py 派生)

---

## PPTSMInferenceModel (新增)

由 `scripts/export_pptsm_inference.py` 从 PPTSMTrainWeight 派生的静态图 inference 双文件.

### 字段

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `pdmodel_path` | str (abs) | 是 | 默认 `data/raw/pretrained/ppTSM.pdmodel` |
| `pdiparams_path` | str (abs) | 是 | 默认 `data/raw/pretrained/ppTSM.pdiparams` |
| `pdmodel_sha256` | str (64-hex) | 是 | 静态图结构文件 sha256 |
| `pdiparams_sha256` | str (64-hex) | 是 | 静态图权重文件 sha256 |
| `combined_sha256` | str (64-hex) | 是 | sha256(pdmodel_bytes + pdiparams_bytes) — 一对的 ID |
| `derived_from_train_weight_sha256` | str (64-hex) | 是 | 来源训练权重的 sha256 (审计链) |
| `paddle_version` | str | 是 | 导出时的 paddle 版本 (例如 "2.6.2") |
| `exported_at` | str (ISO8601) | 是 | 导出完成时间 |
| `marker_path` | str (abs) | 是 | `data/raw/pretrained/.export_marker.json` 元数据落点 |

### 约束

- `paddle.jit.save` 是确定性的: 相同输入权重 + 相同 paddle 版本 → 相同输出文件 (假设没有非确定性 op).
- 不一致时 (例如 `derived_from_train_weight_sha256` 与当前 PPTSMTrainWeight.sha256 不匹配) 自动重新导出.
- `paddle_version` 不一致时 (例如用户升级 paddle) 自动重新导出.
- **不入库**: 与训练权重一致, 全在 `data/raw/pretrained/`.

### 关系

- `1:1 ← PPTSMTrainWeight` (反向)
- `1:N → ImageFeaturePkl` (一个 inference 模型用于抽 N 个视频特征)

---

## RawVideoTimelineResult (新增, US1 输出 schema)

`pp infer-rawvideo` 的端到端输出 — `timeline.json` 文件内容.

### 字段 (顶层)

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `schema` | str | 是 | `"rawvideo-timeline-v1"` |
| `input_video` | str (abs) | 是 | 输入 mp4 路径 |
| `input_video_clip_id` | str (32-hex) | 是 | 与 RawVideo.clip_id 一致 |
| `input_video_duration_sec` | float | 是 | 视频时长 |
| `extraction` | object | 是 | 抽特征环节元信息 (见下) |
| `bmn_inference` | object | 是 | BMN 推理环节元信息 (见下) |
| `command_version` | str | 是 | git rev-parse HEAD (= 命令版本) |
| `produced_at` | str (ISO8601) | 是 | UTC 时间 |
| `results` | array<TimelineSegment> | 是 | 排序后的候选时间区间列表 |

### `extraction` 子对象

```json
{
  "fps_original": 30.0,
  "fps_used": 25,
  "n_frames": 7810,
  "pp_tsm_weight_sha256": "<64-hex>",
  "pp_tsm_inference_sha256": "<64-hex>",
  "pp_tsm_config_hash": "<16-hex>",
  "feature_pkl_path": "<out>/feature.pkl"
}
```

### `bmn_inference` 子对象

```json
{
  "checkpoint": "experiments/<run>/BMN_epoch_00020.pdparams",
  "checkpoint_sha256": "<64-hex>",
  "subset": "validation",
  "ar_at": null
}
```

`ar_at` 始终为 `null` (推理时无 GT).

### `results[i]` (扩展现有 TimelineSegment)

| 字段 | 类型 | 必填 | 描述 | 与 TimelineSegment 关系 |
|------|------|------|------|------------------------|
| `start_sec` | float | 是 | 区间起点 (秒) | 同 |
| `end_sec` | float | 是 | 区间终点 (秒) | 同 |
| `label_id` | int | 是 | 0..13 | 同 |
| `label_name` | str | 是 | 中文 (摆短/拉/...) | 同 |
| `score` | float | 是 | BMN proposal 置信度 [0, 1] | 同 |
| `rank_in_window` | int | 是 (新) | 该区间在所属 8s 窗口内的排名 (0 = 最高分) | **新字段** |

### 约束

- `results` 必须按 `start_sec` 升序; 同 `start_sec` 时按 `score` 降序.
- `score ∈ [0, 1]`; BMN 上游已 sigmoid.
- `rank_in_window` 是新字段, 用于让用户做"每 8 秒只取 top-1" 的过滤而不重新跑 BMN.

### 关系

- `1:1 ← RawVideo` (反向, 一个视频一个 timeline.json)
- `1:N → TimelineSegment` (扩展 001 中已有的 TimelineSegment, 增加 `rank_in_window` 字段)

---

## 数据对象关系图 (002 增量)

```
RawVideo  (sha256(file_bytes)[:32] = clip_id)
  │
  │ pp extract-feat / pp build-feature-pkls
  │   ← 用 PPTSMTrainWeight + PPTSMInferenceModel
  ▼
ImageFeaturePkl  ({'image_feature': (N, 2048)})
  │
  │ scripts/prepare_bmn_inputs.py (新拆: prepare_bmn_inputs_for_inference)
  ▼
BmnSlidingWindowSlice  (8s × 25fps = 200 帧 × 2048-d, .npy)
  │
  │ pp infer-rawvideo / pp eval (走 _run_bmn_eval, gt_required=False)
  │   ← 用 BMN ckpt
  ▼
RawVideoTimelineResult  (timeline.json, schema=rawvideo-timeline-v1)
  │
  │ inference/visualize.py (US3 已有)
  ▼
visualized.mp4  (在原视频上叠加候选区间文本)
```

外侧 (训练分支, 与 v0.2.x 衔接):

```
RawVideo + GT JSON (label_cls14_<name>.json)
  │
  │ pp build-feature-pkls (--gt-json)
  ▼
ImageFeaturePkl + 重写后的 GT JSON (url 字段已替换为 clip_id.mp4)
  │
  │ scripts/prepare_bmn_inputs.py (现有, 训练流程)
  ▼
data/bmn_inputs/<custom>/feature/*.npy + label_fixed.json
  │
  │ pp train --resume <baseline> --override dataset.bmn_inputs_dir=...
  ▼
微调后的 BMN ckpt (实验目录新 manifest)
```

---

## 与 001 data-model.md 的关系

- **不修改**: ActionClass, VideoClip, DatasetSplit, Experiment, Model, PredictionResult.
- **扩展**: TimelineSegment 增加 `rank_in_window` 字段 (向后兼容: 旧 schema reader 默认值 0 即可).
- **新增**: RawVideo, ImageFeaturePkl, PPTSMTrainWeight, PPTSMInferenceModel, RawVideoTimelineResult.
