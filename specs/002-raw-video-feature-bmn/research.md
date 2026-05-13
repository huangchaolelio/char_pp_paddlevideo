# 阶段 0 研究: 原始视频到 BMN 时序定位的端到端适配 (002)

**功能**: 002-raw-video-feature-bmn
**日期**: 2026-05-13
**前置**: spec.md (已完成 specify + clarify)

本文件**只**记录 002 feature 引入的新研究项. 已在 001 research.md 中解决的问题 (R1-R9) 不重复.

---

## R10 — PP-TSM 训练权重 → inference 模型 (`.pdmodel + .pdiparams`) 转换

### 背景

上游 `applications/FootballAction/extractor/extract_feat.py` 使用 `paddle.inference.Predictor` 加载
**inference 双文件** (`ppTSM.pdmodel` 静态图结构 + `ppTSM.pdiparams` 静态图权重) 来抽特征. 这是
PaddlePaddle "动态图训练 + 静态图推理" 标准流水线.

但 BCEBOS 公开下载的是 **训练权重** `ppTSM_k400_dense.pdparams`, 即动态图 `state_dict`. 两者不能直接互换.

### 决策 (Session 2026-05-13 clarify Q2 已固化)

写 `scripts/export_pptsm_inference.py`, 首次运行 `pp extract-feat` 时**自动**:
1. 检测 inference 双文件是否存在于 `data/raw/pretrained/ppTSM.{pdmodel,pdiparams}`
2. 不存在则: (a) 检查 `ppTSM_k400_dense.pdparams` 是否存在 (FR-038); (b) 不存在则抛出退出码 1 + 提示 curl 下载
3. 加载训练权重到动态图 `paddle.vision.models.resnet50` 风格 PP-TSM 结构
4. `paddle.jit.to_static` 包装成静态图
5. `paddle.jit.save(model, "data/raw/pretrained/ppTSM", input_spec=[InputSpec(shape=[None, 8*1*3, 224, 224], dtype='float32')])`
6. 第二次运行 `pp extract-feat` 直接复用缓存

### 实操要点

- **PP-TSM 网络结构定义**: 上游 `paddlevideo/modeling/backbones/resnet_tsn.py` (ResNet50 + TSM module).
  导出时需要剥掉分类头 (head), 或者保留但取倒数第二个 output. **更简洁的做法**: 用 `applications/TableTennis/extractor/configs/configs.yaml` 描述的
  推理配置直接通过 `paddlevideo.modeling.builder.build_model(cfg.MODEL)` 构造, 然后注入 .pdparams,
  再 `paddle.jit.to_static + save`.
- **InputSpec 形状**: `[batch_size, num_seg * seg_len * 3, 224, 224]` = `[B, 24, 224, 224]` (上游
  `seg_num=8 / seglen=1`, 通道维拼接). 让 batch 维设 `None` 以支持动态批量.
- **取第二个输出**: 模型 forward 默认返回 logits. 抽特征时需要 `output_names[1]`. 实现办法: 在
  `to_static` 之前重写 forward 让它返回 `(logits, pool_feat)` 元组, 这样静态图导出时两个都被记录, 推理
  时手动选 `output_handle = predictor.get_output_handle(output_names[1])`.
- **CUDA / CPU 兼容**: `paddle.jit.save` 与硬件无关, 同一份 .pdmodel/.pdiparams 在 CPU/GPU 都可加载.
  但**导出时**显存峰值约 1GB; 用户 GPU < 1GB free 时退化到 CPU 导出 (慢一点, 一次性).
- **幂等性**: `data/raw/pretrained/.export_marker.json` 记录 `{src_sha256, paddle_version, exported_at}`,
  下次启动时比对; 如果训练权重 sha256 变了 (用户换了 ckpt) 就重新导出.

### 替代方案 (拒绝原因)

- **让用户手动转**: 违反章程 VII (端到端 ≤ 5 条命令).
- **从上游 BCEBOS 找已导出版**: 风险: BCEBOS 上 `PaddleVideo-release2.1/PPTSM/inference/` 目录可能不存在或 URL 不稳定; 实地测试 (curl HEAD) 发现 `https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams` 是 200 OK, 但 `ppTSM.pdmodel` 同目录为 404. 自己导出更可控.
- **走动态图直接抽特征 (跳过 inference 模型)**: 慢 (动态图前向比静态图慢 1.5-2x), 与上游模式不一致, 与 SC-013 一致性目标冲突.

### 风险与缓解

- **风险**: `paddle.jit.to_static` 对某些 layer (如自定义 TSM shift module) 转换失败.
  **缓解**: 上游 PaddleVideo 团队已对 PP-TSM 做过 to_static 测试 (有 `tools/export_model.py` 脚本作证).
  如果仍失败, 退化到从上游 `tools/export_model.py` 直接调用 (它已经验证可用), 实质上把工作委托给上游.
- **风险**: 不同 paddle 版本导出的 inference 文件不能互换.
  **缓解**: marker.json 记录 `paddle_version`; 不一致时提示用户重新导出.

### 输出 → 阶段 1

- 在 `data-model.md` 增加 **PPTSMInferenceModel** 实体 (派生自 PPTSMExtractorWeights)
- 在 `contracts/cli.md` 列 `scripts/export_pptsm_inference.py` 的 CLI 接口 (即使是脚本不是 `pp` 子命令)
- 在 `tasks.md` 增加 T201/T202: 实现 export 脚本 + 测试

---

## R11 — ffmpeg 抽帧策略与 fps 一致性

### 背景

spec.md FR-035 要求严格对齐上游: ffmpeg `-r 25 -q 0 %08d.jpg`. 但用户输入视频可能是 30 fps / 60 fps / 24 fps,
直接 `-r 25` 会做帧重采样, 改变实际帧数与时长比.

### 决策

**两阶段抽帧** (与上游 FootballAction `extract_feat.py` 一致):
1. **第 1 阶段** (本 feature 新增): 不指定 `-r`, 用原始 fps 抽全部帧 → `frames/%08d.jpg`. 这样保留完整信息.
2. **第 2 阶段** (内部): 喂给 PP-TSM 时按"每 8 帧采 1 帧" (上游 `seg_num=8, seglen=1` 的语义) 的方式
   构成 batch. 这一步**不改动帧序列本身**, 只是采样.
3. **fps 标记**: 在 manifest.csv 中记录 `fps_original` (从 ffmpeg `-i` 读) + `fps_used_for_extraction = fps_original`. label JSON
   中的时间戳是基于**原始视频 fps**, 因此用户传入的 GT JSON `fps: 30` 时系统会按 30 fps 解析时间戳, 不强制重采样到 25.

### 与上游 BMN GT 不一致风险

上游 AI Studio #127 数据集 GT 用 fps=25. 如果用户视频是 30 fps + GT 用 30 fps, 直接喂 BMN 训练时, BMN 内部
`tscale=200, dscale=200` 是基于 8 秒 × 25 帧 = 200 帧的假设; 30 fps × 8 秒 = 240 帧, **超出 tscale 上限**.

**缓解**:
- **训练时**: prepare_bmn_inputs.py 已经把视频按 8s 滑窗切成 .npy, 切片时按 GT 中的 fps 采样到 200 帧 (重采样到 25 fps 等价). 这一步在 v0.2.x 已存在, 002 feature 不需要改.
- **推理时** (US1 `pp infer-rawvideo`): 视频如果是 30 fps, 抽特征时按 30 fps 抽, 然后 prepare_bmn_inputs.py 等价路径会把 8s 窗口内的 240 帧重采样到 200 帧. 或者**第 1 阶段就用 ffmpeg `-r 25` 强制 25 fps**, 一劳永逸.

**最终决策**: **强制 `-r 25`**, 与上游 GT 同 fps. 边界情况段落已写"系统应日志告知用户做了重采样, 不报错". 用户 GT JSON 的 fps 字段会被规范化到 25.

### 输出 → 阶段 1

- 在 `data-model.md` 的 **RawVideo** 字段增加 `fps_resampled = 25`
- 在 `quickstart.md` 增加 "如何为 30 fps 手机视频跑推理" 一节, 提示 fps 重采样不会丢动作信息

---

## R12 — manifest.csv 与 timeline.json 元信息字段表

### 背景

spec.md FR-048 要求"透传至少一份完整的产出元信息: PP-TSM sha256 / config_hash / fps / commit", 但具体字段未列.
延后到 plan 阶段定 (clarify Q5 已推迟).

### 决策

#### `manifest.csv` (US2 `pp build-feature-pkls` 输出)

| 列名 | 类型 | 描述 | 来源 |
|------|------|------|------|
| `video_path` | str | 输入 mp4 绝对路径 | CLI 参数 |
| `clip_id` | str (32-hex) | sha256(file_bytes)[:32] | 流式 hash |
| `n_frames` | int | 抽出特征向量数 (= image_feature.shape[0]) | PP-TSM 抽出后实测 |
| `fps_original` | float | 原视频 fps (ffmpeg `-i` 探测) | ffmpeg |
| `fps_used` | int | 强制重采样后 fps (= 25 默认) | 命令参数 |
| `pkl_path` | str | 写出的 .pkl 绝对路径 | 命令计算 |
| `pkl_sha256` | str | pickle 文件本身的 sha256 (校验下游消费时未篡改) | 写后计算 |
| `pp_tsm_weight_sha256` | str | `ppTSM_k400_dense.pdparams` 的 sha256 (= 训练权重身份证) | 一次性算 |
| `pp_tsm_inference_sha256` | str | `ppTSM.pdmodel` + `pdiparams` 拼接 sha256 (= 推理模型身份证) | 一次性算 |
| `pp_tsm_config_hash` | str (16-hex) | 业务 yaml `configs/models/pp_tsm_extractor.yaml` 的 config_hash | utils/config |
| `extraction_commit` | str | git rev-parse HEAD (= 命令版本号) | 命令计算 |
| `extracted_at` | str (ISO8601) | UTC 时间 | 命令计算 |
| `error` | str | 空或错误信息 (例如 "video too short < 8s") | 边界情况 |
| `duration_sec` | float | 视频时长 = n_frames / fps_used | 命令计算 |

#### `timeline.json` (US1 `pp infer-rawvideo` 输出)

ActivityNet 1.3 风格 + 14 类中文名 + 元信息 envelope:

```json
{
  "schema": "rawvideo-timeline-v1",
  "input_video": "/abs/path/input.mp4",
  "input_video_clip_id": "<32-hex>",
  "input_video_duration_sec": 312.4,
  "extraction": {
    "fps_original": 30.0,
    "fps_used": 25,
    "n_frames": 7810,
    "pp_tsm_weight_sha256": "<64-hex>",
    "pp_tsm_inference_sha256": "<64-hex>",
    "pp_tsm_config_hash": "af295a1a6c5bb73e",
    "feature_pkl_path": "<out>/feature.pkl"
  },
  "bmn_inference": {
    "checkpoint": "experiments/<run>/BMN_epoch_00020.pdparams",
    "checkpoint_sha256": "<64-hex>",
    "subset": "validation",
    "ar_at": null
  },
  "command_version": "<git commit hash>",
  "produced_at": "2026-05-13T14:30:00Z",
  "results": [
    {
      "start_sec": 12.4,
      "end_sec": 14.2,
      "label_id": 7,
      "label_name": "侧旋",
      "score": 0.842,
      "rank_in_window": 0
    },
    ...
  ]
}
```

字段规则:
- `schema = "rawvideo-timeline-v1"` 与 `bmn-eval-v1` (v0.2.x) 严格区分
- `bmn_inference.ar_at` 始终为 null (推理时无 GT, 不计指标)
- `results[i].label_name` 是 14 类中文名, 直接来自 `pingpong_competition_bmn.yaml::classes::display_name`
- `extraction.pp_tsm_inference_sha256` 是导出后的 inference 双文件确定性 hash, 用于审计推理一致性

---

## R13 — `pp infer-rawvideo` 与现有 BMN eval 模块的复用边界

### 背景

spec.md FR-042 强制要求复用现有 `models/bmn.py + cli/eval.py + scripts/prepare_bmn_inputs.py`. 但这些模块当前签名是面向"训练完的 ckpt + 已有 GT 数据集"的, 不是"任意 pkl + 无 GT"的.

### 决策: 暴露 3 个新公共 API, 不分叉

1. **`upstream_adapter.trainer.run_upstream_bmn_eval(reuse_existing=True, gt_required=False)`**: 现有函数已有 `reuse_existing`; 新加 `gt_required` 参数. 当 `False` 时跳过 cal_metrics (没有 GT 算不了 AR@AN), 只产 `bmn_results_<subset>.json`.

2. **`scripts/prepare_bmn_inputs.py` → `prepare_bmn_inputs_for_inference(features_dir, output_dir)`**: 把现有 main() 拆成两个函数, 一个用于训练 (有 GT, 切窗 + label.json), 一个用于推理 (无 GT, 只切窗到 `feature/<name>_<start>_<end>.npy` + 写最小 `label_fixed.json` 让 BMN dataloader 不报错).

3. **`cli/infer_rawvideo.py`** (新文件): 编排
   - `extract_feat.extract_one(video, output_pkl)` (从 R10 export 出的 inference 模型抽)
   - `prepare_bmn_inputs.prepare_bmn_inputs_for_inference(...)`
   - `upstream_adapter.trainer.run_upstream_bmn_eval(..., gt_required=False)`
   - 解析 `bmn_results_<subset>.json` → 写 `timeline.json` (R12 schema)
   - 调可视化模块 (US3 已有 `inference/visualize.py`) 渲染 mp4

### 输出 → 阶段 1

- contracts/cli.md 增加 `scripts/export_pptsm_inference.py` + 3 个新 cli 子命令的契约
- data-model.md 增加 PPTSMInferenceModel + 更新 ImageFeaturePkl + 增加 RawVideoTimelineResult 实体

---

## 已解决的所有 NEEDS CLARIFICATION

| 来源 | 问题 | 解决方式 |
|------|------|---------|
| spec clarify Q1 | clip_id hash 来源 | sha256(file_bytes)[:32] |
| spec clarify Q2 | PP-TSM inference 模型获取 | scripts/export_pptsm_inference.py 自动转换 |
| R10 | `paddle.jit.to_static` 是否对 PP-TSM 有效 | 上游 tools/export_model.py 已验证, 失败时退化到上游脚本 |
| R11 | 30/60 fps 视频如何处理 | ffmpeg 强制 `-r 25` 重采样, 与 BMN GT 一致 |
| R12 | manifest.csv / timeline.json 字段 | 13 列 manifest + rawvideo-timeline-v1 schema (本文件已定) |
| R13 | 复用 v0.2.x 还是分叉 | 强制复用; 暴露 `gt_required=False` + 拆 prepare_bmn_inputs.main() |

**结论**: 阶段 0 所有未知项已解决, 可进入阶段 1 (data-model 增量 + contracts 增量 + quickstart 增量).
