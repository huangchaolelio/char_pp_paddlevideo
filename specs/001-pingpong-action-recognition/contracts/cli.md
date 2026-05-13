# CLI 契约: `pp` 命令

**功能**: `001-pingpong-action-recognition`
**形态**: CLI 工具 (entry point: `pp`, 由 `pyproject.toml` 声明 → `src/pingpong_av/cli/__init__.py`)
**调用环境约束**: 只能在项目隔离的 Python 3.11 `.venv` 中运行 (章程 VIII); `env-check` 首当其冲负责
快速识别违规并给出修复指引.

本契约是本项目**唯一**对外接口. 所有命令都遵循以下通则:

1. **配置驱动优先**: 每个命令均接受 `--config <path>` 指向一份 YAML; CLI 参数仅用于覆盖配置或指定
   input/output 路径. 严禁通过 CLI 参数引入业务决策 (章程 III).
2. **退出码**: `0` 成功; `1` 用户输入错误; `2` 环境/依赖问题 (由 env-check 统一判定); `3` 章程硬约束
   违反 (工作区脏 / 测试集误用 / 划分泄漏); `4` 运行时失败.
3. **日志去向**: 默认 stderr 打印人类可读日志, stdout 只打印结构化结果 (JSON / 文件路径), 方便管道化.
4. **章程约束映射**: 每个命令最后给出它受哪些章程原则约束, 便于评审核对.

---

## 命令列表总览

| 命令 | 主要目的 | 对应 FR | 典型位置在 quickstart |
|------|----------|---------|----------------------|
| `pp env-check` | 验证 Python 3.11 + 隔离环境 + PaddlePaddle 可导 | FR-001, FR-003 | 1/5 |
| `pp data-prepare` | 拉取/整理公开乒乓球数据集, 生成划分 list | FR-004~007 | 2/5 |
| `pp train` | 启动训练 | FR-008~010, FR-012, FR-018 | 3/5 |
| `pp eval` | 在测试集上评估模型 | FR-011 | 4/5 |
| `pp infer-clip` | 单片段推理 | FR-013 | — (调试用) |
| `pp infer-video` | 长视频端到端推理 + 可视化 | FR-014~016 | 5/5 |
| `pp infer-pkl` | 用上游 VideoSwin TableTennis 权重推理样例 pkl (US5) | FR-020~022 | — (上游演示) |

**quickstart 的 5 条命令** 对应上表中标注了位置的命令, 满足章程 VII (≤ 5 条命令端到端可运行).
`pp infer-pkl` 是 US5 的独立路径, 不在 quickstart 主链, 但可作为上游官方乒乓球任务的端到端验证.

---

## `pp env-check`

**目的**: 快速验证当前运行环境满足章程 VIII 的硬要求, 并给出明确修复指引.

**用法**:
```
pp env-check [--strict]
```

**参数**:
- `--strict`: 启用时, 不仅验证 Python 与解释器路径, 还尝试 `import paddle` + `paddle.utils.run_check()`
  + `import paddlevideo`, 验证 submodule 已安装.

**成功输出** (stdout, JSON):
```json
{
  "python_version": "3.11.9",
  "interpreter_path": "/abs/path/.venv/bin/python",
  "is_project_venv": true,
  "paddle_importable": true,
  "paddle_version": "2.6.x",
  "paddlevideo_importable": true,
  "gpu_available": true,
  "cuda_version": "11.8",
  "gpu_model": "NVIDIA GeForce RTX 3090"
}
```

**失败模式**:
- 解释器**不是**项目 `.venv/bin/python` → 退出码 2, stderr 打印:
  `ERROR: 当前解释器为 /usr/bin/python, 不是项目隔离环境. 请先 source .venv/bin/activate 或直接使用 .venv/bin/pp.`
- Python 版本 != 3.11.x → 退出码 2, stderr 打印:
  `ERROR: Python 版本为 3.10.x, 本项目锁定 Python 3.11. 请参照 quickstart 重建 .venv.`
- `--strict` 下 paddle 不可导 → 退出码 2, 指引运行 `scripts/bootstrap.sh`.

**章程约束**: VIII (必须过此门才能进行后续命令, quickstart 中设计为第 1 步).

---

## `pp data-prepare`

**目的**: 幂等地拉取/整理公开乒乓球动作数据集, 切分片段, 生成 `data/splits/*.txt` + `*.meta.jsonl`.

**用法**:
```
pp data-prepare --config configs/datasets/pingpong_public.yaml [--force]
```

**参数**:
- `--config <path>` (必填): 数据集配置 YAML.
- `--force`: 忽略已有产物, 重新整理 (默认为幂等, 已有产物则跳过).

**输入**:
- `--config` 指向的 YAML, 至少包含:
  - `source`: 数据集来源, **支持 4 种 type**:
    - `url_list` (公网 HTTP/HTTPS, 含 `urls: [...]`, 可选 `insecure: true`)
    - `local_dir` (本地目录, 含 `path: <abs>`)
    - `manual` (用户手动准备, 含 `sentinel_relpath` + `manual_steps` 引导)
    - `cos` (腾讯云 COS, 含 `keys: [...]` + 可选 `bucket/region/prefix/extract/max_thread`; 凭据从 `.env` 读 `COS_SECRET_ID/COS_SECRET_KEY/COS_BUCKET/COS_REGION/COS_VIDEO_PREFIX`, 不入 yaml — FR-023)
  - `classes`: ActionClass 数组 (见 data-model.md)
  - `split_strategy`: `official` (使用数据集自带划分) 或 `by_video_ratio` (按 source_video_id 分层划分, 带 ratio)
  - `split_version`: 字符串 (每次重新划分必须 bump)
- 对于 `cos` 模式, 当数据是 PaddleVideo BMN 训练用的 `Features_competition_train/*.pkl` 时, 系统通过 `_try_read_bmn_features` 路径自动识别 (FR-025): 每个 .pkl 视为一个 source_video, label_id 从 `label_cls*.json` 中按 url 取众数填充, 用作 splitter 分层校验.

**产出**:
- `data/raw/`, `data/clips/` 下的实际数据文件 (**不入库**)
- `data/splits/{train,val,test}.txt`, `data/splits/{train,val,test}.meta.jsonl` (**入库**)

**约束 (硬)**:
- 完成后必须运行内置的 "no-leakage check": 扫描三份 meta, 确认不同 split 间没有共同的
  `source_video_id`; 发现泄漏 → 退出码 3, 清理当前划分并拒绝写入 list 文件. (章程 IV)

**成功输出** (stdout, JSON):
```json
{
  "dataset": "pingpong_public",
  "split_version": "v1.0",
  "counts": { "train": 1234, "val": 300, "test": 380 },
  "num_classes": 8,
  "list_files": [
    "data/splits/train.txt",
    "data/splits/val.txt",
    "data/splits/test.txt"
  ]
}
```

**章程约束**: III (配置驱动), IV (无泄漏, 划分入库), VII (幂等可重放).

---

## `pp train`

**目的**: 启动一次训练, 产生 Experiment 目录.

**用法**:
```
pp train --config configs/models/pp_tsm_pingpong.yaml
         [--seed 2026]
         [--resume experiments/<run_id>/checkpoints/epoch_X.pdparams]
         [--allow-dirty]
         [--output-root experiments/]
```

**参数**:
- `--config` (必填): 模型 + 训练配置 YAML. 内部通过 `!include` 引用数据集 YAML.
- `--seed` (可选, 覆盖配置中的 seed)
- `--resume`: 从某 checkpoint 恢复 (FR-010)
- `--allow-dirty`: 允许 git 工作区有未提交修改; 不加时, 工作区脏则**退出码 3**, 拒绝启动 (章程 II).
- `--output-root`: Experiment 根目录 (默认 `experiments/`).

**行为** (与 data-model.md Experiment 实体对齐):
1. 读取配置; 设置随机种子到 paddle/numpy/random (FR-018).
2. 计算 git commit + 工作区状态 + config_hash; 创建 `experiments/<run_id>/` 目录.
3. 写入初始 `manifest.json` (status=running).
4. 通过 `src/pingpong_av/upstream_adapter/trainer.py` 调用 PaddleVideo 训练主循环.
5. 定期保存 checkpoint 到 `experiments/<run_id>/checkpoints/`.
6. 结束时 (成功/失败/中断) 更新 `manifest.json` 的 status 与 finished_at.

**stdout 输出** (最终 JSON):
```json
{
  "run_id": "20260511-203015-a3f1c8e-pp_tsm_pingpong",
  "status": "succeeded",
  "best_checkpoint": "experiments/.../checkpoints/best.pdparams",
  "best_val_top1": 0.812
}
```

**章程约束**: II (commit + config_hash + seed 入 manifest), III (配置驱动), VIII (必须在 .venv 中运行).

---

## `pp eval`

**目的**: 在测试集上评估模型, 产生章程 V 要求的完整指标报告.

**用法**:
```
pp eval --checkpoint experiments/<run_id>/checkpoints/best.pdparams
        [--split test]
        [--batch-size 16]
        [--output experiments/<run_id>/metrics.json]
```

**参数**:
- `--checkpoint` (必填): Model 实体 (data-model.md); 工具自动从同目录的 `config.yaml` 重建模型.
- `--split`: 默认 `test`; **仅允许 `test` 或 `val`**. 传入其他值 → 退出码 1.
  **特别约束**: 在默认的"发布型评估"模式下, 若同一 run_id 的 metrics.json 已存在且 split=test,
  再次传 `--split test` 必须携带 `--rerun` 标志, 否则**退出码 3** (章程 IV/V: 测试集不做反复挑选).
- `--batch-size`, `--output`: 标准参数.

**产出**:
- JSON 写入 `--output` (默认 `experiments/<run_id>/metrics.json`); 同时追加到 `manifest.json`.

**JSON schema (metrics-v1, PP-TSM 路径)**:
```json
{
  "schema": "metrics-v1",
  "checkpoint": "experiments/.../best.pdparams",
  "split": "test",
  "n_samples": 380,
  "top1": 0.742,
  "top5": 0.921,
  "macro_avg": { "precision": 0.711, "recall": 0.698, "f1": 0.701 },
  "per_class": {
    "serve":           { "precision": 0.82, "recall": 0.80, "f1": 0.81, "support": 48 },
    "forehand_attack": { "precision": 0.76, "recall": 0.74, "f1": 0.75, "support": 95 }
  },
  "confusion_matrix_path": "experiments/.../confusion_matrix.png",
  "produced_at": "2026-05-11T21:30:00Z"
}
```

**JSON schema (bmn-eval-v1, BMN 时序定位路径; FR-029 / SC-009)**:

BMN 模型 (model.name=bmn) 自动走时序定位评估分支, 与 PP-TSM 路径**共用 cli 入口**但输出不同 schema:
```json
{
  "schema": "bmn-eval-v1",
  "checkpoint": "experiments/.../BMN_epoch_00007.pdparams",
  "run_id": "20260512-145311-09f9d63-train-bmn_pingpong",
  "split": "val",
  "subset": "validation",
  "n_videos_evaluated": 1967,
  "n_proposals": 196700,
  "metrics": {
    "ar@1":   28.78,
    "ar@5":   59.17,
    "ar@10":  68.27,
    "ar@100": 80.37
  },
  "class_names": ["摆短", "拉", "控制", "侧身拉", "劈长", "拧", "挑", "侧旋", "转不转", "中性", "勾球", "普通", "逆旋转", "下蹲"],
  "result_path": "experiments/.../bmn_eval/results/bmn_results_validation.json"
}
```

BMN 评估特性:
- **默认 ``reuse_existing=True``** (FR-030): 若同 ckpt 的 ``bmn_results_<subset>.json`` 已存在, 跳过 GPU 前向, 仅重算 metrics (~30s 而非 ~8min). 用于训练并行 / 调参 / 失败重试.
- **AR@AN** (Average Recall at Average Number of proposals) 取代 top1/top5, 是 ActivityNet 1.3 检索 + 时序定位的标准指标.
- ``result_path`` 指向上游写出的 ActivityNet 1.3 风格 JSON, 含 per-video proposals 列表, 可被下游 NMS / 后处理工具直接消费.
- ``data/bmn/BMN_Test_results/`` 目录由 eval 自动创建以满足上游 ``anet_prop.py:167`` 硬编码路径 (FR-031), 已加入 ``.gitignore``.

**checkpoint 路径布局 (FR-032)**:
- PP-TSM: ``<run>/checkpoints/best.pdparams``
- BMN: ``<run>/BMN_epoch_NNNNN.pdparams`` (上游直接写到 run 根目录, 命名按 epoch)

``_find_run_dir`` 自动识别两种布局, 不需要用户指定.

**章程约束**: V (必须含 top1/top5/per-class/macro-avg 或 AR@AN), IV (禁止反复测试集选型), II (结果写回 manifest).

---

## `pp infer-clip`

**目的**: 对单个已切分的视频片段做动作分类; 调试 / 下游集成用.

**用法**:
```
pp infer-clip --checkpoint <path> --input <video_file> [--topk 5] [--output <json_path>]
```

**参数**:
- `--checkpoint`, `--input` (必填)
- `--topk` (默认 5)
- `--output`: 不提供时只输出到 stdout.

**产出**: PredictionResult (单片段 JSON, 见 data-model.md).

**错误行为**:
- 视频无法打开 / 过短 / 格式不支持 → 退出码 1, stderr 清晰报错 (与 FR-016 一致).

**章程约束**: III (配置驱动), V (输出结构化可机器读).

---

## `pp infer-video`

**目的**: 对一段完整乒乓球比赛/训练视频做端到端推理: 滑窗 + 阈值 + 合并 + MP4 叠加 + JSON 时间轴.

**用法**:
```
pp infer-video --checkpoint <path>
               --input <video_file>
               --inference-config configs/inference/sliding_window.yaml
               --output-dir <dir>
               [--no-viz]
```

**参数**:
- `--checkpoint`, `--input`, `--output-dir` (必填)
- `--inference-config` (必填): 包含 window_sec / stride_sec / conf_threshold / merge_gap_sec
- `--no-viz`: 只写 JSON, 不渲染 MP4 (性能测试用). 默认**必须**产出 MP4 + JSON (FR-015).

**产出** (在 `<output-dir>` 下):
- `<basename>.timeline.json` — TimelineSegment 数组格式 (data-model.md)
- `<basename>.viz.mp4` — 叠加了类别 + 置信度文本的 MP4 (未加 `--no-viz`)

**错误行为**:
- 输入视频不可读 → 退出码 1.
- 单个滑窗推理失败 (例如 OOM 单例、解码失败) **必须**跳过并在 JSON `warnings[]` 数组中记录,
  不得使整体流程终止 (FR-016); 仅当 > 50% 窗口失败时才退出码 4.

**SC-003 时延对齐**: 该命令的实现必须保证单 GPU 环境下对 ≤ 10 分钟视频, 端到端耗时 ≤ 视频时长 × 2.

**章程约束**: III (滑窗参数配置化), V (结构化 JSON), VII (是 quickstart 最后一步).

---

## `pp infer-pkl`

**用途 (US5 / FR-020~022)**: 用 PaddleVideo 上游官方乒乓球训练好的权重 (`VideoSwin_tennis.pdparams`, 380MB) 推理一个上游样例 pkl (`example_tennis.pkl`, 7.4MB), 在 AI Studio 数据未就绪前演示完整推理路径.

**与 `pp infer-clip` 的区别**:
- `pp infer-clip` 接受**用户提供的 mp4** + **任意已训练的 PP-TSM checkpoint** (业务主线)
- `pp infer-pkl` 接受**上游官方 pkl** + **上游官方 VideoSwin 权重** (US5 独立路径); 不走 PP-TSM, 不依赖 `pp train` 产物

**用法**:

```text
pp infer-pkl --pkl <path.pkl> --checkpoint <VideoSwin_tennis.pdparams> [--topk 5] [--num-seg 32] [--output <json>]
```

**参数**:
- `--pkl <path>`: 上游样例 pkl, 元组格式 `(video_name, label_dict, list[jpeg_bytes])`. 必填.
- `--checkpoint <path>`: `VideoSwin_tennis.pdparams` 路径. 必填. 缺失时 stderr 给出**完整 curl 下载命令** (FR-021).
- `--topk <int>`: 默认 3. 必须 ≥ 1.
- `--num-seg <int>`: 均匀采样的帧数, 默认 32 (与 `videoswin_tabletennis.yaml::runtime_cfg.test.num_seg` 对齐).
- `--output <path>`: 可选 JSON 输出路径; 不指定时只输出到 stdout.

**输出 JSON schema** (data-model.md `pkl-prediction-v1`):
- `input`: `{pkl_path, video_name, n_frames_in_pkl, n_frames_sampled}`
- `model`: `{checkpoint, framework=RecognizerTransformer, backbone=SwinTransformer3D, head=I3DHead, num_classes=8}`
- `ground_truth`: pkl 中**完整透传**的 labels dict (含 `正反手 / 动作类型 / 发球` 等)
- `ground_truth_action_id`: 模型推理目标对应 GT (即 `ground_truth.动作类型`) — 单独命名避免多任务歧义
- `prediction.topk`: 排序后 Top-K 列表, 每项 `{id, name, score}`
- `prediction.top1_match_gt`: 布尔或 null
- `produced_at`: ISO 8601 UTC

**退出码**:
- `0` 成功
- `1` 用户输入错 (pkl/checkpoint 不存在, topk/num-seg 非法). checkpoint 缺失时 stderr 必须含 curl 下载命令.
- `2` 环境问题 (上游不可导, 例如 patches 未应用)
- `4` 运行时失败 (模型加载 / 推理异常 / 帧预处理异常)

**验收闸门 (SC-007)**:
在 `example_tennis.pkl` 上, Top-1 必须等于 pkl 内 `ground_truth.动作类型`, 且 Top-1 置信度 ≥ 0.90.

**章程约束**: III (帧采样 / 标准化参数与上游 yaml 对齐, 不硬编码), V (JSON 必含 `ground_truth_action_id` 与 `top1_match_gt` 让验收可机器读), VI (`videoswin_tennis.py` 通过上游 `build_model` 接入, 不复制上游网络代码).

---

## 退出码总览

| code | 含义 | 常见触发 |
|------|------|---------|
| 0 | 成功 | — |
| 1 | 用户输入错误 | 文件不存在 / 参数非法 / 视频不可读 |
| 2 | 环境问题 | 解释器非项目 .venv / Python != 3.11 / paddle 不可导 |
| 3 | 章程硬约束违反 | 工作区脏未加 --allow-dirty / 划分泄漏 / 测试集重复评估未加 --rerun |
| 4 | 运行时失败 | 训练发散 / 超过阈值的推理失败 |

---

## 不属于本契约的接口

以下情况**不**在本项目 MVP 范围内, 故契约中无对应内容:

- HTTP / gRPC / WebSocket 服务
- Web 前端 / 桌面 GUI
- 直接向第三方存储 (S3 / OSS) 读写
- 在线学习 / 增量学习入口

如未来需要扩展, 必须走新的 `/speckit.specify` → `/speckit.plan` 流程, 不得在本 CLI 上临时加 flag.
