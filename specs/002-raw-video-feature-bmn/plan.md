# 实施计划: 原始视频到 BMN 时序定位的端到端推理与训练适配

**分支**: `002-raw-video-feature-bmn` | **日期**: 2026-05-13 | **规范**: [spec.md](./spec.md)
**输入**: 来自 `/specs/002-raw-video-feature-bmn/spec.md` 的功能规范

**注意**: 此计划由 `/speckit.plan` 命令填充. 章程见 `.specify/memory/constitution.md` (v1.1.0).

## 摘要

为本仓库 v0.2.x 已建立的 BMN 时序定位主线补上**最后一块拼图**: 把原始 mp4 视频自动转为本项目兼容的 PP-TSN 特征 pkl, 让任何用户拿到本项目就能用自己的视频走端到端推理 (US1, P1) / 数据扩充 (US2, P2) / 微调基线 (US3, P3).

**技术方法** (来自 spec clarify Q1/Q2 + research.md R10-R13):
- **特征提取器**: PP-TSM ResNet50 + Kinetics-400 dense 预训练 (BCEBOS 公开 `ppTSM_k400_dense.pdparams`), 通过 `paddle.jit.to_static` 派生为 inference 双文件 (`scripts/export_pptsm_inference.py`).
- **抽帧策略**: ffmpeg 强制 25 fps + 224×224 ImageNet 标准化 + batch_size=32, 与上游 `applications/FootballAction/extractor/extract_feat.py` 完全对齐 (FR-035).
- **clip_id**: sha256(file_bytes)[:32], 抗改名 + 跨机器一致 (SC-013).
- **CLI 接口**: 3 个新子命令 (`pp extract-feat / build-feature-pkls / infer-rawvideo`) + 1 个一次性脚本 (`scripts/export_pptsm_inference.py`); 全部接受 `--allow-dirty`, 退出码体系与 v0.2.x 一致.
- **复用规则**: `pp infer-rawvideo` 内部**必须**复用 `models/bmn.py` + `cli/eval.py::_run_bmn_eval` + `scripts/prepare_bmn_inputs.py`, 不允许分叉 (FR-042). 必要时把现有函数拆出 `gt_required=False` / `prepare_bmn_inputs_for_inference()` 公共 API.
- **章程**: 不新增 PaddleVideo patches (SC-015 硬约束); 所有业务参数在 `configs/models/pp_tsm_extractor.yaml` (新文件); Python 3.11 隔离 venv 不变.

## 技术背景

**语言/版本**: Python **3.11** (固定, 与 v0.1.x/v0.2.x 一致).

**主要依赖** (无新增):
- PaddlePaddle-GPU 2.6.2 (与 v0.2.x 一致, paddle.jit API 在此版本稳定)
- PaddleVideo (上游 submodule, `release/2.2.0` @ `da9a8ce8`, 4 patches 不变)
- ffmpeg (CLI, 已在 env-check 中检测; v0.1.x 已要求)
- numpy / Pillow / pickle (标准库或已存在)
- click + PyYAML (CLI 框架, v0.1.x 已用)

**存储**: 文件系统; 新增固定路径 `data/raw/pretrained/` (PP-TSM 训练权重 + 派生 inference 文件) 与 `data/raw/.tmp/` (抽帧临时目录, 命令结束清理), 全部 gitignore.

**测试** (新增):
- `tests/unit/test_pp_tsm_extractor.py`: 验证 PP-TSM 抽出的特征 shape == (N, 2048), dtype == float32 (FR-037).
- `tests/unit/test_clip_id.py`: 验证同一文件内容跨调用 hash 一致, 不同文件 hash 不同 (clarify Q1).
- `tests/unit/test_export_pptsm_inference.py`: mock paddle.jit.save, 验证 marker.json 写入正确; 实际导出不进 unit (太慢, 走 integration).
- `tests/integration/test_extract_feat_e2e.py`: 用一个 5 秒小 mp4 (本仓库自带 fixture) 跑完 extract-feat → 检查 .pkl 形状.
- `tests/integration/test_infer_rawvideo_e2e.py`: 用同一 fixture + mock BMN ckpt → 检查 timeline.json schema.

**目标平台**: Linux + 单 NVIDIA GPU (T4 / V100 / 3090+), 显存 ≥ 8GB. CPU 模式仅作 env-check 回退.

**项目类型**: CLI 工具扩展 (单一 Python 项目, 不需要 web/前后端拆分).

**性能目标**:
- 抽特征吞吐 ≥ 80 帧/秒 (SC-011, T4, batch_size=32, 224×224)
- 端到端推理 5 分钟视频 ≤ 5 分钟 (SC-010)
- 抽特征数值确定性: 同一视频跨机器 cosine ≥ 0.999 (SC-013)

**约束条件**:
- **不新增 patches** (SC-015 硬约束 + 章程 VI)
- **不修改训练代码** (`pp train` / `prepare_bmn_inputs.py` 不变, FR-045)
- **不破坏向后兼容** (现有 `Features_competition_train.tar.gz` 仍可直接训练)
- **抽帧文件大小**: 5 分钟 25fps 视频抽帧约 7500 张 224×224 jpg ≈ 350MB; 必须走 `data/raw/.tmp/` 不污染系统 tmp.

**规模/范围**:
- 业务代码: 4 个新模块 (extract_feat / build_feature_pkls / infer_rawvideo / extractors/pp_tsm) + 1 个脚本 (export_pptsm_inference) ≈ ~600 行
- 配置: 1 份新 yaml (`pp_tsm_extractor.yaml`) ≈ ~80 行
- 测试: 5 个新测试文件 ≈ ~300 行

## 章程检查

> **门控**: 阶段 0 研究前必须通过. 阶段 1 设计后重新检查.

| 原则 | 状态 | 论证 |
|------|------|------|
| **I 规范与计划优先** | ✅ | spec.md → clarify (2 轮 Q&A) → 本 plan.md, 顺序无跳跃. 18 FR + 6 SC 都可追溯到具体测试用例. |
| **II 可复现实验** | ✅ | manifest.csv (R12) + timeline.json (R12) 都含 PP-TSM 训练权重 sha256 + inference 模型 combined sha256 + config_hash + git commit, 满足"四元组" (这里是"五元组": commit + config_hash + 训练权重 sha256 + inference 模型 sha256 + 抽帧 fps). |
| **III 配置驱动, 拒绝硬编码** | ✅ | `configs/models/pp_tsm_extractor.yaml` 唯一权威; `seg_num=8 / seglen=1 / batch_size=32 / fps=25 / mean=[0.485,0.456,0.406] / std=[0.229,0.224,0.225]` 全部在 yaml. CLI `--fps / --batch-size` 仅做 yaml override, 不引入新硬编码. |
| **IV 数据完整性** | ✅ | clip_id = sha256(file_bytes)[:32] (clarify Q1) — **抗改名 + 跨机器一致**, 任何用户合并自己视频与上游数据集都不会冲突. 命令**不**做 split 划分 (那是 `pp data-prepare` 的职责). |
| **V 评估纪律** | ✅ | 推理路径 (US1) **不**输出 top1/top5/AR@AN — 推理时无 GT, 这些指标算不出. 但 timeline.json 输出 `n_proposals` + `score` 分布, 仍可下游做人工评估. 训练扩充路径 (US2) 不直接评估, 只产出能喂给 `pp eval` 的数据. |
| **VI 上游兼容与最小侵入** | ✅ | **不新增 patches** (SC-015). PP-TSM 网络通过 `paddlevideo.modeling.builder.build_model(cfg)` 在外层 import, **不改其源码**. 抽特征代码完全在本仓库 `src/pingpong_av/extractors/`. |
| **VII 端到端可运行** | ✅ | quickstart 4 条命令完成 (curl + export + infer-rawvideo + 可选 build), 严格 ≤ 5 (实际是 4). 第一次运行自动触发 export (FR-038a), 用户感知 ≤ 5 条. |
| **VIII 隔离 Python 3.11** | ✅ | 所有命令通过 `.venv/bin/` 调用; 无新依赖 (ffmpeg / numpy / Pillow / paddle 全在 v0.2.x requirements). 不需要 lock 文件刷新. |

**结论**: 全部通过, 无需"复杂度跟踪". 进入阶段 0.

## 项目结构

### 文档(此功能)

```
specs/002-raw-video-feature-bmn/
├── plan.md              # 此文件
├── research.md          # 阶段 0 输出 (R10-R13: 模型导出 + ffmpeg 策略 + manifest schema + 复用边界)
├── data-model.md        # 阶段 1 输出 (RawVideo / ImageFeaturePkl / PPTSMTrainWeight / PPTSMInferenceModel / RawVideoTimelineResult)
├── quickstart.md        # 阶段 1 输出 (3 个场景)
├── contracts/
│   └── cli.md           # 4 个命令契约
├── checklists/
│   └── requirements.md  # 已完成 (specify 阶段)
└── tasks.md             # 阶段 2 输出 (/speckit.tasks 命令产生, 不在 plan 阶段创建)
```

### 源代码(仓库根目录, 002 增量)

```
char_pp_prj/
├── configs/
│   ├── models/
│   │   ├── pp_tsm_pingpong.yaml      # (已存在, v0.1.x 训练用)
│   │   ├── bmn_pingpong.yaml         # (已存在, v0.2.x 训练用)
│   │   ├── videoswin_tennis.yaml     # (已存在, v0.1.1 推理用)
│   │   └── pp_tsm_extractor.yaml     # ★ 新增 (002, 抽特征用)
│   └── (其他不变)
├── scripts/
│   ├── bootstrap.sh                  # (已存在)
│   ├── apply_upstream_patches.sh     # (已存在, 4 patches)
│   ├── prepare_bmn_inputs.py         # (已存在, 002 拆出 prepare_bmn_inputs_for_inference 公共 API)
│   ├── wait_and_eval.sh              # (已存在, v0.2.1)
│   └── export_pptsm_inference.py     # ★ 新增 (002, FR-038a)
├── src/pingpong_av/
│   ├── cli/
│   │   ├── extract_feat.py           # ★ 新增 (002, FR-033)
│   │   ├── build_feature_pkls.py     # ★ 新增 (002, FR-034)
│   │   ├── infer_rawvideo.py         # ★ 新增 (002, FR-039)
│   │   └── (其他已存在子命令不变)
│   ├── extractors/                   # ★ 新增子包 (002)
│   │   ├── __init__.py
│   │   ├── pp_tsm_inference.py       # PP-TSM inference Predictor 包装
│   │   ├── ffmpeg_frames.py          # ffmpeg 抽帧 + 临时目录管理
│   │   ├── clip_id.py                # sha256(file_bytes)[:32] 工具
│   │   └── manifest.py               # manifest.csv 写出器
│   ├── inference/
│   │   ├── visualize.py              # (已存在, US3, 002 复用)
│   │   └── (其他不变)
│   ├── models/
│   │   ├── bmn.py                    # (已存在, 002 不改)
│   │   ├── pp_tsm.py                 # (已存在, 训练用; 002 的 extractor 是另一份)
│   │   └── registry.py               # (已存在)
│   └── upstream_adapter/
│       └── trainer.py                # (已存在, v0.2.1; 002 加 gt_required 参数到 run_upstream_bmn_eval)
├── tests/
│   ├── unit/
│   │   ├── test_pp_tsm_extractor.py  # ★ 新增 (002)
│   │   ├── test_clip_id.py           # ★ 新增 (002)
│   │   └── test_export_pptsm_inference.py  # ★ 新增 (002, mock paddle.jit)
│   ├── integration/
│   │   ├── test_extract_feat_e2e.py   # ★ 新增 (002)
│   │   ├── test_build_feature_pkls_e2e.py  # ★ 新增 (002)
│   │   └── test_infer_rawvideo_e2e.py # ★ 新增 (002)
│   └── fixtures/
│       └── mini_pingpong_5s.mp4      # ★ 新增 (~500KB, 用于 e2e 测试)
└── data/raw/
    ├── pretrained/                   # ★ gitignore (已被 data/raw/** 默认忽略)
    │   ├── ppTSM_k400_dense.pdparams # 用户下载, ~120MB
    │   ├── ppTSM.pdmodel             # export 派生, ~2MB
    │   ├── ppTSM.pdiparams           # export 派生, ~120MB
    │   └── .export_marker.json       # 幂等标记
    └── .tmp/                         # ★ gitignore (新增 pattern)
        └── extract_<run_id>/         # 临时抽帧, 命令结束清理
```

**结构决策**: 单一项目结构 (与 v0.1.x/v0.2.x 一致). 关键设计:
- **`src/pingpong_av/extractors/` 是新子包**, 与 `models/` (训练时的网络结构) 分离 — 抽特征是 inference-only 路径, 与训练 PP-TSM 的代码不同 (后者用动态图, 前者用静态图 Predictor).
- **3 个新 cli 都在 `src/pingpong_av/cli/`**, 通过 `cli/__init__.py` 注册, 与现有 `train_cmd / eval_cmd` 一致.
- **测试 fixture `mini_pingpong_5s.mp4`** 入库 (~500KB) — 这是唯一一个真实 mp4 入库的例外, 因为 e2e 测试需要它. 5 秒视频通过 ffmpeg 自合成, 不涉及任何真实人物/版权.

## 阶段 0 输出

✅ `research.md` 已生成, 含 R10-R13:
- R10: PP-TSM 训练权重 → inference 模型转换实操
- R11: ffmpeg 抽帧策略与 fps 一致性
- R12: manifest.csv 与 timeline.json 字段表
- R13: 与现有 BMN eval 模块的复用边界

## 阶段 1 输出

✅ `data-model.md` 已生成 (5 个新实体: RawVideo / ImageFeaturePkl / PPTSMTrainWeight / PPTSMInferenceModel / RawVideoTimelineResult)
✅ `contracts/cli.md` 已生成 (1 脚本 + 3 cli 契约)
✅ `quickstart.md` 已生成 (3 个使用场景)

## 阶段 1 后再次章程检查

| 原则 | 重新检查后状态 | 备注 |
|------|---------------|------|
| I-VIII | 全部 ✅ | 阶段 1 设计未引入新违规. data-model.md 中 5 个实体的字段都尊重 yaml 配置驱动 (III), 都有审计字段 (II), clip_id 抗冲突 (IV). |

## 复杂度跟踪

> 章程检查全部通过, 无违规. 本表为空.

| 违规 | 为什么需要 | 拒绝更简单替代方案的原因 |
|-----|----------|------------------------|

---

## 下一步

✅ 阶段 0 (research.md) + 阶段 1 (data-model + contracts + quickstart) 全部完成. 可执行 `/speckit.tasks` 进入阶段 2 (任务分解).

阶段 2 预期产出:
- `tasks.md` 含 ~10-15 个 T200 系列任务 (T200-T214 估计)
- 任务按依赖图排序: 单测先于集成测试; 模块实现先于 CLI 集成; 公共 API 重构先于新模块调用.
- 每个任务对应至少 1 个 FR, 标注是否阻塞 quickstart 的 4 条命令.
