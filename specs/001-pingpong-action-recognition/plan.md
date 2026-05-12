# 实施计划: 基于 PaddleVideo 的乒乓球视频动作识别系统

**分支**: `001-pingpong-action-recognition` | **日期**: 2026-05-11 | **规范**: [spec.md](./spec.md)
**输入**: 来自 `/specs/001-pingpong-action-recognition/spec.md` 的功能规范

**注意**: 此计划由 `/speckit.plan` 命令填充. 章程见 `.specify/memory/constitution.md` (v1.1.0).

## 摘要

在本仓库以 Git submodule 方式集成上游 PaddlePaddle/PaddleVideo, 在隔离的 Python 3.11 环境中构建一套
**配置驱动 + 可复现 + 端到端可运行**的乒乓球视频动作识别系统. 默认基线模型为 **PP-TSM** (使用官方
Kinetics 预训练权重微调), 数据来自公开的乒乓球动作数据集 (具体在阶段 0 研究中选定). 长视频识别采用
**固定窗口 + 步长滑窗 + 同类合并**, 同时输出 MP4 可视化视频和 JSON 时间轴.

技术方法 (来自 spec.md Clarifications + 章程 + 阶段 0 研究):
- **环境**: Python 3.11 隔离 venv + PaddlePaddle-GPU; 上游 PaddleVideo 作为 submodule, 通过适配层兼容 3.11
- **项目布局**: 业务代码 (`src/pingpong_av/`) 与上游 (`third_party/PaddleVideo/`) 严格隔离
- **接口**: CLI 命令行 (≤5 条命令完成全流程); 无 Web/HTTP 服务, 无外部 API
- **数据**: 数据准备脚本拉取公开数据集, 切分片段并产出训练/验证/测试 list 文件 (入库)
- **训练/评估**: 直接调用 submodule 中的 PaddleVideo 训练入口, 通过本仓库 `configs/` 下的乒乓球专用配置
- **推理**: 提供片段级 + 长视频两个 CLI 入口; 长视频内部用滑窗 + 后处理 + 可视化

## 技术背景

**语言/版本**: Python **3.11** (固定, 与章程原则 VIII 一致, 不允许在主分支引入其他 Python 版本兼容代码)
**主要依赖**:
- PaddlePaddle-GPU (与 CUDA 11.8 / 12.x 匹配, 由 PaddleVideo 上游 tag 决定具体版本)
- PaddleVideo (上游, 以 submodule 方式接入, 固定到 `release/2.2.0` 分支的具体 commit `da9a8ce8`, 由阶段 0 研究 R1 实地查询确定)
- OpenCV (cv2), decord 或 PyAV (视频读取/采样, 由 PaddleVideo 选型决定)
- ffmpeg (CLI 依赖, 用于长视频可视化产物渲染)
- numpy, scipy, scikit-learn (评估指标 + 后处理), tqdm, PyYAML, click (CLI)

**存储**: 文件系统; 数据集存于 `data/` (gitignore), 实验产物存于 `experiments/<run_id>/`; 数据划分
list 文件 (`data/splits/*.txt`) 入库

**测试**: pytest (轻量); 测试范围限定在: 数据准备脚本 (划分无泄漏校验)、滑窗后处理逻辑、配置加载、
CLI 入口可启动性. **不**为 PaddleVideo 上游模型本身写单元测试 (上游负责).

**目标平台**: Linux (Ubuntu 20.04+ / 22.04) + 单 NVIDIA GPU (≥ 12GB 显存推荐); 保留 CPU 回退路径
仅作环境验证 (spec.md 边界情况, FR-016).

**项目类型**: **CLI 工具 + 业务代码库** (单一 Python 项目结构, 不需要 Web 前后端拆分).

**性能目标**:
- 训练: 单 V100/A10/3090 在所选公开乒乓球数据集上完成 baseline 训练在合理时间内 (具体上限由阶段 0
  研究中数据集规模确定, 不低于 SC-002 指标要求).
- 推理: 长视频端到端流程在视频时长的 **2 倍以内** 完成 (SC-003).
- 可复现性: 同种子两次训练 Top-1 差异 ≤ ±2 个百分点 (SC-004).

**约束条件**:
- 仅 Python 3.11; 上游若有 3.11 不兼容代码, 必须由本项目 patch 层修复, 不得降级 (章程 VIII).
- 上游通过 submodule 接入, 不在主仓库内复制源码 (章程 VI).
- 所有可调参数必须配置驱动, 源码无硬编码业务参数 (章程 III).
- 训练/验证/测试在视频源层面互不重叠, 划分文件入库 (章程 IV).
- 评估必须给出 Top-1, Top-5, 每类指标; 长尾时附宏平均 (章程 V).
- main 分支始终保持 ≤ 5 条命令端到端可运行 (章程 VII).

**规模/范围**:
- 公开乒乓球数据集典型规模: 数千~数万个动作片段, ~10~20 个类别 (具体由阶段 0 选型决定).
- CLI 命令: ~6 个 (env-check, data-prepare, train, eval, infer-clip, infer-video).
- 配置文件: 1 份基线 (PP-TSM 乒乓球微调) + 1 份示例 (PaddleVideo 自带跑通用), 后续可扩展.

## 章程检查

*门控: 必须在阶段 0 研究前通过. 阶段 1 设计后重新检查.*

### 初次检查 (阶段 0 研究前)

| 原则 | 状态 | 说明 |
|------|------|------|
| I. 规范与计划优先 | ✅ | 已有 spec.md (含 Clarifications) 与本 plan; 所有设计回溯至 FR-xxx / 用户故事 |
| II. 可复现实验 | ✅ | 训练入口将强制设置随机种子, 实验目录 (`experiments/<run_id>/`) 记录 commit + 配置哈希 + 种子 + 指标 |
| III. 配置驱动, 拒绝硬编码 | ✅ | 所有数据路径/类别/超参由 `configs/*.yaml` 驱动; 类别表通过数据集元信息文件加载 |
| IV. 数据完整性 | ✅ | 数据准备脚本按 video-id 切分, list 文件入库 (`data/splits/*.txt`); 测试集仅在 `eval` 入口使用 |
| V. 评估纪律 | ✅ | `eval` CLI 默认输出 Top-1/Top-5 + 每类指标 + 测试集大小; 类别不平衡时附宏平均 |
| VI. 上游兼容与最小侵入 | ✅ | PaddleVideo 以 submodule 在 `third_party/PaddleVideo/` 接入; 业务代码独立于 `src/pingpong_av/`; patch 通过 `third_party/patches/` 维护 |
| VII. 端到端可运行 | ✅ | quickstart 设计为 5 条命令: env → data-prepare → train → eval → infer-video |
| VIII. 隔离的 Python 环境 | ✅ | 强制 `.venv/` (Python 3.11), `requirements/*.txt` 与锁文件入库; quickstart 含环境自检 (`env-check` CLI) |

**结论**: 全部通过, 无违规, 无需进入"复杂度跟踪"表格. 可进入阶段 0.

### 设计后重新检查 (阶段 1 完成后)

| 原则 | 状态 | 设计后核对要点 |
|------|------|----------------|
| I | ✅ | 每个 CLI 契约 (`contracts/cli.md`) 都映射到至少一条 FR |
| II | ✅ | `data-model.md` 中的 **Experiment** 实体强制包含 commit/config_hash/seed/metrics 四元组 |
| III | ✅ | 所有 CLI 参数都有对应的配置字段或显式来自 `configs/*.yaml`; 没有 CLI 默认值会"偷偷决定"业务行为 |
| IV | ✅ | `data-prepare` 契约明确: 划分按 video-id, 输出列表入库; `eval` 契约禁止读非测试集 split |
| V | ✅ | `eval` 契约规定输出至少含 top1/top5/per-class/macro-avg |
| VI | ✅ | `contracts/cli.md` 显式说明 train/eval 通过 `paddlevideo.tools.train/test` 入口调用, 不复制上游模块 |
| VII | ✅ | `quickstart.md` 列出 ≤ 5 条命令 (env-check / data-prepare / train / eval / infer-video) |
| VIII | ✅ | `quickstart.md` 包含 `python --version` 自检; `env-check` CLI 校验解释器路径与版本; `requirements-lock.txt` 入库 |

**结论**: 设计后全部通过, 无新增违规.

## 项目结构

### 文档(此功能)

```
specs/001-pingpong-action-recognition/
├── plan.md              # 此文件 (/speckit.plan 命令输出)
├── spec.md              # 功能规范 (含 Clarifications)
├── research.md          # 阶段 0 输出 (/speckit.plan 命令)
├── data-model.md        # 阶段 1 输出 (/speckit.plan 命令)
├── quickstart.md        # 阶段 1 输出 (/speckit.plan 命令)
├── contracts/           # 阶段 1 输出 (/speckit.plan 命令)
│   └── cli.md           # CLI 命令契约 (本项目唯一对外接口)
├── checklists/
│   └── requirements.md  # 来自 /speckit.specify
└── tasks.md             # 阶段 2 输出 (/speckit.tasks 命令 - 非 /speckit.plan 创建)
```

### 源代码(仓库根目录)

本项目为单一 Python CLI 工具 + 业务库, 选择"单一项目"布局并对应章程要求做严格隔离.

```
char_pp_prj/
├── .venv/                          # Python 3.11 隔离环境 (gitignore, 章程 VIII)
├── .specify/                       # speckit 工件 (已存在)
├── specs/                          # 规范与计划 (speckit 管理)
│
├── src/
│   └── pingpong_av/                # 业务代码 (章程 VI: 与上游严格隔离)
│       ├── __init__.py
│       ├── cli/                    # CLI 入口
│       │   ├── __init__.py
│       │   ├── env_check.py        # `pp env-check`  (FR-001, 章程 VIII)
│       │   ├── data_prepare.py     # `pp data-prepare`  (FR-004, FR-005, FR-007)
│       │   ├── train.py            # `pp train`  (FR-008, FR-009, FR-010, FR-012, FR-018)
│       │   ├── eval.py             # `pp eval`  (FR-011, 章程 V)
│       │   ├── infer_clip.py       # `pp infer-clip`  (FR-013)
│       │   └── infer_video.py      # `pp infer-video`  (FR-014, FR-015, FR-016)
│       ├── data/                   # 数据准备与划分
│       │   ├── __init__.py
│       │   ├── public_datasets.py  # 拉取/解压公开乒乓球数据集
│       │   ├── splitter.py         # 按 video-id 划分 train/val/test (章程 IV)
│       │   └── list_writer.py      # 生成 PaddleVideo 训练用 list 文件
│       ├── models/                 # 与上游对接的薄封装
│       │   ├── __init__.py
│       │   ├── pp_tsm.py           # PP-TSM 配置/权重加载封装 (FR-012)
│       │   └── registry.py         # 通过 configs/*.yaml 选择模型
│       ├── inference/              # 推理与后处理
│       │   ├── __init__.py
│       │   ├── clip_runner.py      # 单片段推理 (FR-013)
│       │   ├── sliding_window.py   # 长视频滑窗 (FR-014, default w=2s s=1s)
│       │   ├── post_process.py     # 阈值过滤 + 同类合并 (FR-014)
│       │   └── visualizer.py       # MP4 叠加 + JSON 写出 (FR-015)
│       ├── evaluation/             # 评估指标
│       │   ├── __init__.py
│       │   ├── metrics.py          # top-k, per-class, macro-avg (章程 V)
│       │   └── reporter.py         # 写入 experiments/<run>/metrics.json
│       ├── experiment/             # 实验记录
│       │   ├── __init__.py
│       │   └── run_manifest.py     # commit/config_hash/seed/metrics 四元组
│       ├── upstream_adapter/       # 上游适配层 (章程 VI/VIII)
│       │   ├── __init__.py
│       │   ├── importer.py         # 把 third_party/PaddleVideo 加入 sys.path
│       │   ├── trainer.py          # 调用 paddlevideo 训练入口
│       │   └── compat_py311.py     # Python 3.11 兼容补丁 (热修复, 不动上游源码)
│       └── utils/
│           ├── __init__.py
│           ├── config.py           # YAML 配置加载 (章程 III)
│           ├── seeding.py          # 随机种子统一设置 (FR-018)
│           ├── logging.py          # 结构化日志
│           └── env.py              # 解释器/版本/CUDA 自检
│
├── configs/                        # 所有训练/评估/推理配置 (章程 III)
│   ├── datasets/
│   │   └── pingpong_public.yaml    # 数据集元信息 + 类别表 + 路径
│   ├── models/
│   │   └── pp_tsm_pingpong.yaml    # PP-TSM 在乒乓球上的微调配置 (基线)
│   ├── inference/
│   │   └── sliding_window.yaml     # 滑窗参数 (window=2s, stride=1s, threshold)
│   └── examples/
│       └── upstream_smoke.yaml     # 跑通上游示例的最小配置 (US1)
│
├── data/                           # 数据集 (gitignore 大文件; 划分 list 入库)
│   ├── raw/                        # 公开数据集原始文件 (gitignore)
│   ├── clips/                      # 切分后的动作片段 (gitignore)
│   └── splits/                     # 训练/验证/测试 list 文件 (入库, 章程 IV)
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── experiments/                    # 实验输出 (gitignore, 章程 II)
│   └── <YYYYMMDD-HHMMSS-runid>/
│       ├── manifest.json           # commit / config_hash / seed / 起止时间
│       ├── config.yaml             # 该次训练实际使用的配置 (snapshot)
│       ├── log/                    # 训练日志
│       ├── checkpoints/            # 模型权重
│       └── metrics.json            # 测试集 Top-1/Top-5/per-class/macro-avg
│
├── third_party/                    # 上游与补丁 (章程 VI)
│   ├── PaddleVideo/                # Git submodule, 固定到 release/2.2.0 的 commit da9a8ce8 (R1)
│   └── patches/                    # 上游适配补丁 (3.11 兼容 / 必要小修)
│       └── README.md               # 每个补丁需说明原因
│
├── tests/                          # 业务代码测试 (上游模型测试不在此处)
│   ├── unit/
│   │   ├── test_splitter.py        # 验证按 video-id 划分无泄漏 (章程 IV)
│   │   ├── test_post_process.py    # 验证滑窗合并行为
│   │   └── test_config.py          # 验证配置加载与必填项
│   └── integration/
│       ├── test_env_check.py       # 验证 env-check 在 3.11 隔离环境中通过
│       └── test_cli_smoke.py       # 验证 6 个 CLI 命令均可 --help 启动
│
├── requirements/
│   ├── base.txt                    # 业务代码依赖 (固定主版本)
│   ├── upstream-py311.txt          # 上游 PaddleVideo 经过 3.11 适配后的依赖清单 (章程 VIII)
│   └── lock.txt                    # 完整锁文件 (pip-compile 产出, 入库)
│
├── scripts/                        # 一次性脚本 (非 CLI)
│   ├── bootstrap.sh                # 初始化 .venv + 安装依赖 + 拉取 submodule + 应用 patch
│   └── apply_upstream_patches.sh   # 把 third_party/patches/ 应用到 submodule
│
├── pyproject.toml                  # 包定义 + entry point: `pp` → src/pingpong_av/cli/__init__.py
├── .gitignore                      # 含 .venv/, data/raw/, data/clips/, experiments/
├── .gitmodules                     # PaddleVideo submodule 定义
└── README.md                       # 顶层入口 (待 tasks 阶段创建)
```

**结构决策**:
- 选择 **"单一项目 (Option 1) + 严格的 src/third_party 分层"**, 不使用 web/mobile 选项 — 本项目是 CLI 工具 + 业务库, 没有前后端, 没有移动端. 与 spec.md FR-019 的"提供清晰运行入口"一致.
- `src/pingpong_av/` 与 `third_party/PaddleVideo/` 物理分离, 上游通过 submodule 引入, 通过 `upstream_adapter/` 单点接入 — 实现章程 VI (最小侵入).
- `configs/` 是所有可调参数的唯一权威来源 — 实现章程 III (配置驱动).
- `data/splits/*.txt` 入库, 其他大文件 gitignore — 实现章程 IV (数据完整性).
- `experiments/<run_id>/` 目录强制 manifest + config snapshot + metrics — 实现章程 II (可复现实验).
- `requirements/*.txt` 与锁文件入库, `.venv/` gitignore — 实现章程 VIII (隔离环境).

## 复杂度跟踪

> **仅在章程检查有必须证明的违规时填写**

无违规. 本表为空.

| 违规 | 为什么需要 | 拒绝更简单替代方案的原因 |
|------|------------|--------------------------|
| (无) | (无) | (无) |
