# char_pp_prj 开发指南

基于所有功能计划自动生成. 最后更新时间: 2026-05-11

## 活跃技术

- **语言**: Python 3.11 (固定, 由章程 VIII 规定; 任何其他版本都不允许进入主分支)
- **框架**: PaddlePaddle (GPU wheel, 官方支持 3.11 版本) + PaddleVideo (上游, 以 Git submodule 方式接入 `third_party/PaddleVideo/`, 固定到 `release/2.2.0` 分支 commit `da9a8ce8`)
- **基线模型**: PP-TSM + ResNet50 + 8 frames + Kinetics-400 预训练 (在 `configs/models/pp_tsm_pingpong.yaml` 中定义, 通过配置驱动, 禁止硬编码)
- **依赖管理**: `requirements/base.txt` + `requirements/upstream-py311.txt` + `requirements/lock.txt` (全部入库); 隔离环境位于 `.venv/` (gitignore)
- **测试**: pytest (仅覆盖业务代码: 划分无泄漏、滑窗后处理、配置加载、CLI 启动性)
- **CLI 入口**: `pp` (pyproject.toml entry point) → `src/pingpong_av/cli/`
- **数据来源**: PaddleVideo 官方乒乓球动作识别示例数据集 (公开), 类别表由数据集 metadata 派生
- **长视频推理**: 固定窗口 + 步长滑窗 (默认 window=2s, stride=1s, threshold=0.5), 相邻同类合并

## 项目结构

```text
char_pp_prj/
├── .venv/                          # Python 3.11 隔离环境 (gitignore)
├── .specify/                       # speckit 工件
├── specs/001-pingpong-action-recognition/
│   ├── spec.md, plan.md, research.md, data-model.md, quickstart.md
│   └── contracts/cli.md
├── src/pingpong_av/
│   ├── cli/                        # 6 个 CLI 子命令
│   ├── data/                       # 拉取/划分/list 生成
│   ├── models/                     # PP-TSM 等薄封装
│   ├── inference/                  # 片段推理 + 滑窗 + 后处理 + 可视化
│   ├── evaluation/                 # top-k / per-class / macro-avg
│   ├── experiment/                 # manifest 四元组 (commit/config_hash/seed/metrics)
│   ├── upstream_adapter/           # 上游 PaddleVideo 单点接入 + 3.11 兼容 patch
│   └── utils/                      # config / seeding / logging / env
├── configs/
│   ├── datasets/pingpong_public.yaml
│   ├── models/pp_tsm_pingpong.yaml
│   ├── inference/sliding_window.yaml
│   └── examples/upstream_smoke.yaml
├── data/
│   ├── raw/, clips/                # gitignore
│   └── splits/{train,val,test}.txt + *.meta.jsonl   # 入库 (章程 IV)
├── experiments/<run_id>/           # gitignore; 每次训练/评估一个目录
├── third_party/
│   ├── PaddleVideo/                # submodule
│   └── patches/                    # 3.11 兼容补丁
├── tests/unit/, tests/integration/
├── requirements/{base,upstream-py311,lock}.txt
├── scripts/{bootstrap.sh,apply_upstream_patches.sh}
├── pyproject.toml                  # entry point: pp
└── .gitmodules
```

## 命令

全部命令**必须**通过 `.venv/bin/pp` 或激活 `.venv` 后调用; 禁止使用系统 Python (章程 VIII).

```bash
# quickstart 5 条命令 (章程 VII)
.venv/bin/pp env-check --strict
.venv/bin/pp data-prepare --config configs/datasets/pingpong_public.yaml
.venv/bin/pp train        --config configs/models/pp_tsm_pingpong.yaml
.venv/bin/pp eval         --checkpoint experiments/<run_id>/checkpoints/best.pdparams
.venv/bin/pp infer-video  --checkpoint <...> --input <...> --inference-config configs/inference/sliding_window.yaml --output-dir outputs/<...>

# 一次性初始化
git submodule update --init --recursive
bash scripts/bootstrap.sh

# 测试
.venv/bin/pytest tests/unit tests/integration
```

退出码: `0` 成功 · `1` 用户输入错 · `2` 环境问题 · `3` 章程硬约束违反 · `4` 运行时失败.

## 代码风格

- **Python**:
  - 3.11 专属特性可用 (match/case, PEP 646 等), 但**不**需要为 3.10 兼容保留写法
  - 格式化: black (line-length=100), isort, ruff
  - 类型标注: 所有对外函数 (CLI 入口、data-prepare、inference 接口) 必须标注; 内部工具函数建议标注
  - 禁止硬编码业务参数 (章程 III): 路径、类别、超参必须来自 YAML 或环境信息, 不在源码字面量中

## 章程硬约束速查 (对应 `.specify/memory/constitution.md` v1.1.0)

| # | 要点 | 落地位置 |
|---|------|---------|
| I | 规范与计划优先 | 所有代码回溯 FR-xxx |
| II | 可复现实验 | `experiments/<run>/manifest.json` 必含 commit+config_hash+seed+metrics |
| III | 配置驱动 | `configs/` 是业务参数唯一权威来源 |
| IV | 数据完整性 | 划分按 source_video_id, `data/splits/` 入库, test 不做选型 |
| V | 评估纪律 | `eval` 必出 top1/top5/per-class/macro-avg |
| VI | 上游最小侵入 | submodule + `third_party/patches/`, 不复制源码 |
| VII | 端到端 ≤ 5 条命令 | 见上"quickstart" |
| VIII | 隔离 Python 3.11 | `.venv/` + `env-check --strict` + 锁文件入库 |

## 最近变更

- **2026-05-11**: `001-pingpong-action-recognition` — 初始建立完整项目骨架规范 (spec + plan + research + data-model + contracts + quickstart), 确立 Python 3.11 隔离环境 + PaddleVideo submodule + PP-TSM 基线 + 公开乒乓球数据集的技术主线.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
