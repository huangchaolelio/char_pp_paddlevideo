# char_pp_prj — 基于 PaddleVideo 的乒乓球视频动作识别

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Constitution](https://img.shields.io/badge/constitution-v1.1.0-green.svg)](.specify/memory/constitution.md)

> **目标**: 在隔离的 Python 3.11 环境中以 Git submodule 形式集成
> [PaddlePaddle/PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo), 构建一套
> **配置驱动 + 可复现 + 端到端可运行**的乒乓球视频动作识别系统.

详细规范见 [`specs/001-pingpong-action-recognition/spec.md`](specs/001-pingpong-action-recognition/spec.md);
完整 quickstart 见 [`specs/001-pingpong-action-recognition/quickstart.md`](specs/001-pingpong-action-recognition/quickstart.md).

---

## TL;DR — 5 条命令端到端

> **章程 VIII 强约束**: 所有命令必须通过 `.venv/bin/pp` 或激活 `.venv` 后调用,
> **禁止**使用系统 Python 或共享 conda 环境.

```bash
# 一次性初始化 (创建 .venv + 安装依赖 + 拉取 PaddleVideo submodule + 应用 3.11 兼容补丁)
git submodule update --init --recursive
bash scripts/bootstrap.sh --smoke

# quickstart 5 条命令
.venv/bin/pp env-check    --strict
.venv/bin/pp data-prepare --config configs/datasets/pingpong_public.yaml
.venv/bin/pp train        --config configs/models/pp_tsm_pingpong.yaml
.venv/bin/pp eval         --checkpoint experiments/<run_id>/checkpoints/best.pdparams
.venv/bin/pp infer-video  --checkpoint <...> --input <...> \
                          --inference-config configs/inference/sliding_window.yaml \
                          --output-dir outputs/<...>
```

---

## 立即上手: 上游官方乒乓球样例推理 (US5, 不需要训练数据)

> 如果你想在 1 分钟内看到完整推理路径工作 (无需注册 AI Studio 下载训练数据):

```bash
# 一次性下载 380MB 权重 (BCEBOS, 真公开)
mkdir -p data/raw/pingpong_public/checkpoints
curl -fL -o data/raw/pingpong_public/checkpoints/VideoSwin_tennis.pdparams \
  https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_tennis.pdparams

# 7.4MB 样例 pkl 已在 `pp data-prepare` 时自动下载到 data/raw/pingpong_public/smoke/
# 直接推理:
.venv/bin/pp infer-pkl \
  --pkl data/raw/pingpong_public/smoke/example_tennis.pkl \
  --checkpoint data/raw/pingpong_public/checkpoints/VideoSwin_tennis.pdparams \
  --topk 5
```

**预期 (SC-007)**: Top-1 = `动作7`, 置信度 ≥ 0.99, 与 pkl 内 GT 一致.

> 这条路径用的是上游官方 **VideoSwin TableTennis** 模型 (`SwinTransformer3D + I3DHead`, 8 类), 与本项目 PP-TSM 业务主线并行存在. 详见 [research.md R7](specs/001-pingpong-action-recognition/research.md).

---

## 端到端真实业务训练: BMN 时序定位 (US6, 通过私有 COS)

> 在团队 COS bucket 已有 AI Studio 竞赛 #127 数据 (43.5GB PP-TSN 特征 + 14 类时序标注) 的前提下:

```bash
# 1) .env 中需含 COS 凭据 (REGION/BUCKET/SECRET_ID/SECRET_KEY/VIDEO_PREFIX)

# 2) pp data-prepare 自动从 COS 拉取 + 流式解压 + 写 splits
.venv/bin/pp data-prepare --config configs/datasets/pingpong_competition_bmn.yaml

# 3) 转换为上游 BMN 期望的 .npy 滑窗 + label_fixed/label_gts.json
.venv/bin/python scripts/prepare_bmn_inputs.py

# 4) 训练 (上游 BMN, 14 类时序定位)
.venv/bin/pp train --config configs/models/bmn_pingpong.yaml --allow-dirty
```

**预期 (SC-008 架构验收)**: 训练循环成功启动, GPU 100% 利用, loss 在前 1000 step 内从 ~1.77 降到 ~1.5 以下. 完整 20 epoch 训练 (上游推荐) 在 T4 上预计 ~24 小时.

> 这条路径用的是上游 **BMN (Boundary-Matching Network) + BMNLoss**, 输入是预提取的 PP-TSN feature (2048-d), 输出时序候选区间. 与 PP-TSM 主线 (US2, 视频分类) **互补共存**, 不替换.
> 14 个动作类别: 摆短 / 拉 / 控制 / 侧身拉 / 劈长 / 拧 / 挑 / 侧旋 / 转不转 / 中性 / 勾球 / 普通 / 逆旋转 / 下蹲. 详见 [research.md R8](specs/001-pingpong-action-recognition/research.md).

---

## 项目结构

```text
char_pp_prj/
├── .venv/                          # Python 3.11 隔离环境 (gitignore, 章程 VIII)
├── .specify/                       # 规范驱动开发工件 (speckit)
├── specs/001-pingpong-action-recognition/
│   ├── spec.md, plan.md, research.md, data-model.md, quickstart.md
│   ├── tasks.md
│   └── contracts/cli.md            # `pp` CLI 唯一对外契约
├── src/pingpong_av/                # 业务代码 (与上游严格隔离, 章程 VI)
│   ├── cli/                        # 6 个 `pp` 子命令
│   ├── data/, models/, inference/, evaluation/, experiment/, utils/
│   └── upstream_adapter/           # PaddleVideo 单点接入 + 3.11 兼容
├── configs/                        # 训练/评估/推理配置 (章程 III, 唯一权威来源)
├── data/
│   ├── raw/, clips/                # gitignore (大文件)
│   └── splits/{train,val,test}.txt + *.meta.jsonl   # **入库** (章程 IV)
├── experiments/<run_id>/           # gitignore; 每次训练/评估的实验目录
├── third_party/
│   ├── PaddleVideo/                # Git submodule, 锁定到 release/2.2.0 commit da9a8ce8
│   └── patches/                    # 3.11 兼容补丁 (上游不动入库版本, 章程 VI)
├── tests/{unit,integration}/       # 仅业务代码测试 (上游模型测试不在此处)
├── requirements/{base,upstream-py311,lock}.txt
├── scripts/{bootstrap.sh,apply_upstream_patches.sh}
├── pyproject.toml                  # entry point: `pp`
├── CODEBUDDY.md                    # AI 助手开发指引
└── .gitmodules
```

---

## 版本锁定

| 组件 | 版本 |
|------|------|
| Python | **3.11.x** (锁死, 章程 VIII; 不接受 3.10 / 3.12) |
| PaddlePaddle-GPU | `2.6.2` (官方首版支持 3.11 的稳定 GPU wheel) |
| PaddleVideo | `release/2.2.0` 分支 commit `da9a8ce8` |
| 业务依赖 | 见 [`requirements/base.txt`](requirements/base.txt) (固定主版本) |
| 上游适配依赖 | 见 [`requirements/upstream-py311.txt`](requirements/upstream-py311.txt) (override 上游钉死的旧 wheel) |

**升级版本的方式**: 修改 `.gitmodules` / requirements + 重新生成 `requirements/lock.txt` + 重跑全套
quickstart + 在 PR 中说明影响. 请参考章程 VI / VIII.

---

## 章程硬约束速查 (`.specify/memory/constitution.md` v1.1.0)

| # | 要点 | 落地位置 |
|---|------|---------|
| I | 规范与计划优先 | 所有代码回溯 FR-xxx; tasks.md 严格分阶段 |
| II | 可复现实验 | `experiments/<run>/manifest.json` 必含 commit + config_hash + seed + metrics |
| III | 配置驱动 | `configs/` 是业务参数唯一权威; 源码零硬编码 |
| IV | 数据完整性 | 划分按 `source_video_id`, `data/splits/` **入库**, test 不做反复挑选 |
| V | 评估纪律 | `eval` 必出 top1 / top5 / per-class / macro-avg |
| VI | 上游最小侵入 | submodule + `third_party/patches/`, 不复制源码 |
| VII | 端到端 ≤ 5 条命令 | 见上 quickstart |
| VIII | 隔离 Python 3.11 | `.venv/` + `pp env-check --strict` + 锁文件入库 |

---

## 项目进度 (live)

> 任务总览见 [`specs/.../tasks.md`](specs/001-pingpong-action-recognition/tasks.md); 章程合规性自查见 [`checklists/constitution-compliance.md`](specs/001-pingpong-action-recognition/checklists/constitution-compliance.md).

| 阶段 | 任务 | 完成 | 说明 |
|------|------|------|------|
| 1 设置 | T001–T006 | ✅ 6/6 | 骨架 / pyproject / submodule / patches |
| 2 基础 | T007–T027 | ✅ 21/21 | utils / upstream_adapter / experiment / configs / CLI 骨架 |
| 3 US1 复现 | T028–T034 | ✅ 7/7 | env-check 全绿 + smoke 模型 build_model + forward 通过 |
| 4 US2 训练 | T035–T056 | ✅ 22/22 | **架构全通**; T056 端到端实测; 业务指标 SC-002 待 AI Studio 数据 |
| 5 US3 长视频 | T057–T065 | ✅ 9/9 | 滑窗 / 后处理 / 可视化 / 性能 0.01x ≤ 2x SC-003 |
| 6 US4 数据扩充 | T066–T070 | ✅ 5/5 | local_dir / 类别表 sentinel / 端到端实测 |
| 7 完善 | T071–T076 | ✅ 6/6 | 100 测试 + 章程合规自查 |
| **8 US5 上游样例** | **T077–T080** | **✅ 4/4** | **VideoSwin TableTennis + pkl 推理; SC-007 实测 Top-1 0.9999 命中 GT** |
| **9 US6 私有 COS + BMN** | **T101–T106** | **✅ 6/6** | **腾讯云 COS 接入 + 43.5GB 数据集; BMN 时序定位训练 (loss 1.77→0.81 in 1050 steps); SC-008 架构验收 ✓** |

**测试**: 100/100 通过 (单元 + 集成).
**业务代码**: ~5000 行 (含本项目业务) + 4 个上游 patch (~300 行).
**MVP 架构完成度**: **86/86 任务 = 100%**. SC-007 + SC-008 实测验收; SC-002 (PP-TSM top1 ≥ 70%) 待用户原始视频数据.

---

## 开发流程 (speckit)

新功能必须按以下顺序推进, 跳步必须在 PR 描述中说明理由 (章程治理条款):

```
/speckit.specify  → spec.md
/speckit.clarify  → 解决高影响不确定性 (可选)
/speckit.plan     → plan.md + research.md + data-model.md + contracts/ + quickstart.md
/speckit.tasks    → tasks.md
                  → 实施 (按 P1 → P2 → P3 顺序)
                  → 评估 (按章程 V)
                  → 合并
```

每个 PR 必须:
- 列出受影响的章程原则编号 (例如 "对应 III, IV, VIII");
- 引用至少一个 FR-xxx 或用户故事 (US1~US4);
- 工作区脏 → 实验目录里追加 `--allow-dirty` 注记 + 该次结果**不得**作为正式指标 (章程 II).

---

## 退出码约定 (`pp` CLI 全局)

| code | 含义 | 常见触发 |
|------|------|---------|
| 0 | 成功 | — |
| 1 | 用户输入错误 | 文件不存在 / 参数非法 / 视频不可读 |
| 2 | 环境问题 | 解释器非项目 .venv / Python ≠ 3.11 / paddle 不可导 |
| 3 | 章程硬约束违反 | 工作区脏未加 `--allow-dirty` / 划分泄漏 / 测试集重复评估未加 `--rerun` |
| 4 | 运行时失败 | 训练发散 / 超过阈值的推理失败 |

完整契约见 [`specs/001-pingpong-action-recognition/contracts/cli.md`](specs/001-pingpong-action-recognition/contracts/cli.md).

---

## 运行测试

```bash
# 业务代码测试 (单元 + 集成); 不依赖 paddle
.venv/bin/pytest tests/unit tests/integration

# 仅集成测试
.venv/bin/pytest tests/integration -v
```

测试范围严格限定在: 划分无泄漏 / 滑窗后处理 / 配置加载 / CLI 启动性. **不**为 PaddleVideo
上游模型本身写单元测试 (上游负责).

---

## 上游 PaddleVideo Smoke 测试 (US1)

在 P1 用户故事 1 中, 我们用上游官方示例验证环境健康度 (不需要乒乓球数据):

```bash
.venv/bin/pp env-check --strict     # 必须先全绿
# 直接调用上游 main.py (绕过 pp train, 因为 pp train 在 T044 才完整接通)
.venv/bin/python third_party/PaddleVideo/main.py \
    -c third_party/PaddleVideo/configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
    --validate \
    -o epochs=1 -o DATASET.batch_size=2 -o DATASET.test_batch_size=2 \
    -o DATASET.num_workers=0 -o log_interval=1
```

具体可调参数与产物见 [`configs/examples/upstream_smoke.yaml`](configs/examples/upstream_smoke.yaml).

如在 Python 3.11 下报 `ImportError` / 类似兼容性错误:
1. **不要**修改 `third_party/PaddleVideo/` 内的文件 (那是 submodule, 章程 VI 禁止);
2. 在 `third_party/patches/` 下新增一个最小 `.patch` 文件 (规范见 [其 README](third_party/patches/README.md));
3. 重新运行 `bash scripts/bootstrap.sh`, 让 `apply_upstream_patches.sh` 把补丁应用到 submodule 工作区.

---

## 许可证 / 致谢

- 本仓库代码: **Apache License 2.0**
- 上游 [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo): Apache License 2.0 (随 submodule 保留原 LICENSE)
- 数据: 公开乒乓球动作数据集, 由 `pp data-prepare` 按需引导. 本仓库**不**打包数据.
  - PaddleVideo 上游的官方乒乓球数据集仅通过 [百度 AI Studio 竞赛 #127](https://aistudio.baidu.com/aistudio/competition/detail/127/0/introduction) 分发, **需用户注册**后下载并放到 `data/raw/pingpong_public/`, 在该目录创建空 `.ready` 哨兵文件后即可继续. 详见 [research.md R2 修正版](specs/001-pingpong-action-recognition/research.md).
  - 自定义数据 (US4 场景): 见 [`pingpong_custom.example.yaml`](configs/datasets/pingpong_custom.example.yaml), `source.type: local_dir` 模式.

---

## 进一步阅读

- [章程 (Constitution)](.specify/memory/constitution.md)
- [功能规范 (Spec)](specs/001-pingpong-action-recognition/spec.md)
- [技术研究 (Research)](specs/001-pingpong-action-recognition/research.md)
- [实施计划 (Plan)](specs/001-pingpong-action-recognition/plan.md)
- [数据模型 (Data Model)](specs/001-pingpong-action-recognition/data-model.md)
- [CLI 契约 (CLI Contract)](specs/001-pingpong-action-recognition/contracts/cli.md)
- [Quickstart](specs/001-pingpong-action-recognition/quickstart.md)
- [任务列表 (Tasks)](specs/001-pingpong-action-recognition/tasks.md)
