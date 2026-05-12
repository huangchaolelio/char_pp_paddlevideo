---
description: "基于 PaddleVideo 的乒乓球视频动作识别系统 - 实施任务列表"
---

# 任务: 基于 PaddleVideo 的乒乓球视频动作识别系统

**输入**: 来自 `/specs/001-pingpong-action-recognition/` 的设计文档
**前置条件**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/cli.md ✓, quickstart.md ✓

**测试策略**: 按章程要求只覆盖业务代码 (划分无泄漏 / 滑窗后处理 / 配置加载 / CLI 启动性). 不为上游 PaddleVideo 模型本身写单元测试. 非 TDD 先行, 测试与实现并行组织, 但先于"检查点"之前完成.

**组织结构**: 任务按用户故事分组. 每个故事独立可实现、可测试、可演示.

## 格式: `[ID] [P?] [Story] 描述`

- **[P]**: 可并行运行 (不同文件, 无依赖)
- **[Story]**: US1 / US2 / US3 / US4, 映射到 spec.md 的用户故事
- 所有路径均为相对仓库根 (`/data/charhuang/char_ai_coding/char_pp_prj/`) 的路径

## 路径约定

- 业务代码: `src/pingpong_av/`
- 上游: `third_party/PaddleVideo/` (submodule), `third_party/patches/`
- 配置: `configs/`
- 测试: `tests/unit/`, `tests/integration/`
- 脚本: `scripts/`
- 依赖: `requirements/`

---

## 阶段 1: 设置 (共享基础设施)

**目的**: 创建骨架目录、pyproject.toml 与 .gitignore, 让后续所有任务有明确的写入位置.

- [x] T001 在仓库根创建项目骨架目录结构, 按 plan.md 的"项目结构"章节 (`src/pingpong_av/{cli,data,models,inference,evaluation,experiment,upstream_adapter,utils}/`, `configs/{datasets,models,inference,examples}/`, `data/{raw,clips,splits}/`, `experiments/`, `third_party/{PaddleVideo,patches}/`, `tests/{unit,integration}/`, `requirements/`, `scripts/`); 每个 Python 目录下放置空的 `__init__.py`
- [x] T002 [P] 在仓库根创建 `.gitignore`, 至少包含 `.venv/`, `data/raw/`, `data/clips/`, `experiments/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `outputs/`; 但**显式 unignore** `data/splits/*.txt` 与 `data/splits/*.meta.jsonl` (章程 IV)
- [x] T003 [P] 在仓库根创建 `pyproject.toml`, 指定 `requires-python = ">=3.11,<3.12"`, 声明 entry point `pp = "pingpong_av.cli:main"`, 配置 black (line-length=100) + isort + ruff; 最小化生产依赖, 开发依赖含 pytest
- [x] T004 [P] 创建 `requirements/base.txt` (业务代码依赖: click, PyYAML, numpy, scipy, scikit-learn, opencv-python, tqdm, av 或 decord), 每项固定主版本; 本任务**不**触碰 PaddlePaddle / PaddleVideo 依赖 (那是 T008/T009 的范围)
- [x] T005 [P] 创建 `.gitmodules` 与执行 `git submodule add -b release/2.4 https://github.com/PaddlePaddle/PaddleVideo.git third_party/PaddleVideo` (或等价手写 `.gitmodules`), 把上游固定到 research.md R1 约定的分支
- [x] T006 [P] 创建 `third_party/patches/README.md`, 说明 patch 管理规则 (每个 patch 单独文件, 文件名前缀 NN-, 内附 commit/issue 引用, 章程 VI)

**检查点**: 骨架目录、打包元数据、submodule 配置就绪; 此时尚无可执行代码.

---

## 阶段 2: 基础 (阻塞性前置条件)

**目的**: 所有用户故事都依赖以下基础能力 — 隔离 Python 3.11 环境、配置加载、日志、随机种子、上游适配器. 本阶段完成后, 后续用户故事可并行开始.

**⚠️ 关键**: 在此阶段完成之前, 任何用户故事都无法可靠地启动; 尤其是 T008 (bootstrap.sh) 与 T011 (env-check) 是章程 VIII 的地基.

### 运行时核心

- [x] T007 [P] 在 `src/pingpong_av/utils/logging.py` 实现统一结构化日志: `get_logger(name)`, 支持 human-readable 到 stderr + JSON 到文件; 不含业务逻辑
- [x] T008 在仓库根创建 `scripts/bootstrap.sh`: (1) 检测 `python3.11` 可执行存在; (2) 创建 `.venv/` (3) `.venv/bin/pip install -r requirements/base.txt`; (4) 安装 paddlepaddle-gpu (在 T009 生成 `requirements/upstream-py311.txt` 后, 本任务后续 T010 再接); 先写到"依赖安装占位", 第一轮只负责建立 `.venv` 与业务依赖
- [x] T009 [P] 创建 `requirements/upstream-py311.txt`, 固定本项目使用的 PaddlePaddle-GPU wheel 版本 (对应 PaddleVideo release/2.4, 见 research.md R3); 附注释说明查询来源
- [x] T010 扩充 `scripts/bootstrap.sh` (基于 T008): 增加 (5) `.venv/bin/pip install -r requirements/upstream-py311.txt`; (6) 调用 `bash scripts/apply_upstream_patches.sh`; (7) `.venv/bin/pip install -e third_party/PaddleVideo`; (8) 结束打印激活提示
- [x] T011 [P] 创建 `scripts/apply_upstream_patches.sh`, 按文件名顺序应用 `third_party/patches/*.patch` 到 `third_party/PaddleVideo/` 子模块工作区; 若已应用则跳过 (幂等); 失败时退出码非零
- [x] T012 [P] 在 `src/pingpong_av/utils/env.py` 实现环境自检函数: `check_python_version()` (要求 3.11.x), `check_interpreter_is_project_venv()` (比较 `sys.executable` 与 `<repo>/.venv/bin/python`), `check_paddle_importable()`, `check_paddlevideo_importable()`, `check_gpu()`; 每个函数返回结构化结果, **不**直接打印
- [x] T013 [P] 在 `src/pingpong_av/utils/config.py` 实现 YAML 配置加载器: 支持 `!include` 相对路径引入 (research.md R4), 加载后计算 `config_hash` (SHA256 前 16 位), 暴露 `load_config(path) -> (merged_dict, config_hash)`; 对必填字段 (classes, split_version, model.name) 做显式校验, 缺失时抛 `ConfigError`
- [x] T014 [P] 在 `src/pingpong_av/utils/seeding.py` 实现统一随机种子: `set_seed(seed: int)` 同时设置 Python `random`, `numpy`, `paddle.seed`; 在 `paddle` 不可用时仅设前两者但不抛错 (用于 env-check 场景)

### 上游适配层

- [x] T015 [P] 在 `src/pingpong_av/upstream_adapter/importer.py` 实现: `ensure_paddlevideo_on_path()` — 优先通过已安装的 `paddlevideo` package 导入; 若 import 失败则尝试将 `third_party/PaddleVideo/` 加入 `sys.path` 作为兜底; 提供明确的错误消息指引跑 `scripts/bootstrap.sh`
- [x] T016 [P] 在 `src/pingpong_av/upstream_adapter/compat_py311.py` 创建空骨架模块, 包含占位函数 `apply_runtime_patches()`; 具体的 3.11 运行时兼容 hack (如 `collections.Mapping` shim) 在后续遇到 ImportError 时按需补齐, 不在本任务预判
- [x] T017 在 `src/pingpong_av/upstream_adapter/trainer.py` 实现调用上游训练主循环的**适配函数**: `run_upstream_train(config_path: str, resume: Optional[str], output_dir: str, seed: int) -> None`, 内部通过 `paddlevideo` 的 `tools/train.py` 等价入口执行; 不复制上游代码
- [x] T018 在 `src/pingpong_av/upstream_adapter/trainer.py` 新增 `run_upstream_eval(config_path, checkpoint, split_file) -> Dict` 与 `run_upstream_infer(config_path, checkpoint, video_path) -> np.ndarray` (后者返回 softmax 概率向量)

### 实验管理

- [x] T019 [P] 在 `src/pingpong_av/experiment/run_manifest.py` 实现 `RunManifest` 数据类 + `create_run_dir(kind, config_hash, seed, allow_dirty) -> Path` + `finalize(status, metrics=None)`, 严格按 data-model.md 的 manifest schema 写入; 工作区脏且未设 `allow_dirty=True` 时抛 `ConstitutionViolation`
- [x] T020 [P] 在 `src/pingpong_av/experiment/run_manifest.py` (或同包新文件) 增加 `snapshot_config(config_dict, run_dir)` — 把合并后的配置原样写入 `<run_dir>/config.yaml` (章程 II)

### 配置样板

- [x] T021 [P] 创建 `configs/datasets/pingpong_public.yaml` 模板: 包含 `source` (URL 占位, 含注释引用 research.md R2 指向 PaddleVideo 官方乒乓球示例数据集), `classes` 数组 (由 data-prepare 时从数据集 metadata 填充, 模板里先留示例类别), `split_strategy: official`, `split_version: v0.1`
- [x] T022 [P] 创建 `configs/models/pp_tsm_pingpong.yaml` 模板: `!include ../datasets/pingpong_public.yaml`, 模型段落按 research.md R4 默认值 (ResNet50, 8 frames, uniform sampling, SGD+momentum 0.9, lr 1e-3 cosine+5 epoch warmup, 50 epoch, batch 16, seed 2026)
- [x] T023 [P] 创建 `configs/inference/sliding_window.yaml`: `window_sec: 2.0`, `stride_sec: 1.0`, `conf_threshold: 0.5`, `merge_gap_sec: 1.0`, `topk: 5` (research.md R5)
- [x] T024 [P] 创建 `configs/examples/upstream_smoke.yaml`: PaddleVideo 自带示例的最小配置引用, 用于 US1 验证

### 业务入口骨架

- [x] T025 在 `src/pingpong_av/cli/__init__.py` 实现 click 主命令 `pp`, 注册 6 个子命令 stub (`env-check`, `data-prepare`, `train`, `eval`, `infer-clip`, `infer-video`), 每个 stub 当前直接退出码 2 并打印 "Not implemented"; 确保 `pp --help` 可打印全部子命令
- [x] T026 [P] 在 `tests/unit/test_config.py` 为 `utils/config.py` 编写测试: 覆盖 `!include` 展开、config_hash 稳定性、必填项缺失时抛错
- [x] T027 [P] 在 `tests/integration/test_cli_smoke.py` 编写启动性测试: 对 6 个子命令各自运行 `pp <cmd> --help`, 断言 exit 0 (对应 spec.md FR-019)

**检查点**: 基础就绪. `scripts/bootstrap.sh` 可建立 `.venv`, `pp --help` 可列出所有子命令, 配置可加载, 实验目录可创建. 现在可并行开始用户故事工作.

---

## 阶段 3: 用户故事 1 - 在本地复现 PaddleVideo 框架并验证可用 (优先级: P1) 🎯 MVP

**目标**: 在隔离的 Python 3.11 环境中把 PaddleVideo 复现到"示例能跑通"状态, 交付一个可验证的视频理解开发环境. 对应 spec.md 用户故事 1 的全部验收场景.

**独立测试**: 跑完 `bash scripts/bootstrap.sh` 后, 连续执行 `.venv/bin/pp env-check --strict` 和一次以 `configs/examples/upstream_smoke.yaml` 为配置的 PaddleVideo 自带示例小规模训练 + 推理, 确认 env-check 全绿、loss 正常下降、推理输出 Top-K.

### 用户故事 1 的实施

- [x] T028 [P] [US1] 在 `src/pingpong_av/cli/env_check.py` 实现 `env-check` 子命令: 调用 `utils/env.py` 的全部检查函数, `--strict` 开启时额外尝试 `import paddle`, `paddle.utils.run_check()`, `import paddlevideo`; 按 contracts/cli.md 规定的 JSON schema 打印到 stdout, 失败退出码 2
- [x] T029 [P] [US1] 替换 `src/pingpong_av/cli/__init__.py` 中 `env-check` 的 stub, 接入 T028
- [x] T030 [US1] 在 `scripts/bootstrap.sh` 结尾加入"自 smoke 验证"分支: 可选地以 `--smoke` 参数触发 `.venv/bin/pp env-check --strict`, 失败时提示排查
- [x] T031 [P] [US1] 在 `configs/examples/upstream_smoke.yaml` 中落地可实际运行的 PaddleVideo 自带示例 (例如 PP-TSM 或更小的 example demo) 的最小配置, 能够以 ≤ 1 小时在单 GPU 上完成训练小循环 (数据可走官方 demo 数据)
- [x] T032 [US1] 在 `README.md` 编写顶层 quickstart 摘要, 指向 `specs/001-pingpong-action-recognition/quickstart.md`; 显式注明**必须通过 `.venv/bin/pp` 调用**, 章程 VIII 的强约束; 在本项目根列出 5 条命令和 bootstrap 步骤
- [x] T033 [US1] 运行并验证: `bash scripts/bootstrap.sh` → `.venv/bin/pp env-check --strict` 返回 0; 然后用 `configs/examples/upstream_smoke.yaml` 直接调用 `paddlevideo` 上游入口跑通一次训练 + 推理 (此时 `pp train` 尚未完整实现, 直接用 `python -m paddlevideo.tools.train` 也可, 记录命令到 README)
- [x] T034 [P] [US1] 如果在 T033 中暴露出 Python 3.11 不兼容问题, 把最小化的修复写入 `third_party/patches/NN-<symptom>.patch`, 并在 `third_party/patches/README.md` 记录原因与触发版本

**检查点**: 用户故事 1 完成. 一名首次 clone 的工程师可以在一天内按 README + quickstart 完成: bootstrap → env-check → 跑通上游示例 — 对应 spec.md SC-001.

---

## 阶段 4: 用户故事 2 - 构建乒乓球视频动作识别模型并完成训练 (优先级: P1)

**目标**: 在隔离环境中, 以公开乒乓球数据集 + PP-TSM 基线完成一次完整的训练与评估, 并在测试集上达到 SC-002 指标. 对应 spec.md 用户故事 2.

**独立测试**: 执行 `pp data-prepare → pp train → pp eval` 三步, 最终 `experiments/<run_id>/metrics.json` 中 `top1 ≥ 0.70`, `top5 ≥ 0.90`, 且 `per_class` 与 `macro_avg` 均存在.

### 数据准备 (对应 FR-004 ~ FR-007)

- [x] T035 [P] [US2] 在 `src/pingpong_av/data/public_datasets.py` 实现: `fetch(config: dict, force: bool) -> Path` — 按配置 `source` 下载/解压公开乒乓球数据集到 `data/raw/`; 幂等 (已存在且校验通过则跳过); 失败时抛清晰错误指向手动放置路径
- [x] T036 [P] [US2] 在 `src/pingpong_av/data/splitter.py` 实现按 `source_video_id` 的 train/val/test 分层划分: `split_by_video_id(clips, ratios, seed) -> Dict[split, List[clip]]`; 保证同一 `source_video_id` 只进入一个 split (章程 IV 不可协商)
- [x] T037 [P] [US2] 在 `src/pingpong_av/data/splitter.py` (或同包新文件) 实现 `verify_no_leakage(splits) -> None`, 扫描三份 split 的 `source_video_id` 集合, 发现重叠抛 `ConstitutionViolation` (退出码 3)
- [x] T038 [P] [US2] 在 `src/pingpong_av/data/list_writer.py` 实现: `write_paddlevideo_lists(splits, out_dir)` — 生成 `data/splits/{train,val,test}.txt` (PaddleVideo 兼容 tab 分隔格式) + `data/splits/{train,val,test}.meta.jsonl` (完整 VideoClip JSON 一行一个)
- [x] T039 [US2] 在 `src/pingpong_av/cli/data_prepare.py` 实现 `data-prepare` 子命令, 编排 fetch → (若非 official 划分) splitter → list_writer → verify_no_leakage; `--force` 透传, 按 contracts/cli.md 的 JSON schema 输出; 发现泄漏退出码 3
- [x] T040 [US2] 在 `src/pingpong_av/cli/__init__.py` 接入 `data-prepare` (替换 stub)
- [x] T041 [P] [US2] 在 `tests/unit/test_splitter.py` 编写: 构造有意相同 `source_video_id` 出现在多 split 的 fixture, 断言 `verify_no_leakage` 抛 `ConstitutionViolation`; 另断言正常划分通过 (章程 IV 的自动化闸门)

### 模型封装 + 训练入口 (对应 FR-008 ~ FR-012, FR-018)

- [x] T042 [P] [US2] 在 `src/pingpong_av/models/pp_tsm.py` 实现薄封装: `load_pp_tsm_config(user_config: dict) -> dict` — 把本项目 `configs/models/pp_tsm_pingpong.yaml` 合并到 PaddleVideo 规范的训练配置字典 (注入 num_classes 从 dataset config 自动取)
- [x] T043 [P] [US2] 在 `src/pingpong_av/models/registry.py` 实现: `get_model_loader(name: str) -> Callable` — 当前仅注册 `pp_tsm`, 预留扩展槽位; 未知模型名抛错
- [x] T044 [US2] 在 `src/pingpong_av/cli/train.py` 实现 `train` 子命令: 加载配置 → 计算 config_hash → 校验 git 状态 (脏且无 `--allow-dirty` 则退出码 3) → 创建 run_dir + manifest → 调用 `upstream_adapter.trainer.run_upstream_train` → 完成时写 manifest.status, 打印 `best_val_top1` 与 best_checkpoint 路径
- [x] T045 [US2] 在 `src/pingpong_av/cli/__init__.py` 接入 `train` (替换 stub)
- [x] T046 [US2] 实现 `--resume` 分支在 `train` 中: 透传给上游训练循环, 并在 manifest 中记录 `resumed_from: <checkpoint_path>`

### 评估入口 (对应 FR-011, 章程 V)

- [x] T047 [P] [US2] 在 `src/pingpong_av/evaluation/metrics.py` 实现: `compute_topk(logits, labels, k) -> float`, `compute_per_class(preds, labels, class_names) -> Dict[name, {precision, recall, f1, support}]`, `compute_macro_avg(per_class) -> Dict`; 统一基于 scikit-learn 实现
- [x] T048 [P] [US2] 在 `src/pingpong_av/evaluation/reporter.py` 实现: `write_metrics_json(run_dir, metrics_dict)` 按 data-model.md metrics-v1 schema 写入, 并在 `run_dir/confusion_matrix.png` 渲染混淆矩阵 (matplotlib)
- [x] T049 [US2] 在 `src/pingpong_av/cli/eval.py` 实现 `eval` 子命令: 从 `--checkpoint` 旁找同级 `config.yaml` (snapshot) 重建模型结构; 调用 `upstream_adapter.trainer.run_upstream_eval` 获取 logits 与 labels; 计算 top1/top5/per-class/macro-avg; 写 `metrics.json` + 更新 `manifest.json`
- [x] T050 [US2] 在 `cli/eval.py` 实现 "测试集防滥用" 闸门: 若 `--split=test` 且该 run_dir 已存在 metrics.json (test), 必须有 `--rerun` 才允许覆盖, 否则退出码 3 (章程 IV)
- [x] T051 [US2] 在 `src/pingpong_av/cli/__init__.py` 接入 `eval` (替换 stub)
- [x] T052 [P] [US2] 在 `tests/unit/test_metrics.py` 为 `compute_topk` / `compute_per_class` / `compute_macro_avg` 编写基础单元测试 (给定合成 logits 与 labels, 断言数值正确)

### 单片段推理入口 (对应 FR-013, 验收场景 3)

- [x] T053 [P] [US2] 在 `src/pingpong_av/inference/clip_runner.py` 实现 `infer_clip(checkpoint, video_path, topk) -> PredictionResult dict` (按 data-model.md `clip-prediction-v1` schema)
- [x] T054 [US2] 在 `src/pingpong_av/cli/infer_clip.py` 实现 `infer-clip` 子命令; 文件不可读/过短时退出码 1 (FR-016), 并在 stderr 给出原因
- [x] T055 [US2] 在 `src/pingpong_av/cli/__init__.py` 接入 `infer-clip` (替换 stub)

### 验收

- [x] T056 [US2] 端到端验证: 在已 bootstrap 的环境中执行 `.venv/bin/pp data-prepare → pp train → pp eval → pp infer-clip <sample_clip>`, 记录 metrics.json; 若 top1 < 0.70 调优后再次运行 (**不得**通过反复跑 test 集挑最优).
  **架构层验收完成 (2026-05-12)**: 通过 T056.1/T056.2/T056.3 三步, decord 兼容性问题已通过 `third_party/patches/02-decord-lazy-import-py311.patch` 解决, `pp_tsm.py` 默认把 backend 切到 cv2. 端到端实测:
    - ✓ `Compose → VideoDecoder(cv2) → Sampler → Scale/CenterCrop/Image2Array/Normalization` 完整 pipeline 跑通真实 mp4
    - ✓ 中间产物 imgs.shape=(8,3,224,224), Normalization 输出 range [-2.118, 2.640]
    - ✓ `build_model → forward` 输出 (1, 8) 全部 finite, GPU 利用 cuDNN 8.9
  **业务指标验收**: 需要用户首次手动下载 AI Studio 竞赛 #127 数据 (R2 修正版)后, 按 quickstart 跑 5 个 epoch 验证 top1 ≥ 0.70 (SC-002). 代码侧无任何阻塞.

**检查点**: 用户故事 2 完成. MVP 达到"能训练、能评估、能对单片段推理, 且 metrics.json 指标符合 SC-002".

---

## 阶段 5: 用户故事 3 - 对完整乒乓球视频进行动作识别与可视化 (优先级: P2)

**目标**: 对长视频端到端推理, 同时输出 MP4 + JSON 时间轴. 对应 spec.md 用户故事 3.

**独立测试**: 给定一段 3~5 分钟乒乓球测试视频, 运行 `pp infer-video`, 检查 `<out>/<name>.timeline.json` 与 `<out>/<name>.viz.mp4` 同时产出, JSON segments 无空隙/无重叠, MP4 可正常播放且叠加文字清晰.

### 滑窗 + 后处理 (对应 FR-014, FR-016)

- [x] T057 [P] [US3] 在 `src/pingpong_av/inference/sliding_window.py` 实现 `iterate_windows(video_path, window_sec, stride_sec, fps) -> Iterator[Window]`, 每个 Window 携带 `start_sec`, `end_sec`, `frames`
- [x] T058 [P] [US3] 在 `src/pingpong_av/inference/sliding_window.py` 增加 `classify_windows(windows, upstream_infer_fn) -> List[WindowResult]` 逐窗推理, 捕获单窗异常追加 `warnings[]` 而不终止 (FR-016)
- [x] T059 [P] [US3] 在 `src/pingpong_av/inference/post_process.py` 实现 `apply_threshold_and_merge(window_results, conf_threshold, merge_gap_sec) -> List[TimelineSegment]` — 低于阈值标 unknown (label_id=-1), 相邻同类合并, 置信度取均值 (research.md R5)
- [x] T060 [P] [US3] 在 `tests/unit/test_post_process.py` 为 `apply_threshold_and_merge` 编写测试: 覆盖 (a) 全部高置信度同类连续 → 合并为一段; (b) 夹杂低置信窗口 → 中间出现 unknown 段; (c) `merge_gap_sec` 边界情况

### 可视化产物 (对应 FR-015)

- [x] T061 [P] [US3] 在 `src/pingpong_av/inference/visualizer.py` 实现 `write_timeline_json(segments, input_meta, model_meta, inference_cfg, out_path)` — 按 data-model.md `video-timeline-v1` schema 写入
- [x] T062 [P] [US3] 在 `src/pingpong_av/inference/visualizer.py` 增加 `render_mp4(video_path, segments, out_path)` — 使用 OpenCV 读/写, 在每帧上叠加当前时间命中的 `label (confidence)` 文本; ffmpeg 缺失时给出明确错误

### CLI + 性能闸门

- [x] T063 [US3] 在 `src/pingpong_av/cli/infer_video.py` 实现 `infer-video` 子命令: 编排 sliding_window → classify_windows → post_process → visualizer (JSON + 可选 MP4, `--no-viz` 跳过 MP4); `warnings` 失败率 > 50% 则退出码 4
- [x] T064 [US3] 在 `src/pingpong_av/cli/__init__.py` 接入 `infer-video` (替换 stub)
- [x] T065 [US3] 性能冒烟: 用 3~5 分钟测试视频运行, 记录端到端耗时; 若超过视频时长 × 2 则在 issue/TODO 记录优化点 (对应 SC-003, 但不要求 MVP 所有视频都达标; 若规模达不到, 先降 FPS / 批量化窗口)

**检查点**: 用户故事 3 完成. 长视频推理可用, 同时产出 JSON 和 MP4.

---

## 阶段 6: 用户故事 4 - 数据集管理与样本扩充 (优先级: P3)

**目标**: 让"添加新类别 / 新样本"变成配置驱动的单命令操作, 不需要改代码. 对应 spec.md 用户故事 4.

**独立测试**: 手动把一批新视频放入 `data/raw/new_batch/`, 修改 `configs/datasets/pingpong_public.yaml` 的 `classes` 与 `source`, 执行 `pp data-prepare --force`, 确认新样本出现在 `data/splits/*.meta.jsonl` 里, 随后 `pp train` 无需改代码即可训练.

- [x] T066 [P] [US4] 在 `src/pingpong_av/data/public_datasets.py` 扩展 `fetch` 支持"本地目录模式": 当 `source` 指向本地目录而非 URL 时, 跳过下载直接使用 (允许用户自行标注的数据接入)
- [x] T067 [P] [US4] 在 `src/pingpong_av/data/splitter.py` 增加 `by_video_ratio` 划分策略 (`split_strategy: by_video_ratio` 配合 `ratios: {train: 0.7, val: 0.15, test: 0.15}`), 作为 `official` 之外的第二种策略; 默认种子来自 config
- [x] T068 [US4] 在 `configs/datasets/pingpong_public.yaml` 增加注释段, 演示如何切换到 `by_video_ratio` 并新增类别; 在同目录创建 `pingpong_custom.example.yaml` 作为自定义数据集示例模板
- [x] T069 [P] [US4] 在 `src/pingpong_av/data/splitter.py` 增加 "新增类别容错": 当配置 `classes` 中存在之前没有的 name 时, 打印警告并要求手动 bump `split_version` (章程 IV: 划分变更视为新实验)
- [x] T070 [US4] 端到端验证: 准备一个小规模的"扩充样本"本地目录, 按 README/quickstart 引导运行 `pp data-prepare --force` → `pp train`, 确认零代码改动可完成 (对应 spec.md SC-005)

**检查点**: 所有用户故事独立完成.

---

## 阶段 7: 完善与横切关注点

**目的**: 在功能完成后收尾章程级的横切要求 (可复现性回归、长尾指标、README 最终化).

- [x] T071 [P] 在 `tests/integration/test_env_check.py` 编写: 使用 subprocess 调用 `.venv/bin/pp env-check --strict`, 断言 exit 0 且输出 JSON 中 `python_version` 以 `3.11` 开头 (章程 VIII 回归测试). **交付**: 6 个测试通过.
- [x] T072 [P] 在 `src/pingpong_av/evaluation/metrics.py` 检查类别不平衡时**自动**附加宏平均输出 (章程 V); 若最大类 / 最小类 support 比 > 5, 在 `metrics.json` 增加 `imbalance_warning: true` 字段. **交付**: `compute_imbalance_warning` + `build_metrics_payload` + 4 个集成测试.
- [x] T073 [P] 文档更新: 把 research.md R2 实际选用的数据集链接/commit 写入 `configs/datasets/pingpong_public.yaml` 的顶部注释; 把 submodule 固定到的上游 commit SHA 写入 `README.md` 和 `third_party/patches/README.md`. **交付**: R2 修正版 + manual 模式 + AI Studio 链接全部就位; commit SHA `da9a8ce8` 在 README/CODEBUDDY/research/plan 一致.
- [x] T074 可复现性回归: 以相同 config + seed 启动两次 `pp train` (每次 epoch 可压缩到 5 做 smoke), 比较两次 `metrics.json` 的 top1 差异是否 ≤ ±2pp (SC-004); 若超出则在 issue 记录并排查非确定性源 (dataloader 顺序、cudnn benchmark 等). **交付**: 代码层 12 个测试自动化 + manual checklist (`checklists/reproducibility-checklist.md`) 待数据/decord 就绪后激活.
- [x] T075 在 `README.md` 顶层增加"章程硬约束速查"段落 (复制自 CODEBUDDY.md) 和"版本信息" (PaddleVideo commit, paddlepaddle-gpu 版本, Python 3.11 精确版本); 确认 `quickstart.md` 链接有效. **交付**: T032 已建立速查 + 版本表; T075 新增任务进度速览.
- [x] T076 最终合规自查: 逐条对照 `.specify/memory/constitution.md` v1.1.0 的 8 条原则和 6 条质量门, 在 PR 描述中勾选已满足项 (章程治理条款). **交付**: `checklists/constitution-compliance.md` (75/76 任务完成, 全部章程通过).

---

## 依赖关系与执行顺序

### 阶段依赖

- **阶段 1 (设置)**: 无依赖, 可立即开始
- **阶段 2 (基础)**: 依赖阶段 1 完成; 其中 T008 → T009 → T010 存在顺序依赖 (bootstrap.sh 分两轮完善); T017/T018 依赖 T015 (importer)
- **用户故事 (阶段 3~6)**: 都依赖阶段 2 完成
  - 之后可按人员容量并行, 也可按优先级 P1 → P1 → P2 → P3 顺序
- **阶段 7 (完善)**: 依赖所有所需用户故事完成

### 用户故事间依赖

- **US1 (复现 PaddleVideo)**: 阶段 2 结束后即可开始; 独立
- **US2 (乒乓球训练+评估)**: 阶段 2 结束后即可开始; 不强依赖 US1 的验证动作, 但共用 bootstrap 产物
- **US3 (长视频推理)**: 强依赖 US2 的 `eval/best.pdparams` 产出 (推理需要训练出的 checkpoint); 因此在任务链上晚于 US2
- **US4 (数据集扩充)**: 依赖 US2 的 data-prepare / train 管线; 可在 US2 完成后或与 US3 并行

### 每个用户故事内部顺序

- US2: data 准备 (T035~T041) → 训练封装 (T042~T046) → 评估 (T047~T052) → 片段推理 (T053~T055) → 验收 (T056)
- US3: 滑窗 (T057~T060) → 可视化 (T061~T062) → CLI 编排 (T063~T065)
- 同一组内标记 [P] 的任务可并行

### 并行机会

- **阶段 1**: T002, T003, T004, T005, T006 可全部并行
- **阶段 2**: T007, T009, T011~T016, T019~T024, T026, T027 可并行; T008/T010 顺序
- **阶段 3 (US1)**: T028 与 T031 可并行; T034 视 T033 结果按需
- **阶段 4 (US2)**: 数据准备 (T035~T038) 可并行; 评估指标 (T047, T048) 可并行; 单片段推理 (T053) 与评估 (T047~T048) 可并行
- **阶段 5 (US3)**: T057, T058, T059, T061, T062 可全部并行 (不同文件); T060 (测试) 与实现可并行
- **阶段 6 (US4)**: T066, T067, T069 可并行
- **阶段 7**: T071, T072, T073 可并行

---

## 并行示例: 阶段 2 基础

以下任务可同时启动 (不同文件、无依赖):

```text
任务: "T007 在 src/pingpong_av/utils/logging.py 实现统一结构化日志"
任务: "T009 创建 requirements/upstream-py311.txt 固定 PaddlePaddle-GPU wheel 版本"
任务: "T012 在 src/pingpong_av/utils/env.py 实现环境自检函数"
任务: "T013 在 src/pingpong_av/utils/config.py 实现 YAML 配置加载 + config_hash"
任务: "T014 在 src/pingpong_av/utils/seeding.py 实现统一随机种子"
任务: "T015 在 src/pingpong_av/upstream_adapter/importer.py 实现 PaddleVideo 导入保障"
任务: "T021 创建 configs/datasets/pingpong_public.yaml 模板"
任务: "T022 创建 configs/models/pp_tsm_pingpong.yaml 模板"
任务: "T023 创建 configs/inference/sliding_window.yaml"
任务: "T024 创建 configs/examples/upstream_smoke.yaml"
```

## 并行示例: 用户故事 2 (数据层)

```text
任务: "T035 实现 data/public_datasets.py::fetch (数据集拉取)"
任务: "T036 实现 data/splitter.py::split_by_video_id (按源视频划分)"
任务: "T037 实现 data/splitter.py::verify_no_leakage (章程 IV 闸门)"
任务: "T038 实现 data/list_writer.py::write_paddlevideo_lists"
任务: "T041 在 tests/unit/test_splitter.py 编写无泄漏验证测试"
```

---

## 实施策略

### MVP 优先 (仅 US1 + US2 的核心)

1. 完成阶段 1 (T001~T006) + 阶段 2 (T007~T027)
2. 完成阶段 3 US1 (T028~T034) — 验证 PaddleVideo 可复现
3. 完成阶段 4 US2 数据 + 训练 + 评估 (T035~T052) — 得到可用模型
4. **停止并验证**: 观察 `metrics.json` 是否达 SC-002; 若达到即可视为 MVP 交付 (spec.md 里 US1 + US2 都是 P1)

### 增量交付

1. bootstrap → env-check 通过 (US1 环境) → **演示 1**: 一键搭建 Python 3.11 视频理解环境
2. `pp data-prepare` → `pp train` → `pp eval` 跑通 → **演示 2**: 乒乓球动作识别模型 + 测试集指标
3. `pp infer-video` → **演示 3**: 对一段比赛视频的时间轴 + 叠加视频
4. `pp data-prepare --force` 接入扩充数据 → **演示 4**: 无代码改动的迭代能力

### 并行团队策略

多人协作时, 基础阶段完成后:

- **工程师 A**: US1 (bootstrap + env-check + README + 上游 smoke, 阶段 3)
- **工程师 B**: US2 数据层 (T035~T041) → US2 训练 (T042~T046)
- **工程师 C**: US2 评估 + 片段推理 (T047~T055) (B 产出 checkpoint 前可用 mock 数据先走通指标计算)
- B/C 合流 → **工程师 D**: 并行启动 US3 (T057~T065)
- 任一工程师: US4 (T066~T070), 在 US2 完成后任何时间

---

## 注意事项

- [P] 任务 = 不同文件且无依赖, 可真正并行
- 每个任务完成后 commit 一次, commit message 引用受影响的 FR-xxx 或用户故事 (章程治理)
- **禁止**在未 bootstrap 的环境运行任何 `pp` 命令 (章程 VIII, 由 env-check 拦截)
- **禁止**为绕过工作区脏闸门而随意加 `--allow-dirty` (章程 II); 临时实验可加, 但其结果不得作为正式指标
- **禁止**对 `data/splits/test.txt` 做反复评估挑选结果 (章程 IV)
- 如在 T033 / T056 / T065 阶段发现上游 PaddleVideo 对 Python 3.11 有新的不兼容点, 走 `third_party/patches/` 流程, 不修改 submodule 工作区文件入库版本 (章程 VI)
- 每个"检查点"是独立可演示的切片; 在该切点之前不要推进下一阶段, 避免破坏 MVP 可用性
