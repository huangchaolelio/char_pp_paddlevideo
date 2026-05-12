# Quickstart: 基于 PaddleVideo 的乒乓球视频动作识别

**功能**: `001-pingpong-action-recognition`
**目标读者**: 首次 clone 本仓库的工程师 / 研究员
**章程要求**: ≤ 5 条命令完成 "环境验证 → 数据准备 → 训练 → 评估 → 推理 + 可视化" 全流程 (章程 VII),
且整个流程严格在项目隔离的 Python 3.11 环境中运行 (章程 VIII).

---

## 0. 前提 (一次性)

**硬件**:
- Linux (Ubuntu 20.04 / 22.04 验证通过)
- NVIDIA GPU ≥ 12GB 显存 (推荐 3090 / A10 / V100 / A100 级别)
- CUDA 11.8 或 12.x (与下文 `paddlepaddle-gpu` 对应)

**软件**:
- Git ≥ 2.30 (需要 submodule 支持)
- Python **3.11.x** 可执行程序 (可通过 `pyenv` / 系统包 / miniforge 获取, **但不会被直接用来跑项目**)
- ffmpeg (用于 `pp infer-video` 的 MP4 渲染)

**初始化** (仅首次):
```bash
git clone <此仓库 URL>
cd char_pp_prj
git submodule update --init --recursive     # 拉取 PaddleVideo
bash scripts/bootstrap.sh                   # 创建 .venv (3.11) + 安装所有依赖 + 应用 3.11 适配补丁
```

`scripts/bootstrap.sh` 必须做到 (由 tasks 阶段实现):

1. 用 `python3.11 -m venv .venv` 创建隔离环境 (**不复用系统 Python**, 章程 VIII);
2. 在 `.venv` 内 `pip install -r requirements/lock.txt`;
3. `bash scripts/apply_upstream_patches.sh` 应用 `third_party/patches/*.patch` 到 submodule;
4. `.venv/bin/pip install -e third_party/PaddleVideo` 以 editable 方式装入上游;
5. 结束时打印 `.venv/bin/activate` 激活提示.

---

## 端到端 5 条命令

> **激活方式**: 之后所有 `pp` 命令都**必须**通过 `.venv/bin/pp` 或先 `source .venv/bin/activate`.
> 若用系统 `pp` (不存在) 或系统 Python, 立即违反章程 VIII 并被 `env-check` 拦截.

### 命令 1/5 — 环境自检

```bash
.venv/bin/pp env-check --strict
```

**预期输出**: 一行 JSON, `python_version` 为 `3.11.x`, `is_project_venv: true`, `paddle_importable: true`,
`paddlevideo_importable: true`. 若任何一项为 `false` → 退出码 2, 按错误提示修复后重试.

这一步强制执行章程 VIII: 如果不经此步骤, 后续任何 `pp` 命令都可能在错误环境下跑出难以排查的问题.

---

### 命令 2/5 — 数据准备 (公开乒乓球数据集)

```bash
.venv/bin/pp data-prepare --config configs/datasets/pingpong_public.yaml
```

**作用**:
- 按配置中 `source` 自动拉取/解压 PaddleVideo 官方乒乓球动作数据集到 `data/raw/`;
- 按配置中 `split_strategy` 生成 `data/clips/` 下的片段 + `data/splits/{train,val,test}.txt` + `.meta.jsonl`;
- 执行"**无泄漏校验**" (章程 IV): 检查不同划分间的 `source_video_id` 不重叠.

**预期输出**: JSON 摘要, 含每个划分的样本数与类别数.

**失败处理**:
- 数据源不可达 → 退出码 1, 按提示手动放置原始数据到 `data/raw/` 再 `--force` 重试.
- 泄漏校验失败 → 退出码 3, 检查数据标注或 split_strategy 后重试.

这一步是 FR-004/005/006/007 的落地.

---

### 命令 3/5 — 训练 (PP-TSM 基线)

```bash
.venv/bin/pp train --config configs/models/pp_tsm_pingpong.yaml
```

**作用**:
- 创建 `experiments/<YYYYMMDD-HHMMSS>-<sha7>-pp_tsm_pingpong/` 目录;
- 写入 `manifest.json` (commit / config_hash / seed / python_version / GPU 信息);
- 拷贝实际使用的合并配置到 `config.yaml` (snapshot, 章程 II);
- 调用上游 PaddleVideo 训练主循环, 使用官方 Kinetics-400 预训练权重作为初始化 (FR-012);
- 日志写入 `log/train.log`, checkpoint 写入 `checkpoints/`.

**默认 50 epoch**; 可通过在 `configs/models/pp_tsm_pingpong.yaml` 中覆写 `train.epochs` 调整.

**恢复训练**:
```bash
.venv/bin/pp train --config configs/models/pp_tsm_pingpong.yaml \
                   --resume experiments/<run_id>/checkpoints/epoch_30.pdparams
```

**可复现性闸门**: 若 git 工作区脏, 默认拒绝启动 (退出码 3); 明确临时实验可加 `--allow-dirty`, 但这次
实验**不得**作为官方指标基础.

---

### 命令 4/5 — 评估

```bash
.venv/bin/pp eval --checkpoint experiments/<run_id>/checkpoints/best.pdparams
```

**作用**:
- 在 `data/splits/test.txt` 上评估 (章程 IV: 只读 test, 且每个 run 默认只允许一次 `--split test`);
- 输出章程 V 要求的完整指标到 `experiments/<run_id>/metrics.json`:
  `top1`, `top5`, `per_class: {cls: {precision, recall, f1, support}}`, `macro_avg`, `confusion_matrix.png`;
- 结果同步写入 `manifest.json`.

**验收目标** (对齐 spec.md SC-002):
- Top-1 ≥ 70%, Top-5 ≥ 90% (在所选公开数据集上, 类别数 ≥ 5 时).

如低于目标, 查看 `per_class` 与 `confusion_matrix.png` 识别长尾类别; 调整 `configs/models/pp_tsm_pingpong.yaml`
中的采样 / 学习率 / 数据增强重训 —— **不要**反复跑 test 集挑结果 (章程 IV).

---

### 命令 5/5 — 长视频端到端推理 + 可视化

```bash
.venv/bin/pp infer-video \
    --checkpoint experiments/<run_id>/checkpoints/best.pdparams \
    --input data/samples/match01.mp4 \
    --inference-config configs/inference/sliding_window.yaml \
    --output-dir outputs/match01/
```

**作用**:
- 按 `configs/inference/sliding_window.yaml` 的默认 (`window=2s, stride=1s, threshold=0.5`) 做滑窗推理 (FR-014);
- 低于阈值的窗口归为 `"unknown"`, 相邻同类窗口合并 (R5);
- 输出两份产物 (FR-015):
  - `outputs/match01/match01.timeline.json` — 结构化时间轴 (TimelineSegment 数组)
  - `outputs/match01/match01.viz.mp4` — 叠加了动作类别 + 置信度的 MP4

**性能目标** (SC-003): 对 ≤ 10 分钟视频, 整体耗时 ≤ 视频时长 × 2.

**异常健壮性** (FR-016, SC-006): 单个窗口解码/推理失败会被捕获并追加到 JSON 的 `warnings[]`,
不终止整体流程; 仅当失败窗口 > 50% 才退出码 4.

---

## 验证全流程的最小脚本 (smoke test, 可选)

```bash
# 全流程跑一次, 通过退出码判定是否健康
.venv/bin/pp env-check --strict                                            && \
.venv/bin/pp data-prepare --config configs/datasets/pingpong_public.yaml   && \
.venv/bin/pp train --config configs/models/pp_tsm_pingpong.yaml            && \
.venv/bin/pp eval  --checkpoint $(ls -t experiments/*/checkpoints/best.pdparams | head -1) && \
.venv/bin/pp infer-video --checkpoint $(ls -t experiments/*/checkpoints/best.pdparams | head -1) \
                         --input data/samples/match01.mp4 \
                         --inference-config configs/inference/sliding_window.yaml \
                         --output-dir outputs/match01/
```

---

## 常见问题 (快速定位)

| 现象 | 根因 | 修复 |
|------|------|------|
| `pp env-check` 报"解释器不是项目 .venv" | 用了系统 Python 或其他 venv | `source .venv/bin/activate` 或直接用 `.venv/bin/pp` |
| `pp env-check` 报 Python 版本 != 3.11 | `.venv` 用了错误 Python 创建 | `rm -rf .venv && bash scripts/bootstrap.sh` |
| `pp train` 报 `ModuleNotFoundError: paddlevideo` | submodule 未初始化 / 未 editable 安装 | `git submodule update --init --recursive && bash scripts/bootstrap.sh` |
| `pp data-prepare` 报划分泄漏 | 自定义划分把同一源视频分到 train/test | 修正数据标注或切换 `split_strategy` 为 `by_video_ratio` |
| `pp train` 启动时报"工作区脏" | 有未提交修改 | `git status` 确认后: (a) 提交, 或 (b) 明确为临时实验时加 `--allow-dirty` |
| `pp infer-video` MP4 打不开 | ffmpeg 缺失 | `sudo apt install ffmpeg` 后重跑, 或加 `--no-viz` 只出 JSON |

---

## 与章程的对应关系 (供 PR 评审核对)

| 章程原则 | 本 quickstart 的体现 |
|----------|----------------------|
| I. 规范与计划优先 | 每条命令对应 spec.md FR-xxx, 引用已在步骤说明中给出 |
| II. 可复现实验 | 命令 3 创建 Experiment 目录 + manifest 四元组; 工作区脏默认拒绝 |
| III. 配置驱动 | 命令 2/3/5 均以 `--config` 为主参数; 无硬编码超参 |
| IV. 数据完整性 | 命令 2 强制无泄漏校验; 命令 4 仅读 test 且禁止反复跑 |
| V. 评估纪律 | 命令 4 输出完整 top1/top5/per-class/macro-avg |
| VI. 上游兼容与最小侵入 | submodule + editable install + patches/ (bootstrap 阶段) |
| VII. 端到端可运行 | 恰好 5 条命令 |
| VIII. 隔离 Python 环境 | 命令 1 强制自检; 所有命令通过 `.venv/bin/pp` 调用 |
