# 阶段 0 研究: 基于 PaddleVideo 的乒乓球动作识别

**功能**: `001-pingpong-action-recognition`
**输入**: [spec.md](./spec.md), [plan.md](./plan.md), [constitution v1.1.0](../../.specify/memory/constitution.md)
**目的**: 解决 `plan.md` 中所有遗留的决策点, 为阶段 1 (data-model / contracts / quickstart) 扫清障碍.

---

## R1 — 上游 PaddleVideo 版本锁定 (submodule tag)

**Decision**: 以 PaddleVideo 最新可用的 release 分支 **`release/2.2.0`** 的尾部 commit
**`da9a8ce8f0beba020727c7e7b6c50308d32df76f`** 作为 submodule 锁定版本; 该 commit 在上游被打了
`v2.1.0-1412-g...` 的 describe, 是 `release/2.2.0` 分支当前的 HEAD.

**Rationale**:
- 通过 `git ls-remote --heads https://github.com/PaddlePaddle/PaddleVideo.git` 实地查询, 上游公开
  release 分支只到 `release/2.2.0` (`release/2.0.0`, `release/2.1.0`, `release/2.1.1`, `release/2.2.0`),
  不存在 `release/2.3.x` 或 `release/2.4.x`. 早期规划文档基于"假设最新"猜测了 `release/2.4`,
  本节按上游真实可用版本予以修正.
- `release/2.2.0` 是上游目前最新的稳定 release 分支, 模型库齐全 (PP-TSM / PP-TSN / SlowFast /
  TimeSformer 等), 覆盖本项目所有潜在扩展.
- 选择 `release/` 前缀分支而非 `develop`, 避免上游非预期变动影响本项目可复现性 (章程 II).
- 固化到具体 commit SHA 是章程 VI 的强要求 ("可审计的 tag 或 commit").

**Alternatives considered**:
- `develop` 最新分支: 被拒 — 变动频繁, 破坏可复现性.
- `release/2.1.1` 较老版本: 被拒 — 早期版本的 Paddle/CUDA 组合与当前主流硬件和 Python 3.11 适配
  风险更高, 且模型支持面更窄.
- Fork 到本组织再引: 被拒 — 本项目预计对上游的修改都是 3.11 兼容性 patch 级, 用 `third_party/patches/`
  足够, 没必要承担 fork 维护成本.

**输出给阶段 1 / tasks**:
- `.gitmodules` 指向 `https://github.com/PaddlePaddle/PaddleVideo.git`, `branch = release/2.2.0`
- submodule HEAD 固定到 `da9a8ce8f0beba020727c7e7b6c50308d32df76f`; bootstrap 脚本与 README/quickstart
  必须显式记录该 SHA.

---

## R2 — 公开乒乓球动作数据集选型

> ⚠ **修正版 (2026-05-12, 实施阶段重要发现)**. 原决议见本节末"原始决议"; 修正原因:
> 实地调研发现 PaddleVideo `release/2.2.0` **不提供**任何直接公开下载的乒乓球训练数据集.

**Decision (修正版)**: 采用 **`source.type: manual` 模式** 把数据集来源声明为
"AI Studio 竞赛 #127 (半公开)", 由用户首次注册 + 报名 + 手动下载到 `data/raw/pingpong_public/`,
通过哨兵文件 `.ready` 通知 `pp data-prepare` 数据已就绪.

**Rationale**:
- 上游 `applications/TableTennis/ActionRecognition/README.md` 的 "数据准备" 段落是字面 `TODO`;
  develop 分支同样未补充. PaddleVideo 把用户引向 AI Studio 竞赛页 (需注册账号 + 报名才能下载).
- BCEBOS 上 (实地 200 OK 验证) 仅有训练好的模型权重 + 7.4MB 单样例 pkl, 没有完整训练集.
- 章程 II / IV 的核心闸门 (commit / config_hash / seed / split-version 四元组, 划分按 video_id,
  splits 入库) 与数据获取方式无关, 仍然完整保留; 只是首次准备多了一个手动注册步骤.
- 章程 III "配置驱动" 也得到尊重: `manual` 模式仍由 yaml 描述, 不在源码硬编码 URL 或 AI Studio 链接.
- 比"编造一个看似官方但实际 404 的 URL"的诱惑更诚实可维护, 也避免给后续维护者埋坑.

**实地验证的可达 URL**:

| URL | 大小 | 用途 |
|-----|------|------|
| `https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_k400.pdparams` | — | K400 预训练权重 |
| `https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_tennis.pdparams` | 380 MB | **乒乓球训练好的模型** (上游 release/2.2 出品) |
| `https://videotag.bj.bcebos.com/Data/example_tennis.pkl` | 7.4 MB | **单样例 pkl, 仅推理用**; 已配置为 `source.smoke_sample.url` 自动下载, 用于 `pp infer-clip` 端到端 smoke |

**Alternatives considered (修正后)**:
- **OpenTTGames (OSAI)**: 已逐一验证可下载, 但任务是球检测/分割/事件检测, 与 PP-TSM 8 类动作分类
  基线**类别不匹配**; 接入需要重写 adapter + 重新定义类别, 可能违反 spec.md 范围 → 被拒.
- **Kinetics-400 子集 (`playing_table_tennis`)**: 真公开可下载, 但样本数极少, 仅供 SC-001 冒烟,
  不达 SC-002 (top1 ≥ 70%) → 被拒作主线, 但保留作为未来低保底方案.
- **编造一个看似来自 BCEBOS 的训练数据 URL**: 强烈被拒 — 不诚实且不可维护.
- **通过 fork PaddleVideo 自建公开镜像**: 被拒 — 涉及上游数据许可, 风险/收益不成比例.

**输出给阶段 1 / tasks**:
- `configs/datasets/pingpong_public.yaml` 的 `source.type` 改为 `manual`,
  含 `origin_url`, `requires_registration`, `sentinel_relpath`, `manual_steps`, `smoke_sample`.
- `src/pingpong_av/data/public_datasets.py` 实现 `_ensure_manual_setup` 与 `DatasetNeedsManualSetup` 异常.
- `cli/data_prepare.py` 把 `DatasetNeedsManualSetup` 映射为退出码 1 + 多行 stderr 引导.
- 章程合规: 划分仍按 `source_video_id` (章程 IV); manifest 四元组不变 (章程 II).

---

### R2 原始决议 (已被上述修正版取代, 仅供溯源)

~~**Decision**: 采用 **PaddleVideo 官方 "乒乓球动作识别" 示例所使用的数据集** 作为默认数据源~~

~~**Rationale**: 与本项目"复现 PaddleVideo 并做乒乓球动作识别"的主线高度契合, 官方示例即自带训练/验证/测试划分与元信息文件, 许可证跟随 PaddleVideo 项目本身的开源协议.~~

**已被取代的原因**: 在 T056-T070 实施阶段 (2026-05-12) 通过实地查看 submodule 文档 + WebFetch
上游 README + 探测 BCEBOS 命名模式, 确认上游**没有**直接发布乒乓球训练数据的公开 URL.
原决议的"自带训练/验证/测试划分"假设不成立, 因此修正为 `manual` 模式如上.

---

## R3 — Python 3.11 + PaddlePaddle / PaddleVideo 兼容性策略

**Decision**: 采用**"官方 wheel 优先 + 最小化兼容补丁"**策略:
1. PaddlePaddle 安装使用**官方发布的支持 Python 3.11 的 GPU wheel** (通过官方 pip 镜像,
   `paddlepaddle-gpu==<对应 release/2.4 兼容的版本>`). **不**从源码编译.
2. PaddleVideo 本身作为 submodule 引入, 以**可编辑安装**的方式接入 `.venv` (`pip install -e third_party/PaddleVideo`),
   由 `src/pingpong_av/upstream_adapter/importer.py` 负责把路径加入 `sys.path`.
3. 上游若存在 Python 3.11 不兼容的点 (常见点: `collections.Mapping` / `distutils` 移除 /
   `numpy.bool` alias), 通过 `third_party/patches/*.patch` 在 bootstrap 阶段应用, **不修改** submodule
   内的文件入库版本.
4. 锁文件 `requirements/lock.txt` 由 `pip-compile` 在 Python 3.11 解释器下生成并入库.

**Rationale**:
- 章程 VIII 明确禁止降级 Python; 官方 wheel 是最稳、最可复现的获取方式, 避免因 CUDA/CuDNN 编译
  问题发散.
- 可编辑安装让上游代码既"物理隔离" (在 `third_party/`), 又"逻辑可用" (在 `.venv` 里能 import),
  与章程 VI 一致.
- Patch-from-outside 避免污染 submodule 工作区, 升级上游只需重新 apply 已有 patch 或丢弃已过时的 patch.

**Alternatives considered**:
- 直接 `pip install paddlevideo`: 被拒 — 目前 PyPI 上的 `paddlevideo` 包不是官方维护主仓库, 且无法
  满足"固定到某个 commit"的可复现性要求 (章程 II).
- 直接修改 submodule 内文件: 被拒 — submodule 在 git 视角是其他仓库的一部分, 直接改会污染工作区、
  升级时产生冲突, 违反章程 VI 的"最小侵入".
- 使用 conda / mamba 而非 venv: 被拒 — conda 会引入额外工具链与环境耦合, 与"纯 pip + venv"相比
  对本项目 CLI 场景没有净收益, 且 `.venv` + `requirements/*.txt` 已足够锁定.

**已知风险与缓解**:
- **风险**: PaddlePaddle 某些版本与 Python 3.11 的 wheel 可能滞后于最新 minor 版本.
  **缓解**: 在 bootstrap 脚本中 `pip install` 时显式 pin 到 3.11 已有 wheel 的版本, 并把该版本记
  入 `requirements/upstream-py311.txt`.
- **风险**: decord 在 Python 3.11 上 wheel 缺失的历史情况.
  **缓解**: 优先用 `av` (PyAV) 或 OpenCV 作为视频读取后备; 若 PaddleVideo 代码路径强制 decord,
  则在 patch 中替换导入 (最小侵入式修复).

**输出给阶段 1 / tasks**:
- `requirements/base.txt` + `requirements/upstream-py311.txt` + `requirements/lock.txt`
- `scripts/bootstrap.sh`: 创建 `.venv` (3.11) → 安装业务依赖 → 安装 paddlepaddle-gpu → 拉取 submodule →
  `apply_upstream_patches.sh` → `pip install -e third_party/PaddleVideo`
- `src/pingpong_av/cli/env_check.py`: 验证 `sys.version_info[:2] == (3, 11)` 且解释器路径在项目
  `.venv/` 内; 失败则退出非零并打印修复指引.

---

## R4 — 基线模型 PP-TSM 的具体配置

**Decision**: 采用 PaddleVideo 官方 **PP-TSM + ResNet50 + 8 frames + Uniform sampling + Kinetics-400
pretrained** 作为乒乓球微调基线, 类别头 (`num_classes`) 由 `configs/datasets/pingpong_public.yaml`
中的类别数自动注入, 不硬编码.

关键默认值 (均可在 `configs/models/pp_tsm_pingpong.yaml` 中覆写):
- 骨干: ResNet50 (imagenet 预训练 + Kinetics-400 上的 PP-TSM 权重初始化分类头前部分)
- 帧数 / 采样: 8 frames, uniform sampling, 短边 resize 256, center crop 224
- 优化器: Momentum(0.9) + WeightDecay(1e-4)
- 学习率: 初始 1e-3, cosine annealing, 5 epoch warmup
- 训练轮数: 50 epoch (可通过配置覆写)
- Batch size: 16 (单卡 ≥ 12GB 显存) / 32 (24GB 显存)
- 随机种子: `configs/models/pp_tsm_pingpong.yaml` 中显式设置 (默认 2026), 通过 `seeding.py` 统一注入

**Rationale**:
- PP-TSM + ResNet50 是 PaddleVideo 中显存/速度/精度平衡最好的官方标配, 对乒乓球这种相对短 (≤ 5s)
  的动作片段尤其合适 (8 frames uniform 覆盖一个击球完整动作).
- 继承 Kinetics-400 预训练能显著缓解小数据过拟合, 符合 FR-012.
- 参数默认值是上游官方 README 和示例配置中的默认, 降低调参发散风险.

**Alternatives considered**:
- **16 frames**: 被拒作默认 — 显存翻倍, 对 "动作 ≤ 3s" 的乒乓球片段边际收益有限; 保留为可配置选项.
- **TSN 而非 TSM**: 被拒 — 时间位移模块 (TSM) 对动作识别精度有一致提升且推理无额外 FLOPs.
- **SlowFast**: 被拒作 MVP 基线 — 训练/推理 cost 接近 2 倍, 与 SC-003 (视频时长 × 2) 时延目标拉紧.

**输出给阶段 1 / tasks**:
- `configs/models/pp_tsm_pingpong.yaml` 定义上述默认; 通过 `!include configs/datasets/pingpong_public.yaml`
  机制注入数据集信息, 避免数据配置与模型配置耦合.

---

## R5 — 长视频滑窗默认参数与后处理

**Decision**:
- 默认**窗口 2.0s, 步长 1.0s** (重叠 50%); 以 seconds 计算而非 frames, 由 CLI 按视频实际 fps 转换.
- 置信度阈值默认 **0.5**; 低于阈值的窗口标签置为 `"unknown"` 类.
- 后处理: **相邻同类窗口合并** — 两个连续窗口如果 (a) label 相同 且 (b) 时间间隔 ≤ 1 × stride,
  则合并为一段; 置信度取窗口平均.
- 可视化产物: MP4 + JSON 必须在同一次调用中产出 (FR-015). MP4 通过 OpenCV / ffmpeg 叠加文字;
  JSON 格式见 contracts/.

**Rationale**:
- 2s 窗口 ≈ 乒乓球一个完整来回动作的典型时长, 既能覆盖单次击球又不过度跨多个动作.
- 50% 重叠平衡召回与推理开销; 更大重叠会使 SC-003 (≤ 视频时长 × 2) 难以达成.
- 0.5 作为 softmax 概率阈值是保守的起点, 可根据评估集再调优 (不在 MVP 硬编码任何其他值).

**Alternatives considered**:
- 变点检测 + 分类: 被拒作 MVP (与 Clarification Q4 一致), 可作为 P2 之后扩展.
- Viterbi 解码时序: 被拒作默认 — 需额外的转移矩阵估计, 超出 MVP 范围.

**输出给阶段 1 / tasks**:
- `configs/inference/sliding_window.yaml`: window_sec, stride_sec, conf_threshold, merge_gap_sec
- `src/pingpong_av/inference/sliding_window.py` 实现, 单元测试覆盖合并逻辑.

---

## R6 — 实验目录 schema 与可复现性机制

**Decision**: 每次 `pp train` / `pp eval` 调用都创建 `experiments/<YYYYMMDD-HHMMSS>-<short-sha>-<slug>/`
目录, 写入:

- `manifest.json`: `{ commit, config_hash, seed, started_at, finished_at, status, python_version, cuda_version, gpu_model }`
- `config.yaml`: 本次实际使用的完整 merged 配置的 snapshot (章程 III)
- `log/train.log`, `log/eval.log`
- `checkpoints/*.pdparams` (定期保存)
- `metrics.json`: `{ top1, top5, per_class: {cls: {precision, recall, f1, support}}, macro_avg, test_size }`

Git 工作区脏时强制要求 `--allow-dirty` flag; 否则拒绝启动训练 (可复现性闸门).

**Rationale**: 直接落地章程 II / V 的硬性要求, 让每个指标从源头就带有可追溯的完整上下文.

**Alternatives considered**:
- MLflow / wandb: 被拒作 MVP 依赖 — 额外的外部服务依赖, 与"端到端可运行 + 隔离环境"的简洁目标
  相冲突; 保留为未来可选扩展.
- 仅用日志文件: 被拒 — `manifest.json` + `metrics.json` 结构化形式让评估结果机器可读, 方便后续报告.

**输出给阶段 1 / tasks**:
- `src/pingpong_av/experiment/run_manifest.py` 实现
- `data-model.md` 中 **Experiment** 实体与此 schema 对齐

---

## R7 — 上游官方乒乓球模型接入策略 (US5, 2026-05-12 新增)

**触发原因**:
US2 (PP-TSM 业务训练) 受阻于 R2 的"半公开数据"约束 (AI Studio 注册下载, 需用户手动一次性准备). 但**上游 BCEBOS 提供了真公开可下载的乒乓球训练好的权重 + 单样例 pkl**, 足以做架构层的端到端推理演示, 与 US2 业务训练**互补**.

**Decision**: 在不影响 US2 主线的前提下, 新增独立路径 `pp infer-pkl` 子命令支持上游官方乒乓球样例推理. 模型用 `RecognizerTransformer + SwinTransformer3D + I3DHead`, 与 US2 的 PP-TSM 并行存在, 两者通过 `src/pingpong_av/models/{pp_tsm,videoswin_tennis}.py` 隔离, 不共用业务配置流水线.

**实地验证的 BCEBOS URL** (2026-05-12 200 OK):

| URL | 大小 | 用途 |
|-----|------|------|
| `https://videotag.bj.bcebos.com/Data/example_tennis.pkl` | 7.4 MB | 单样例 pkl, 用于推理演示 |
| `https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_tennis.pdparams` | 380 MB | 训练好的乒乓球权重 |
| `https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_k400.pdparams` | — | K400 预训练 (本项目不直接用) |

**pkl 实际格式** (实测):
```
(
  video_name: str,                # '2019年世锦赛男单决赛马龙VS法尔克20190428-5-of-11'
  labels: dict,                   # {'正反手': 0, '动作类型': 7, '发球': 1}
  frames: list[bytes],            # JPEG-encoded RGB frames, 1280x720, 约 45 帧/clip
)
```

**关键发现**:
1. **架构**: 上游官方乒乓球模型**不是** PP-TSM (Q3 中默认基线), 而是 **VideoSwin Transformer** + I3DHead, `num_classes=8`. 这与 US2 的 PP-TSM 业务路线并存, 彼此不替换.
2. **多任务标签 vs 单 head**: pkl 含 3 个独立任务标签字段 (`正反手` / `动作类型` / `发球`), 但 yaml 中 `MODEL.head.num_classes=8` 且只有一个 I3DHead — 说明上游训练目标是**动作类型**, 其余两个字段在 pkl 中保留只作多任务标注留痕.
3. **类别名上游 README 未公布**: yaml 与 README 都没给 8 类的具体动作名. 本项目用 `动作0..动作7` 占位, 用户从 AI Studio metadata 中拿到真名后应自行替换.

**预处理参数 (与上游 `videoswin_tabletennis.yaml::PIPELINE.test` 对齐)**:
- 帧采样: `num_seg=32` 均匀采样 (注意: 与 PP-TSM 的 num_seg=8 不同)
- Resize: 短边 = 256
- CenterCrop: 224
- Normalize: ImageNet mean/std

**Alternatives considered**:
- **把 VideoSwin 也接入 `pp_tsm.py` 的统一配置流水线**: 被拒 — 上游 yaml schema 差异大 (3D 骨干 + 多任务相关字段), 强行统一会污染 PP-TSM 主线; 用独立 `videoswin_tennis.py` 更清晰.
- **不提供 `pp infer-pkl`, 让用户直接调用上游 `tools/predict.py`**: 被拒 — 上游 `tools/predict.py` 走的是 paddle inference 引擎 (需要先 `export_model.py` 生成 `.pdmodel` + `.pdiparams`), 步骤多、不与本项目的 manifest / schema 对齐.
- **把 example_tennis.pkl 入库**: 被拒 — 7.4MB 二进制, 通过 `source.smoke_sample` 按需下载更合规.

**输出给阶段 1 / tasks**:
- `src/pingpong_av/models/videoswin_tennis.py` 实现模型加载 (调用上游 build_model + 加载 380MB 权重)
- `src/pingpong_av/cli/infer_pkl.py` 实现 CLI 子命令 (pkl 解析 + 帧预处理 + 推理 + JSON 输出)
- `data-model.md` 增加 `pkl-prediction-v1` schema
- `contracts/cli.md` 增加第 7 个子命令 `pp infer-pkl`
- 输出 schema 包含完整 ground_truth 透传 + 明确的 `ground_truth_action_id` 字段以避免多任务歧义 (FR-022)

**实测结果** (2026-05-12):
```
.venv/bin/pp infer-pkl --pkl example_tennis.pkl --checkpoint VideoSwin_tennis.pdparams --topk 5
→ Top-1: 动作7 (id=7)  prob=0.9999  ← 与 GT 一致 (✓ SC-007 通过)
```

---

## 已解决的所有 NEEDS CLARIFICATION

| 来源 | 问题 | 解决方式 |
|------|------|---------|
| plan.md 技术背景 | 主要依赖的具体版本 | R3: Paddle 官方 3.11 wheel, PaddleVideo release/2.2.0 |
| plan.md 技术背景 | 训练性能具体目标 | R4: 默认 50 epoch; 具体时长由 tasks 阶段在目标硬件上实测 |
| spec.md 假设 | 具体选用哪个公开数据集 | R2 (修正版): 仅通过 AI Studio 半公开发布, manual 模式引导 |
| spec.md Q4 | 滑窗默认参数 | R5: 2.0s / 1.0s / threshold=0.5 |
| 章程 VIII | 上游 3.11 适配方式 | R3: 官方 wheel + patches/ |
| spec.md US5 (新增) | AI Studio 数据未就绪时如何演示完整推理 | R7: 通过 BCEBOS 公开权重 + 7.4MB 样例 pkl, `pp infer-pkl` 实测 Top-1 命中 |

**结论**: 阶段 0 所有未知项已解决, 可进入阶段 1 (data-model / contracts / quickstart).
