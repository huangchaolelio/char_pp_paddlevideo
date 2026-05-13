# 功能规范: 基于 PaddleVideo 的乒乓球视频动作识别系统

**功能分支**: `001-pingpong-action-recognition`
**创建时间**: 2026-05-11
**状态**: 草稿
**输入**: 用户描述: "在当前目录下复现https://github.com/PaddlePaddle/PaddleVideo.git这个项目,主要要使用该框架构建乒乓球视频动作识别"

## Clarifications

### Session 2026-05-11

- Q: 乒乓球动作类别集合与数据来源如何确定? → A: 直接复用公开的乒乓球动作数据集 (例如 OpenMMLab/PaddleVideo 生态或相关论文中公布的 TTNet、乒乓球 Stroke 数据集等), 类别集合随所选公开数据集确定; 具体选用哪一份公开数据集由 `/speckit.plan` 阶段根据可获得性与许可证评估决定, 本项目不在 spec 阶段固化类别名称, 但锁定"数据与类别来自公开数据集而非自建"这一来源约束.
  - **修正 (2026-05-12, 实施阶段实地调研结论)**: PaddleVideo `release/2.2.0` 实际上**不提供**任何直接公开下载的乒乓球训练数据集 URL; 上游仅在 BCEBOS 提供训练好的模型权重 + 7.4MB 单样例 pkl, 完整训练集只通过百度 AI Studio 竞赛 #127 分发, **需要用户注册 + 报名才能下载**. 原 Q1 中 "公开" 应弱化为 "半公开 (需 AI Studio 注册)". 本项目 `configs/datasets/pingpong_public.yaml` 因此采用 `source.type: manual` 模式, 哨兵文件 `data/raw/pingpong_public/.ready` 缺失时 `pp data-prepare` 给出 6 步引导后退出码 1. 详见 research.md R2 修正版.
- Q: 上游 PaddleVideo 的集成方式? → A: 以 Git submodule 方式接入并固定在某个 tag/commit, 本仓库只提交 submodule 指针 + 本项目的适配层 (configs/scripts/patches), 不在主仓库内复制上游源码; 升级上游 = 移动 submodule 指针 + 同步更新适配层与文档.
- Q: 乒乓球动作识别的基线模型选哪个? → A: 以 **PP-TSM** 作为本项目的默认基线模型 (2D 骨干 + 时间位移模块), 使用官方 Kinetics 预训练权重初始化, 在乒乓球数据上微调; 其他模型 (PP-TSN / SlowFast / VideoSwin 等) 作为后续可选对比项, 不纳入 MVP 范围.
- Q: 长视频如何分段以输出带时间戳的动作识别结果? → A: 采用**固定窗口 + 步长滑窗** (默认窗口 2s, 步长 1s, 具体值通过配置文件可调) 逐段推理, 对低于置信度阈值的窗口归为 "未知/背景" 类, 再将相邻同类窗口合并为一段; 不在 MVP 范围内引入专门的时序动作定位模型.
- Q: 长视频推理的可视化产物形态? → A: 同时输出两份产物: (1) 在原视频上叠加"动作类别 + 置信度"文本的 **MP4 可视化视频**, (2) 结构化的 **JSON 时间轴文件** (区间列表, 字段包含 start、end、label、confidence); 可选产物 (HTML 报告、关键帧缩略图) 不纳入 MVP 范围.

### Session 2026-05-12

- Q: 在 AI Studio 数据未就绪时, 如何让团队/股东看到"上游官方乒乓球任务在我们的 3.11 环境下真的能跑起来"? → A: **新增 US5 + FR-020/021/022 + SC-007**, 通过加载上游 BCEBOS 公开提供的 `VideoSwin_tennis.pdparams` (380MB, 已实测 200 OK) + `example_tennis.pkl` (7.4MB) 端到端推理一次, Top-1 必须等于 pkl 内 `ground_truth.动作类型`. **关键发现**: 上游官方乒乓球模型是 **VideoSwin Transformer + I3DHead**, **不是 PP-TSM** — 这与本项目 US2 的 PP-TSM 业务主线并行存在, 不替换; 两者通过不同的 `models/` 子模块隔离 (`pp_tsm.py` vs `videoswin_tennis.py`).
- Q: pkl 中含 `正反手 / 动作类型 / 发球` 三个标签字段, 模型只有一个 head, 推理时如何选择? → A: 上游 `videoswin_tabletennis.yaml::MODEL.head.num_classes=8` + `I3DHead` 单输出, 训练目标是**动作类型 (8 类)**; 其余两个字段在 pkl 中保留只作为多任务标注的留痕, 不被本项目的 `pp infer-pkl` 用作判定依据, 但**必须**在输出 JSON 中透传, 让用户看到完整标签上下文 (FR-022).
- Q: 上游 8 个动作类别的具体名称是什么? → A: 上游 README 与 yaml 均**未公布**, AI Studio 竞赛 #127 的 metadata 中可能有; 本项目代码用 `动作0..动作7` 占位, 用户从 AI Studio 拿到真实 metadata 后应自行替换 (与 `pingpong_public.yaml` 的 8 类占位同步).
- Q: AI Studio 数据集的真实形态是什么? → A: 用户在 COS bucket 上传后实地探测确认: 数据集是 **PP-TSN 预提取特征** (43.5GB tar.gz, 含 729 段视频 × ~9000 帧 × 2048 维) + **时序动作标注** (`label_cls14_train.json`, 19054 个 actions, 14 类), 用于上游 **BMN (Boundary-Matching Network) 时序定位**任务, **不是**视频动作分类 (PP-TSM 的领域). 真实 14 类已通过实地解析 JSON 拿到 (摆短/拉/控制/侧身拉/劈长/拧/挑/侧旋/转不转/中性/勾球/普通/逆旋转/下蹲).
- Q: 在 43GB+ 的私有 COS 数据集上能跑通端到端训练吗? → A: **新增 US6 + FR-023~028 + SC-008**, 通过 `source.type: cos` 模式 + `_try_read_bmn_features` 路径 + `scripts/prepare_bmn_inputs.py` + `models/bmn.py` loader, 加上 patches 03 (inspect.getargspec) + 04 (Tensor[0] → .item()) 解决上游 Python 3.11/paddle 2.6+ 兼容性. 实测训练循环成功启动: 16107 train videos / 1967 val videos, GPU 100% 利用, loss 1.77 → 0.81 在前 1050 step 内, batch_cost 1.07s/step.
- Q: BMN 模型的 eval 输出 schema 与 PP-TSM 一致吗 (top1/top5)? → A: **完全不同**. PP-TSM 是片段分类 (logits[N, C]), 评估指标是 top1/top5 + per-class precision/recall + macro-avg. BMN 是时序定位 (输出候选区间 [start, end, score]), 评估指标是 ActivityNet 1.3 风格 AR@AN (Average Recall at Average Number of proposals) + AUC. 因此 `pp eval` 在 cli 层按 `model.name` 分支 (新增 FR-029), 输出文件名相同 (`<run>/metrics.json`), 但 schema 不同 — PP-TSM 用 `metrics-v1`, BMN 用 `bmn-eval-v1`. 实测 epoch 7/20 ckpt: AR@1=28.78%, AR@100=80.37%, AUC=74.63%, n_videos=1967, n_proposals=196700 (SC-009 通过).
- Q: BMN eval 和 BMN 训练能在同一 GPU 上并行跑吗? → A: 危险但可行. T4 15GB, 训练用 7.1GB; eval 前向需要再 ~6GB. 在已有 cached predictions 的情况下, eval 跳过 GPU 前向只跑 cal_metrics (~30s, 纯 CPU + 12 进程 NMS), 与训练几乎不冲突. **新增 FR-030 reuse_existing 模式**: 默认开启, 让用户中途用 cached 预测复算 metrics, 不抢 GPU.
- Q: 上游 anet_prop.py 在 verbose=True 时写到一个硬编码的 'data/bmn/BMN_Test_results/auc_result.txt', 这个目录怎么处理? → A: **不打 patch** (路径硬编码深, 改一行需要侵入). 改为在 `run_upstream_bmn_eval` 调用前用 Python 预创建该目录 (FR-031). 该目录纯运行时输出 (AUC 文本), 已加入 `.gitignore`.

## 用户场景与测试 *(必填)*

### 用户故事 1 - 在本地复现 PaddleVideo 框架并验证可用 (优先级: P1)

作为一名计算机视觉/体育分析方向的研究人员或工程师, 我需要在本地环境中成功复现 PaddleVideo 项目, 并通过其示例数据/示例模型完成一次端到端的训练或推理验证, 以确认运行环境(依赖、数据集格式、训练流程)就绪, 为后续的乒乓球动作识别任务打好基础。

**优先级原因**: 这是整个项目的"地基"。如果 PaddleVideo 不能在本地正确运行, 后续的乒乓球动作识别建模、训练、评估都无法进行。该故事独立交付的价值是: 一个可重复使用的、可验证的视频理解开发环境。

**独立测试**: 完全可以独立测试与交付——通过下载示例数据集、使用框架自带的某个动作识别模型完成一次小规模的训练和一次推理, 确认 loss 收敛、推理输出格式正确, 即视为故事完成。无须依赖乒乓球数据。

**验收场景**:

1. **给定** 一台具备 GPU 的开发机器和干净的 Python 环境, **当** 按照项目 README 完成依赖安装和示例数据准备后, **那么** 用户可以在不修改源码的情况下成功启动一次示例模型的训练任务, 且训练日志中 loss 正常下降。
2. **给定** 一个已训练好或官方提供的预训练权重, **当** 用户运行框架自带的推理脚本对一段示例视频进行动作识别时, **那么** 系统返回该视频对应的 Top-K 动作类别及置信度。
3. **给定** 复现过程中遇到环境/版本冲突, **当** 用户参照本仓库提供的环境说明执行修复步骤时, **那么** 问题应被预先记录并能在合理步骤内解决。

---

### 用户故事 2 - 构建乒乓球视频动作识别模型并完成训练 (优先级: P1)

作为一名希望分析乒乓球比赛/训练视频的用户(教练、运动分析师或开发者), 我需要使用 PaddleVideo 框架, 在乒乓球动作数据上训练一个动作识别模型, 使其能够识别一段乒乓球视频中运动员所做的动作类别(例如发球、正手攻、反手推挡、削球、扣杀等)。

**优先级原因**: 这是用户最终的业务目标——把通用的视频理解框架"落地"到乒乓球动作识别这一具体场景。它直接交付用户最关心的价值: 一个真正能识别乒乓球动作的模型。

**独立测试**: 在用户故事 1 完成之后即可独立测试——准备一份带动作标签的乒乓球视频片段数据集, 配置训练流程, 完成训练后在留出的测试集上评估指标(如 Top-1/Top-5 准确率), 通过指标即可验证。

**验收场景**:

1. **给定** 一份按统一规范组织的乒乓球动作标注数据(视频片段 + 动作类别标签)和训练配置, **当** 用户启动训练流程时, **那么** 系统能正确读取数据、按配置完成多轮训练, 并输出每轮的训练/验证指标日志。
2. **给定** 一个在乒乓球数据上训练完成的模型权重, **当** 用户在留出的乒乓球测试集上运行评估脚本时, **那么** 系统输出 Top-1 准确率、Top-5 准确率以及各类别的混淆矩阵或分类报告。
3. **给定** 一段未参与训练的乒乓球短视频片段, **当** 用户调用推理接口时, **那么** 系统返回该片段对应的动作类别及置信度排序结果。

---

### 用户故事 3 - 对完整乒乓球视频进行动作识别与可视化 (优先级: P2)

作为一名教练或视频剪辑人员, 我希望上传一整段乒乓球比赛/训练视频(包含多个回合), 让系统自动切分并识别其中每一段的动作类别, 并以时间轴/标注视频的形式展示, 便于回放分析和精彩片段提取。

**优先级原因**: 在 P1 模型可用的基础上, 这一步把"片段级识别"扩展为"长视频识别 + 可视化", 显著提升实际可用性。但它依赖 P1 训练出的模型, 因此优先级低于 P1。

**独立测试**: 在用户故事 2 完成后可独立测试——给定一段长视频, 系统输出一份带时间戳的动作识别结果(JSON/CSV)以及一段在画面上叠加了动作类别文字的可视化视频。

**验收场景**:

1. **给定** 一段时长数分钟的乒乓球比赛视频, **当** 用户运行端到端推理脚本时, **那么** 系统生成包含 [起始时间, 结束时间, 动作类别, 置信度] 的结构化结果文件。
2. **给定** 同一段视频和上述结构化结果, **当** 用户运行可视化脚本时, **那么** 系统在同一次调用中输出: (a) 在原视频画面上叠加了动作类别标签和置信度文本的 MP4 视频文件, (b) 与该视频对齐的结构化 JSON 时间轴文件 (区间列表, 字段含 start/end/label/confidence).

---

### 用户故事 4 - 数据集管理与样本扩充 (优先级: P3)

作为一名数据维护者, 我希望能够方便地添加新的乒乓球动作类别或新的样本视频, 并通过统一的脚本完成数据规范化、划分(训练/验证/测试)和标注文件生成, 以便持续迭代模型。

**优先级原因**: 持续迭代的能力。P1/P2 已经能交付价值, 但要让模型能随时间不断变好, 需要这一步。

**独立测试**: 给定原始视频与标注(CSV/JSON), 运行数据准备脚本后, 在指定目录下生成框架可直接读取的训练/验证/测试列表文件以及切分好的视频片段。

**验收场景**:

1. **给定** 一批新的原始乒乓球视频与对应的动作标注, **当** 用户运行数据准备脚本时, **那么** 系统在标准目录下生成统一格式的训练/验证/测试样本及标注列表。
2. **给定** 上一步生成的数据, **当** 用户启动训练时, **那么** 训练流程无需额外修改即可读取并使用新增数据。

---

### 用户故事 5 - 复用 PaddleVideo 官方乒乓球样例验证完整推理路径 (优先级: P1)

*(2026-05-12 新增) 作为复现 P1 的补充: 在尚未获得 AI Studio 竞赛 #127 训练数据时, 用户希望能用上游 BCEBOS 提供的真公开资源 (训练好的乒乓球权重 + 7.4MB 单样例 pkl) 立即验证整个推理路径 (架构 + 权重加载 + 预处理 + 输出 schema) 是否正确, 而无需先准备数据集.*

**优先级原因**: 直接证明"PaddleVideo 复现 + 乒乓球任务"端到端真实可用; 与 US1 (上游环境健康度) 互补 — US1 验证"能 build_model + 跑随机张量", US5 验证"能加载真实训练权重 + 跑真实数据"; 是 US2 业务指标尚不可达时**唯一**能给团队/股东看的可信演示.

**独立测试**: 不依赖 AI Studio 数据; 只需要互联网 + 既有 `.venv`. 单条命令完成: 下载样例 → 加载模型 → 推理 → 对比 GT.

**验收场景**:

1. **给定** 已 bootstrap 的环境 + 7.4MB `example_tennis.pkl` (`pp data-prepare` 在 manual 模式下自动下载) + 380MB `VideoSwin_tennis.pdparams` (用户用 README 给出的 curl 命令下载), **当** 用户运行 `pp infer-pkl --pkl <pkl> --checkpoint <pdparams>`, **那么** 系统输出 Top-K 预测 + 与 pkl 内 GT 标签的对比, 且 Top-1 应等于 GT (`动作类型: 7`).
2. **给定** 样例 pkl 的标签结构含三个独立字段 (`正反手` / `动作类型` / `发球`, 揭示上游多任务标注但单 head 训练), **当** 系统输出 JSON 时, **那么** 必须把全部 3 个 GT 字段透出, 并明确说明模型推理目标是哪一个 (`ground_truth_action_id` 单独命名以避免歧义).
3. **给定** checkpoint 文件缺失, **当** 用户运行 `pp infer-pkl`, **那么** 系统退出码 1 且 stderr 给出**确切的 curl 下载命令** (含完整 URL).

---

### 用户故事 6 - 通过私有 COS 拉取大型半公开数据集并训练 BMN 时序定位 (优先级: P1)

*(2026-05-12 新增) 作为乒乓球动作识别项目维护者, 我希望本项目能从腾讯云 COS 拉取大型 (43.5GB+) 数据集 (例如 AI Studio 竞赛 #127 已上传到团队 COS bucket 的 `Features_competition_train.tar.gz` PP-TSN 特征 + `label_cls14_train.json` 时序标注), 并在此真实数据上训练 BMN (Boundary-Matching Network) 时序定位模型, 输出与 spec.md FR-014/015 一致的时间区间产物.*

**优先级原因**: US2 (PP-TSM 动作分类) 受阻于 AI Studio 数据需注册 + 6.9GB UCF101 公网下载速率瓶颈; **私有 COS 是真正解决"端到端业务训练"的唯一可行路径**. 一旦 COS 集成 + BMN 路径打通, 本项目就有了**完整的"上游真实模型 + 真实数据 + 端到端训练"链路**, 与 US2 (PP-TSM 主线) 互补 — US2 做片段级分类, US6 做长视频时序定位.

**独立测试**: 给定 `.env` 中的 COS 凭据 + bucket prefix, 跑 `pp data-prepare` 应自动拉取 + 解压 + 写出 splits, 然后跑 `python scripts/prepare_bmn_inputs.py` 转换为 BMN 输入, 最后 `pp train --config configs/models/bmn_pingpong.yaml` 应启动训练循环并打印每步 loss.

**验收场景**:

1. **给定** `.env` 含 `COS_REGION / COS_BUCKET / COS_SECRET_ID / COS_SECRET_KEY / COS_VIDEO_PREFIX`, COS 上的 `<prefix>/Features_competition_train.tar.gz` (43GB+) + `<prefix>/label_cls14_train.json` 已就位, **当** 用户运行 `pp data-prepare --config configs/datasets/pingpong_competition_bmn.yaml`, **那么** 系统自动:
   - 下载 .tar.gz + .json 到 `data/raw/pingpong_competition/<dataset_name>/`
   - 流式解压 .tar.gz 到 `data/clips/...` (避免双遍读, 不阻塞)
   - 通过新增的 BMN 特征发现路径 (`_try_read_bmn_features`) 把每个 .pkl 视为一段独立视频
   - 写 splits + meta jsonl, 退出码 0
2. **给定** 已 prep 好的特征 + label JSON, **当** 用户运行 `python scripts/prepare_bmn_inputs.py`, **那么** 输出 BMN 上游期望的 `feature/*.npy` (8s 滑窗切片) + `label_fixed.json` + `label_gts.json`, 并自动过滤无对应 .npy 的 label 条目.
3. **给定** BMN 输入就绪 + `configs/models/bmn_pingpong.yaml` (含 14 类), **当** 用户运行 `pp train --config configs/models/bmn_pingpong.yaml`, **那么** 训练循环成功启动 (上游 `RecognizerTransformer` → BMN 网络 → loss 输出 → `manifest.json` 含章程 II 四元组). 至少前 1000 step loss 显式下降.
4. **给定** patch 系统已就绪, **当** 用户在新机器上 `bash scripts/apply_upstream_patches.sh`, **那么** 4 个 patches (paddle.fluid 移除、decord lazy import、inspect.getargspec → getfullargspec、record Tensor 标量索引) 全部按字母序应用且幂等.

---

### 边界情况

- 当目标机器没有 GPU 或 CUDA 版本与 PaddlePaddle 不兼容时, 系统应给出明确的环境检查提示, 并提供 CPU 模式下的最小可运行回退路径(用于功能验证, 不保证训练速度)。
- 当输入视频损坏、过短(短于一个采样窗口)或编码格式不被支持时, 推理流程应跳过该样本并在日志中记录, 而不是整体崩溃。
- 当某个乒乓球动作类别样本数极少(类别不平衡), 评估指标除整体准确率外, 还应输出每类指标, 便于发现长尾问题。
- 当长视频中存在非乒乓球动作的画面(例如观众、空镜、暂停)时, 系统对应片段应能输出"未知/背景"类别或低置信度提示, 而不是强行归入某个动作类别。
- 当训练中断(断电、OOM)后重新启动时, 应能从最近一次保存的 checkpoint 继续训练。
- 当数据集划分发生变化时, 不能出现训练集与测试集视频重叠造成评估指标失真。

## 需求 *(必填)*

### 功能需求

#### 框架复现相关

- **FR-001**: 系统必须以 Git submodule 方式接入 PaddleVideo 上游仓库, 固定到一个可审计的 tag 或 commit, 并在本仓库中提供从零开始复现该版本 PaddleVideo 所需的依赖说明、环境准备步骤和验证方法, 用户按文档操作即可完成环境搭建。
- **FR-002**: 系统必须提供至少一个示例动作识别模型的训练与推理可运行流程, 用于验证 PaddleVideo 框架在本地能够正确运行。
- **FR-003**: 系统必须记录复现过程中常见的依赖/版本/数据问题及对应解决方案, 以便重复使用与团队协作。

#### 数据相关

- **FR-004**: 系统必须支持以统一目录结构组织乒乓球动作识别数据, 包含原始视频、切分后的动作片段、标注文件以及训练/验证/测试划分列表。
- **FR-005**: 系统必须提供数据准备脚本, 能够将所选乒乓球动作数据集 (无论来源是直接公开 URL、本地用户目录, 还是需要注册后手动下载的半公开数据集) 整理并转换为框架可直接训练的样本与标注列表. 对于半公开数据集 (如 AI Studio 竞赛 #127), 在用户尚未完成手动准备时, 脚本必须给出明确的步骤化引导 (注册地址 / 下载位置 / 哨兵文件) 与可恢复的退出码, **不得**编造或硬编码不存在的下载 URL.
- **FR-006**: 系统必须支持至少一种业界常见的视频动作识别数据组织方式(片段级动作识别), 并允许后续扩展到时序动作定位场景。
- **FR-007**: 数据准备流程必须保证训练集、验证集、测试集之间没有视频或片段层面的泄漏。

#### 模型训练与评估

- **FR-008**: 系统必须允许用户通过配置文件(无需修改源码)指定: 数据路径、动作类别列表、模型结构、采样策略、批大小、学习率、训练轮数、保存路径等关键训练参数。
- **FR-009**: 系统必须在训练过程中输出可读的训练/验证日志, 包含每轮的 loss 与主要指标(Top-1/Top-5 准确率), 并定期保存模型 checkpoint。
- **FR-010**: 系统必须支持从 checkpoint 恢复训练, 用于应对训练中断或继续微调。
- **FR-011**: 系统必须提供独立的评估流程, 在测试集上输出 Top-1 准确率、Top-5 准确率以及每类指标(如混淆矩阵或分类报告)。
- **FR-012**: 系统必须支持加载预训练权重对模型进行初始化, 以提升小样本乒乓球数据上的训练效果; 默认基线为 **PP-TSM**, 必须能加载其官方 Kinetics 预训练权重用于微调, 其他模型结构作为可选扩展而非 MVP 必需。

#### 推理与应用

- **FR-013**: 系统必须提供对单个乒乓球视频片段的推理接口, 输入为视频文件路径, 输出为 Top-K 动作类别及置信度。
- **FR-014**: 系统必须提供对长乒乓球视频的端到端推理流程, 采用**固定窗口 + 步长滑窗**策略 (窗口长度与步长可通过配置调整, 默认 2s / 1s) 对全视频逐段分类, 将低于置信度阈值的窗口标记为 "未知/背景", 并将相邻同类窗口合并, 最终输出包含 [起始时间, 结束时间, 动作类别, 置信度] 的结构化结果。
- **FR-015**: 系统必须对 FR-014 的推理结果生成两份可视化产物: (a) 在原视频上叠加"动作类别 + 置信度"文本的 **MP4 可视化视频**; (b) 结构化的 **JSON 时间轴文件** (数组, 每个元素至少包含 `start`、`end`、`label`、`confidence` 字段). 两份产物必须在同一次调用中一并产出, 便于下游剪辑/统计工具直接消费.
- **FR-016**: 系统在推理流程中遇到无法处理的输入(损坏文件、不支持格式、过短片段)时, 必须跳过并记录日志, 不得整体崩溃。

#### 上游官方样例支持 (US5)

- **FR-020**: 系统必须提供一条独立的推理命令, 接受 PaddleVideo 上游样例 pkl (元组形式: `(video_name, label_dict, list[jpeg_bytes])`) 与对应的乒乓球预训练权重, 在不需要任何业务数据准备的前提下完成: 反序列化 → JPEG 解码 → 均匀帧采样 (默认 `num_seg=32`, 与上游 `videoswin_tabletennis.yaml` 对齐) → ImageNet 标准化 → 喂给上游 `RecognizerTransformer + SwinTransformer3D + I3DHead`, 输出 Top-K 预测.
- **FR-021**: 当上述命令的 checkpoint 文件缺失时, 退出码必须为 1, 且 stderr 必须包含可直接复制粘贴的 curl 下载命令 (含确切 BCEBOS URL), 不得引导用户去查文档自行查找.
- **FR-022**: 输出 JSON 必须遵循 `pkl-prediction-v1` schema (见 data-model.md), 至少含: `input` (pkl 路径 / video_name / 帧数), `model` (checkpoint / framework / backbone / head / num_classes), `ground_truth` (pkl 中**全部**标签字段, 透传不解释), `ground_truth_action_id` (单独标注模型推理目标对应的 GT id, 避免多任务标签歧义), `prediction.topk` (排序后的 Top-K 列表), `prediction.top1_match_gt` (布尔值或 null).

#### 私有 COS 数据源支持 (US6, 2026-05-12 新增)

- **FR-023**: 系统必须支持以 ``source.type: cos`` 模式接入腾讯云 COS, 凭据通过 `.env` (`COS_REGION` / `COS_BUCKET` / `COS_SECRET_ID` / `COS_SECRET_KEY` / `COS_VIDEO_PREFIX`) 提供, 不在 yaml 内硬编码 secret. 用户可通过 `source.bucket` / `source.region` / `source.prefix` / `source.keys` / `source.extract` / `source.max_thread` 在 yaml 中显式覆盖与扩展.
- **FR-024**: 系统必须能够流式 (`r|gz`) 解压 43GB+ 的 .tar.gz 大文件, **不**做"先 getmembers() 再 extractall()"的双遍扫描 (后者对此规模会耗时 1+ 小时), 必须以单遍逐成员模式工作并打印进度.
- **FR-025**: 系统必须为 BMN 时序定位任务 (上游 `BMNLocalizer + BMN backbone + BMNLoss`) 提供独立的数据发现路径 (`_try_read_bmn_features`): 把 ``Features_competition_train/*.pkl`` 视为视频级单元, 从同级 `label_cls*.json` 中按 ``url`` 匹配, 用动作标签的众数填 ``VideoClip.label_id`` (用作 splitter 分层校验; 实际 BMN 训练时 ``label_id`` 字段被忽略, 真实标签在 JSON 中按 url 索引读取).
- **FR-026**: 系统必须提供 ``scripts/prepare_bmn_inputs.py`` 把上游原始 GT JSON 转换为 BMN 训练所需的 ``label_fixed.json`` + ``label_gts.json`` + ``feature/*.npy`` 滑窗切片, 并自动过滤 (`miss > 0` 时) 没有对应 .npy 的 label 条目以避免 dataloader 文件不存在错误.
- **FR-027**: 系统必须把 BMN 模型纳入 ``models/registry.py``, 并通过 ``configs/models/bmn_pingpong.yaml`` 的业务配置 + ``models/bmn.py`` loader 在运行时合并到上游 ``bmn_tabletennis.yaml`` 模板; 业务 yaml 仅允许覆盖路径与 epochs/batch_size, 网络结构由上游提供.
- **FR-028**: 当上游 release/2.2.0 在 Python 3.11 / paddle 2.6+ 下 BMN 训练触发的兼容性问题 (`inspect.getargspec` 已删除; `Tensor.numpy()[0]` 对 0-d 张量报错) 时, 系统必须通过 ``third_party/patches/`` 提供最小侵入修复 (patches 03 + 04), 与 patches 01 + 02 一同由 ``apply_upstream_patches.sh`` 按字母序幂等应用.
- **FR-029**: `pp eval` 必须根据训练时 snapshot 的 ``model.name`` 字段分支: ``pp_tsm`` 走原有 logits→top1/top5/per-class/macro-avg 路径; ``bmn`` 走时序定位路径 → 调用上游 ``test_model`` (BMNMetric 输出 ActivityNet 1.3 风格 ``bmn_results_<subset>.json``) → 再调 ``cal_metrics`` 拿到 AR@1/5/10/100, 写入 ``bmn-eval-v1`` schema. 两条路径**共享同一 CLI 子命令**, 输出文件路径相同 (``<run>/metrics.json``), schema 不同; 调用方按 ``schema`` 字段判别.
- **FR-030**: BMN eval 必须支持 ``reuse_existing=True`` 模式 (默认): 当 ``<run>/bmn_eval/results/bmn_results_<subset>.json`` 已存在时, 跳过 GPU 前向, 仅重算 metrics. 用于: (a) 训练并行时避免 GPU 争抢; (b) 同一 ckpt 上调 tIoU 阈值或其他后处理参数; (c) 失败重试.
- **FR-031**: BMN eval 必须自动创建上游 ``anet_prop.py`` 在 ``verbose=True`` 时硬编码写入的 ``data/bmn/BMN_Test_results/`` 目录 (用作 AUC 文件 ``auc_result.txt`` 的落点), 避免 ``FileNotFoundError`` 中断评估. 此目录全部在 ``.gitignore`` 中.
- **FR-032**: ``_find_run_dir`` 必须支持两种 ckpt 布局: PP-TSM 路径下 ``<run>/checkpoints/<ckpt>``; BMN 路径下 ``<run>/BMN_epoch_<NNNNN>.pdparams`` (上游直接写到 run 根目录). 判定标准是父目录链中第一个含 ``manifest.json`` 的目录.

#### 可复现性与工程化

- **FR-017**: 系统必须将所有训练/评估/推理使用的配置以文件形式纳入版本管理, 保证实验可复现。
- **FR-018**: 系统必须支持设置随机种子, 在硬件条件相同时, 同一份数据与配置的训练结果在合理范围内可复现。
- **FR-019**: 系统必须提供清晰的运行入口(脚本或命令行), 用户无须深入框架内部即可完成"环境验证 → 数据准备 → 训练 → 评估 → 推理 → 可视化"全流程。

### 关键实体 *(如果功能涉及数据则包含)*

- **乒乓球动作类别 (ActionClass)**: 表示一种乒乓球技术动作(例如: 发球、正手攻、反手推挡、削球、扣杀、搓球等), 至少包含类别 ID、类别名称、可选描述。
- **视频样本 (VideoClip)**: 表示一段被标注的乒乓球动作片段, 至少包含片段唯一标识、源视频、起止时间、对应的动作类别 ID、所属数据划分(训练/验证/测试)。
- **数据集划分 (DatasetSplit)**: 表示训练/验证/测试三类样本集合, 每个划分对应一份样本列表文件, 供训练/评估流程读取。
- **训练实验 (Experiment)**: 表示一次训练任务, 至少包含使用的配置文件、起止时间、最终指标、保存的 checkpoint 路径与日志路径。
- **模型 (Model)**: 表示一个训练得到的或下载的动作识别模型权重, 至少包含模型结构信息、训练配置引用、在测试集上的指标。
- **推理结果 (PredictionResult)**: 表示对单个片段或一段长视频运行模型后得到的结果, 至少包含输入标识、Top-K 动作类别与置信度, 长视频场景下额外包含每段的起止时间。

## 成功标准 *(必填)*

### 可衡量的结果

- **SC-001**: 一名熟悉深度学习的开发者可以在 1 个工作日内, 按照本仓库文档完成 PaddleVideo 的环境搭建并跑通至少一个示例模型的"训练 + 推理"验证流程。
- **SC-002**: 在所选公开乒乓球动作数据集上 (类别数以该数据集官方定义为准, 要求类别数 ≥ 5), 训练得到的模型在其官方/规范化留出测试集上的 Top-1 准确率不低于 70%, Top-5 准确率不低于 90%。
- **SC-003**: 对一段长度不超过 10 分钟的乒乓球视频, 端到端推理流程在单 GPU 环境下可在视频时长的 2 倍时间内完成识别并产出结构化结果与可视化视频。
- **SC-004**: 在同一份数据、同一份配置和同样的随机种子下, 两次完整训练得到的测试集 Top-1 准确率差异不超过 ±2 个百分点。
- **SC-005**: 数据准备脚本可在不修改源码的前提下接入新增的乒乓球类别或样本, 且新增数据后训练流程零改动可直接复用, 数据导入耗时随样本数线性增长。
- **SC-006**: 至少 90% 的输入异常情况(损坏视频、不支持格式、过短片段)在推理过程中被捕获并跳过, 不会引发整体进程崩溃。
- **SC-007**: 在 PaddleVideo 官方提供的 `example_tennis.pkl` 上, 使用官方 `VideoSwin_tennis.pdparams` 权重通过 `pp infer-pkl` 推理, Top-1 预测必须等于 pkl 内的 `ground_truth.动作类型` 真值, 且 Top-1 置信度 ≥ 0.90. (此项是 US5 的硬验收门槛, 与 SC-002 不同 — SC-002 需要 AI Studio 真实训练数据, SC-007 不需要.)
- **SC-008**: US6 BMN 端到端验收: 在 COS 上的 `Features_competition_train.tar.gz` (43.5GB) + `label_cls14_train.json` 数据集上, `pp data-prepare → scripts/prepare_bmn_inputs.py → pp train --config configs/models/bmn_pingpong.yaml --allow-dirty` 必须能成功启动训练循环, 前 1000 step 内 loss 显式下降 (起点 1.7+, 第 1000 step ≤ 1.5), GPU 利用率 ≥ 95%, 不出现 dataloader / IndexError / 上游 API 兼容性错误. (Loss 收敛至业务可用值 < 0.5 的硬门槛留作 Phase 9 / 完整 20 epoch 训练后再验; 本 SC 只验**架构通**.)
- **SC-009**: BMN eval 端到端验收: 给定 BMN checkpoint + ``upstream_config.yaml`` snapshot, ``pp eval --checkpoint <ckpt> --split val`` 必须输出符合 ``bmn-eval-v1`` schema 的 ``metrics.json``, 含 AR@1/5/10/100 四个数值 + ``n_videos_evaluated`` + ``n_proposals`` + ``class_names``. AR@100 必须 ≥ 60% (即模型确实学到了边界, 不是随机猜). 实测: 在 epoch 7/20 ckpt 上 AR@1=28.78%, AR@5=59.17%, AR@10=68.27%, AR@100=80.37%, AUC=74.63% — 通过.

## 假设

- 用户具备一台带有 NVIDIA GPU 的开发/训练机器, 并安装了与 PaddlePaddle 兼容的 CUDA/cuDNN; 否则只能进行 CPU 模式下的功能验证, 不保证训练时间。
- 数据与动作类别集合来自**半公开**乒乓球动作数据集 (PaddleVideo 官方乒乓球任务通过百度 AI Studio 竞赛 #127 分发, 需用户注册 + 报名后下载); 本仓库不打包原始视频数据, 也**不**对该 URL 编造直链, 而是通过 `source.type: manual` 模式 + 哨兵文件机制引导用户首次手动准备数据 (一次性). 后续 `pp data-prepare` / `pp train` 命令链不变.
- 类别集合随所选半公开数据集确定 (上游 README 未公布具体类别名, 由数据集 metadata 在解压后给出), 在一次实验内固定; 切换到不同数据集或更细粒度子类视为新的实验.
- "动作识别"在本规范中默认指片段级分类(给定一个已切分的视频片段, 输出动作类别), 长视频上的"识别+时间戳"通过**固定窗口 + 步长滑窗 + 同类合并**实现 (US3, FR-014), 不要求达到专门时序定位模型的精度水平。
- 复现 PaddleVideo 指的是: 在本仓库中以 Git submodule 方式接入上游, 提供能在 Python 3.11 隔离环境下可运行、可训练、可推理的集成 (含适配层、配置、脚本), 用于支撑乒乓球动作识别业务; 不要求重新实现框架内部所有模型或论文复现实验。
- 模型与权重的分发遵守 PaddleVideo 上游项目的开源许可证, 不在本仓库直接分发受限的第三方数据。
- 评估指标以 Top-1/Top-5 准确率与每类指标为主; 实时性指标按"视频时长的 2 倍以内"作为可接受默认值, 后续可在性能优化阶段再收紧。
