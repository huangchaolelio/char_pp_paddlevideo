# 功能规范: 原始视频到 BMN 时序定位的端到端推理与训练适配

**功能分支**: `002-raw-video-feature-bmn`
**创建时间**: 2026-05-13
**状态**: 草稿
**输入**: 用户描述: "参考applications/TableTennis/extractor/extract_feat.py 脚本及paddlevideo仓库,增加原始视频模型适配模型训练及推理"

## 澄清

### Session 2026-05-13 (实地探测后填补的事实)

- Q: 用户提到的 `applications/TableTennis/extractor/extract_feat.py` 是否存在? → A: **不存在**. TableTennis 路径下只有 `extract_bmn_for_tabletennis.py` (消费已有 pkl 切窗用), 没有产生 pkl 的脚本. 真正产生 `{'image_feature': ndarray(N, 2048)}` pkl 的脚本是 `applications/FootballAction/extractor/extract_feat.py`. 本规范以**实际可参考的 FootballAction 版**为依据.
- Q: 上游用什么模型抽 2048-d 特征? → A: **PP-TSM ResNet50** + Kinetics-400 dense 预训练 (`ppTSM_k400_dense.pdparams`, BCEBOS 公开下载). 选择上游 inference 模型的 **第二个输出** `output_names[1]` (即 ResNet50 全局池化后的 2048-d 特征向量, 跳过最后的分类层). 参见 research.md R9.
- Q: TableTennis 数据集的 fps 是多少? → A: 25 fps (与上游 `applications/TableTennis/extractor/configs/configs.yaml` 及本仓库已有的 `label_cls14_train.json::fps=25` 一致).

### Session 2026-05-13 (clarify 阶段)

- Q: clip_id 计算时 sha256 应 hash 什么内容? → A: **sha256 视频文件内容** (流式整文件 hash). 抗改名 + 跨机器一致 + 与 SC-013 强对齐. clip_id = `sha256(file_bytes)[:32]`, 写入 .pkl 名为 `<32-hex>.pkl`, 写入 GT JSON 时 url 字段为 `<32-hex>.mp4`.
- Q: PP-TSM inference 模型 (`.pdmodel + .pdiparams`) 怎么得到? → A: **写 `scripts/export_pptsm_inference.py`**, 首次运行 `pp extract-feat` 时自动从 `ppTSM_k400_dense.pdparams` (动态图) 通过 `paddle.jit.to_static + paddle.jit.save` 转换出 inference 双文件, 缓存到 `data/raw/pretrained/ppTSM.{pdmodel,pdiparams}`. 用户只需下载一份 .pdparams (FR-038 自动下载入口), 不需要额外文件. 与章程 VII (端到端 ≤ 5 条命令) 对齐.

## 用户场景与测试 *(必填)*

### 用户故事 1 - 给定一段任意 mp4 视频, 用现有 BMN 模型直接出动作时间区间 (优先级: P1)

作为乒乓球教学/比赛视频分析师, 我手上有一段没有任何标注的乒乓球比赛 mp4 视频 (例如直播录像或手机录制), 我希望直接喂给本项目, 输出一份 JSON 时间轴 (含若干 [start_sec, end_sec, label, score] 候选区间), 不需要先去 AI Studio 报名拿什么"预提取特征数据集".

**优先级原因**: 这是**真正闭环业务价值**的最后一块拼图. v0.2.x 已经能从 COS 上的预提取特征训练 + 评估 BMN; 但所有真实用户拿到本项目后第一个问题就是"我自己的视频怎么用?". 没有这一步, 整个 BMN 主线对外只是个 demo, 不是产品.

**独立测试**: 用 PP15pingskills 教学视频 (例如 COS bucket 上 `PP15pingskills/2.正手攻球-PingSkills乒乓球教学片.flv` 转 mp4 后 ~5 分钟) 走端到端推理 → 检查输出 JSON 含 ≥ 5 条候选区间 + 每条含合法字段 (`start_sec` / `end_sec` / `label` / `score`).

**验收场景**:

1. **给定** 一段 ≤ 10 分钟的 mp4 乒乓球视频 + 一份本项目已训练的 BMN ckpt (例如 `experiments/<run>/BMN_epoch_00020.pdparams`) + 上游 ppTSM_k400_dense 权重 (BCEBOS 公开), **当** 用户运行 `pp infer-rawvideo --input <video>.mp4 --bmn-checkpoint <ckpt> --output-dir <out>/`, **那么** 系统:
   - 用 ffmpeg 抽帧到临时目录 (默认 25 fps)
   - 把帧序列分批喂给 PP-TSM 抽 2048-d 特征
   - 把得到的 (N, 2048) ndarray 序列化成上游兼容的 pkl
   - 用 BMN 在该 pkl 上做时序定位 (8s 滑窗 + NMS 后处理)
   - 输出 (a) ActivityNet 1.3 风格 JSON 候选区间, (b) 与现有 `pp infer-video` 一致的 `pkl-prediction-v2-bmn` schema 结果, (c) 可视化 mp4 (在原视频上叠加候选区间), 全部落到 `<out>/` 目录;
   - 退出码 0;
   - 临时帧目录在退出前清理 (默认 ON, 可 `--keep-frames` 关).

2. **给定** PP-TSM 权重缺失, **当** 用户运行该命令, **那么** stderr 必须含可直接复制粘贴的 `curl` 下载行 (含确切 BCEBOS URL `https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams`) + 期望落点路径, 退出码 1 (与 US5 `pp infer-pkl` 缺权重的处理对齐, FR-021 复用).

3. **给定** 输入视频损坏或 ffmpeg 抽帧失败, **当** 命令运行, **那么** 系统不得整体崩溃, 必须记录该视频路径与 ffmpeg 错误码到日志, stderr 给出明确诊断, 退出码 4.

---

### 用户故事 2 - 把"原始视频集合"转成本项目兼容的 pkl 数据集 (用于训练扩充) (优先级: P2)

作为乒乓球项目维护者, 我手上有一批新拍/新收集的原始 mp4 视频 (例如教练自己录的训练数据), 我希望把它们批量转成 `<hash>.pkl + label.json`, 像 `Features_competition_train.tar.gz` 一样直接喂进现有 BMN 训练管线 (`scripts/prepare_bmn_inputs.py` → `pp train`), 不需要重写训练代码.

**优先级原因**: 这是 US1 的批量化版本. AI Studio 竞赛 #127 数据集只有 729 视频; 用户要扩展业务时必须有这条路径. 与 US4 (数据集管理与样本扩充) 衔接, 也是后续做"持续学习/迁移训练"的前置.

**独立测试**: 给 3-5 个 PP15pingskills 教学视频 (短, 各 5-10 min) + 一份用户手写的 label.json (按 `label_cls14_train.json` schema), 跑 `pp build-feature-pkls`, 看产出是否能直接被 `prepare_bmn_inputs.py` 消费 + `pp train` 启动.

**验收场景**:

1. **给定** 一个含 N 个 mp4 的目录 + (可选) 一份按 `label_cls14_train.json` schema 写的 GT JSON, **当** 用户运行 `pp build-feature-pkls --videos-dir <dir> --output-dir <out>/Features_<name>/ [--gt-json <path>]`, **那么** 系统:
   - 对每个 mp4 抽帧 + PP-TSM 前向 + 得到 (N_i, 2048) 特征向量
   - 用稳定的 hash (默认 mp4 文件名 stem 的 sha256 前 32 位) 作为 .pkl 名
   - 写到 `<out>/Features_<name>/<hash>.pkl` (与上游 tar.gz 解压后内部布局一致)
   - 同时写一份 `<out>/manifest.csv` (列: `video_path, clip_id, n_frames, fps, sha256, written_pkl_path, ms_duration`) 供下游审计;
   - 如提供了 `--gt-json`, 同步把 url 字段替换为新 clip_id, 写到 `<out>/label_cls14_<name>.json`, 这样就能直接拼接到现有 `pp data-prepare` 的 cos 拉下来的 label JSON 后续训练.

2. **给定** N = 100 段视频 (合计 ~100 分钟), **当** 命令运行, **那么** GPU 利用率 ≥ 80%, 总耗时 ≤ 视频总长度的 0.5 倍 (即 100 分钟视频 ≤ 50 分钟抽特征, T4 单卡, batch_size=32).

3. **给定** 命令中途因 OOM / Ctrl-C 中断, **当** 用户重新运行同一条命令, **那么** 已成功写出的 .pkl 必须**跳过**, 仅处理未完成的视频 (幂等性, 与 `pp data-prepare` 的 cos 模式一致).

---

### 用户故事 3 - 在新拍视频上微调现有 BMN 模型 (优先级: P3)

作为高阶用户, 我已经有 v0.2.x 训出的 BMN 基线 (在 AI Studio 竞赛 #127 数据上), 我想用 US2 产出的"我自己拍的"少量数据 (例如 50 段视频带 GT JSON) 微调这个基线, 在我的领域 (家用环境) 上提点儿性能.

**优先级原因**: 严格说这只是 US2 + 现有 `pp train --resume` 的组合, 不需要新写代码; 但需要在 spec 里**明确"原始视频参与训练"是 supported 路径**, 不能因为缺少业务流程文档让用户卡住.

**独立测试**: 跑 US2 把 5 段视频转 pkl + 写 mini label.json, 然后 `pp train --config configs/models/bmn_pingpong.yaml --resume <baseline>.pdparams --allow-dirty --train-list-override mini_label.json` 跑 1 epoch, 看 loss 起点 ≤ 基线在原数据上的 final loss + 0.2 (即没有从零学起, resume 起作用了).

**验收场景**:

1. **给定** US2 已生成的 `<out>/Features_<name>/*.pkl` + `label_cls14_<name>.json`, 加上 v0.2.x 训好的 baseline ckpt, **当** 用户跑 `scripts/prepare_bmn_inputs.py --label-json <new>.json --feature-dir <out>/Features_<name>/ --output-dir data/bmn_inputs/<custom>/` 然后 `pp train --resume <baseline> --config configs/models/bmn_pingpong.yaml --override dataset.bmn_inputs_dir=data/bmn_inputs/<custom>/ --allow-dirty`, **那么** 训练可以启动, 第一个 step loss 显著低于从随机权重开始的水平 (即 resume 生效).

---

### 边界情况

- 当输入视频 fps 不是 25 (例如手机录制 30 fps 或 60 fps) 时, 系统应用 ffmpeg 强制重采样到 25 fps (与上游 BMN GT 一致), 不报错; 但日志须明确告知用户做了重采样, 避免用户困惑"为什么我视频长度不一样了".
- 当输入视频长度过短 (< 8 秒, 即 BMN window) 时, 系统应跳过该视频并日志记录, 不要硬塞到 BMN (因为 BMN 至少需要 1 个完整 window); 退出码 0 (其他视频继续), 但 manifest.csv 中相应行的 `error` 列写明.
- 当 GPU 内存不足 (例如 batch_size=32 在小卡上跑 OOM) 时, 系统应自动二分 batch_size 重试一次, 日志告警; 第二次仍 OOM 才退出码 4.
- 当用户传的 `--gt-json` 中的 url 在 videos-dir 里找不到对应 mp4 时, 系统在校验阶段就抛错 (退出码 1) 而不是抽完特征再发现, 避免浪费 GPU 时间.
- 当 ppTSM_k400_dense.pdparams 已下载但 sha256 校验失败时, 系统不得继续, 退出码 2 + 提示用户重新下载 (避免用错权重得到无意义特征).
- 当抽帧产生的总帧数与 fps × 视频时长偏差 > 5% 时, 警告但继续 (ffmpeg 在某些容器格式上的 fps 探测会有小偏差).

## 需求 *(必填)*

### 功能需求

#### 抽特征 (PP-TSM → 2048-d) 与 pkl 序列化

- **FR-033**: 系统必须提供一条独立 CLI 命令 `pp extract-feat`, 接受单个 mp4 路径, 输出一个 `<hash>.pkl` 文件, 内容是 `{'image_feature': ndarray(N, 2048), dtype=float32}`, 与 `Features_competition_train.tar.gz` 内部 .pkl 完全 schema-兼容. 默认输出路径为输入视频的同名 .pkl (相同目录).
- **FR-034**: 系统必须提供 `pp build-feature-pkls` 命令 (US2): 接受目录路径, 批量转 mp4 → .pkl, 写到 `<out>/Features_<name>/`, 并产出 `manifest.csv` (列见 US2 验收). 必须**幂等**: 已存在且**目标 .pkl 文件名 (= sha256(video_bytes)[:32]) 命中**的视频跳过 (改名重压缩则视为新视频, 这是预期行为).
- **FR-035**: 抽帧策略必须严格对齐上游: ffmpeg `-r 25 -q 0 %08d.jpg` (或等价 PIL/decord), 短边 resize 256, center crop 224×224, ImageNet mean/std 标准化 ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).
- **FR-036**: PP-TSM 模型加载必须从上游 `applications/TableTennis/extractor/configs/configs.yaml` (或等价的 FootballAction `configs/configs.yaml`) 派生; 不允许在源码中硬编码 `seg_num=8 / seglen=1 / batch_size=32`. 业务 yaml `configs/models/pp_tsm_extractor.yaml` 显式覆盖时, 以业务 yaml 为准.
- **FR-037**: PP-TSM inference predictor 必须取**第二个输出** (`output_names[1]`), 即 ResNet50 全局池化后的 2048-d 特征, 而**不是** logits[1, 400] (Kinetics-400 类别预测). 单元测试必须验证一个 mp4 → pkl → `pkl['image_feature'].shape[1] == 2048`.
- **FR-038**: 必须提供 PP-TSM 训练权重的自动下载入口 (类似 US5 `pp infer-pkl` 处理 VideoSwin_tennis.pdparams 缺失的方式): 缺失时退出码 1 + stderr 给出 `curl -fL -o <expected_path> <BCEBOS_URL>` 一行. URL 是 `https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams`. 默认落点 `data/raw/pretrained/ppTSM_k400_dense.pdparams`.
- **FR-038a**: 系统必须提供 `scripts/export_pptsm_inference.py`, 把 `ppTSM_k400_dense.pdparams` (动态图 state_dict) 通过 `paddle.jit.to_static + paddle.jit.save` 转换为上游 `extract_feat.py` 期望的 inference 双文件 (`ppTSM.pdmodel` 静态图结构 + `ppTSM.pdiparams` 静态图权重), 默认输出到 `data/raw/pretrained/ppTSM.{pdmodel,pdiparams}`. 当 `pp extract-feat` 首次运行检测到 inference 文件不存在时, **自动调用**该脚本 (无需用户手动跑); 后续运行直接复用缓存的 inference 文件 (与章程 VII "端到端 ≤ 5 条命令" 对齐).

#### 端到端原始视频推理 (US1)

- **FR-039**: 系统必须提供 `pp infer-rawvideo --input <mp4> --bmn-checkpoint <ckpt> --output-dir <out>` 端到端命令, 内部链路是 `ffmpeg → pp extract-feat → BMN forward → BMN post-processing → JSON timeline + 可视化`.
- **FR-040**: 该命令的输出必须含 (a) `<out>/timeline.json` (ActivityNet 1.3 风格 + 14 类 display_name 中文), (b) `<out>/<input_name>_visualized.mp4` (叠加候选区间文本; FR-015 复用), (c) `<out>/feature.pkl` (中间产物, 默认 `--keep-features` 保留, 便于断点续算 / 调参).
- **FR-041**: 该命令必须默认在结束前清理临时帧目录, 但提供 `--keep-frames` 选项让用户保留 (用于调试 / 验证抽帧质量).
- **FR-042**: BMN 推理路径在内部复用现有 `models/bmn.py + scripts/prepare_bmn_inputs.py + cli/eval.py::_run_bmn_eval` 的全部代码 — 不允许在 `pp infer-rawvideo` 中**重复实现**这些逻辑. 重构若必要, 应在原模块中暴露公共 API, 而非分叉.

#### 训练数据扩充 (US2/US3)

- **FR-043**: `pp build-feature-pkls` 必须支持可选 `--gt-json <path>`: 若提供, 系统验证 JSON 中每个 `url` 字段都能在 `--videos-dir` 中找到 (退出码 1 + 列出缺失项), 然后写一份 `<out>/label_cls14_<name>.json` 把 url 字段替换为 `<sha256(video_bytes)[:32]>.mp4` (与 .pkl 名同源 — 见 RawVideo 实体), 这样就能直接拼接到现有 `pp data-prepare` 的 cos 拉下来的 label JSON 后续训练.
- **FR-044**: 当用户**没有**提供 `--gt-json` 时, 命令仍可运行 (只产出 .pkl, 不写 label.json). 这种 .pkl 适用于推理 (US1), 但不能直接 BMN 训练 (因为缺标签).
- **FR-045**: 训练侧 `pp train` / `prepare_bmn_inputs.py` 不需要任何修改; US3 验收场景 1 是组合现有命令 (训练命令已在 v0.2.x 支持 `--resume`), 不需要新代码. 但本规范必须**明确文档化**这条路径, 在 README 或 docs 中给出"用我的数据微调基线"的 5 行示例.

#### 章程对齐与稳定性

- **FR-046**: `pp extract-feat` / `pp build-feature-pkls` / `pp infer-rawvideo` 必须接受 `--allow-dirty` 标志, 在 git 工作区脏时仍允许运行 (与 `pp train` / `pp eval` 一致).
- **FR-047**: 三个命令的退出码必须严格遵守现有约定: 0 成功 / 1 用户输入错 / 2 环境问题 (ffmpeg 缺失、PP-TSM 权重 sha256 校验失败、CUDA 不可用) / 3 章程硬约束违反 (例如标签数据缺失而又要求训练) / 4 运行时失败.
- **FR-048**: 命令必须在 manifest.csv (US2) 与 timeline.json (US1) 中**透传**至少一份完整的产出元信息: PP-TSM 权重的 sha256 / pp_tsm config_hash / 抽帧 fps / 命令版本号 (commit hash) — 与章程 II 的可复现实验四元组思想一致.
- **FR-049**: 所有抽帧临时目录必须落在 `data/raw/.tmp/extract_<run_id>/` (默认), 不允许散落到系统 `/tmp` 或 `~/.cache`. `data/raw/.tmp/` 加入 `.gitignore`, 命令结束 (成功或失败) 都应清理.

### 关键实体 *(如果功能涉及数据则包含)*

- **RawVideo**: 一段 mp4 / avi / mov 文件. 字段: `path`, `clip_id` (= `sha256(file_bytes)[:32]`, 流式 hash 整文件), `fps_original`, `duration_sec`, `n_frames`. **关系**: 1:1 → ImageFeaturePkl (相同 clip_id 即同一视频, 跨机器/跨改名稳定).

- **ImageFeaturePkl**: pickle 文件, 内容 `{'image_feature': ndarray(N, 2048) float32}`. 字段: `path` (对应 RawVideo.clip_id 的 .pkl), `n_frames` (= ndarray.shape[0]), `feat_dim` = 2048, `extracted_at`, `pp_tsm_config_hash`, `pp_tsm_weight_sha256`. **关系**: 1:1 ← RawVideo, 1:1 → BmnSlidingWindowSlice (经过 prepare_bmn_inputs.py).

- **TimelineSegment** (已存在, US3): 字段 `start_sec / end_sec / label_id / label_name / score`. 本 feature 不修改其 schema, 只是增加来源 (从原始视频).

- **PPTSMExtractorWeights**: 两个文件:
  - **训练权重** `ppTSM_k400_dense.pdparams` (BCEBOS 公开下载, ~120 MB). 字段: `path`, `sha256`, `source_url`, `downloaded_at`. 默认落点 `data/raw/pretrained/ppTSM_k400_dense.pdparams`.
  - **Inference 双文件** `ppTSM.pdmodel + ppTSM.pdiparams` (本地从训练权重转换, 见 FR-038a). 字段: `pdmodel_path`, `pdiparams_path`, `derived_from_sha256` (= 训练权重 sha256), `exported_at`, `paddle_version`. 默认落点 `data/raw/pretrained/ppTSM.{pdmodel,pdiparams}`. **关系**: 1:1 ← 训练权重 (一对动态图权重导出一对静态图文件, 转换是确定性的).

## 成功标准 *(必填)*

### 可衡量的结果

- **SC-010**: 用户拿到本项目 v0.3.x 后, 在 ≤ 5 min 内 (含权重下载) 能用 1 条命令把任意 5 分钟乒乓球 mp4 跑出 timeline.json (US1). 实测门槛: PP15pingskills `2.正手攻球.flv` (5.2 MB, ~3 分钟) 端到端 ≤ 5 分钟在 T4 上完成, 输出 ≥ 5 条候选区间.
- **SC-011**: PP-TSM 抽特征吞吐: ≥ 80 帧/秒 (T4, batch_size=32, 224×224). 即 100 分钟 25fps 视频 (= 150000 帧) ≤ 31 分钟抽完, 远小于视频本身时长.
- **SC-012**: 抽出的 pkl 与 AI Studio 竞赛 #127 提供的 pkl **数值上等价**: 把 `Features_competition_train/0018d6cbdf1f43f1a8a6d801b847f326.pkl` 的源视频 (假设可获得) 重抽特征, 然后比较两个 ndarray, cosine similarity ≥ 0.99 (允许 numerical noise + 不同 ffmpeg 版本的 ε). 若源视频不可获得, 则退化为 SC-013.
- **SC-013**: 跨视频抽特征结果一致性: 同一段视频在两台机器 (相同 PP-TSM 权重 + 相同 ffmpeg 版本) 上抽特征, ndarray 必须 bit-wise 一致 或者 cosine ≥ 0.999. 这保证 manifest.csv::sha256(image_feature) 可作为去重 key.
- **SC-014**: US3 微调可观测: 用 5 段额外视频微调基线 BMN 1 epoch 后, 在原 val set 上 AR@100 不低于基线 -2% (即"没把模型搞坏"); 在新 5 段视频自身的 holdout 上 AR@100 ≥ 50% (即"学到了新视频的分布"). 业务硬指标延后到 US3 完整迭代时再定.
- **SC-015**: 章程兼容: 4 个上游 patches (paddle.fluid / decord / inspect / record) 仍生效, 不需要新增第 5 个 patch (即抽特征代码完全在本仓库实现, 不修改 PaddleVideo 源码).

### 不在范围内 (Out of Scope)

- **不**新增任何上游 PaddleVideo patch (FR-047 + SC-015 已界定).
- **不**实现"在线抽特征 + 即时推理" (流式 / WebSocket); 本期是文件 → 文件批处理.
- **不**支持音频特征 / OCR 字幕 / 多模态融合 (上游 FootballAction `extract_feat.py` 中 `audio_feature` / `pcm_feature` 字段直接置空, 与现有 `Features_competition_train.tar.gz` 一致).
- **不**支持非 14 类的标签体系 (要换标签必须重新训 BMN, 那是另一个 spec).
- **不**支持 PP-TSM 之外的特征提取器 (例如 Kinetics-700 预训练 / SlowFast); 后续需要再开新 spec.

## 假设

- 用户机器有 ffmpeg 命令, version ≥ 4.0 (env-check 现有自检会确认).
- 用户机器单卡 GPU 显存 ≥ 8 GB (T4 / V100 / 3060+); CPU 模式仅作环境验证回退, 业务 SC 全部基于 GPU.
- 上游 `ppTSM_k400_dense.pdparams` URL `https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams` 持续有效 (已实测 200 OK; 与 VideoSwin_tennis 同源).
- `Features_competition_train.tar.gz` 内的 .pkl 是用与 ppTSM_k400_dense 相同模型抽出的 (SC-012 等价性的前提). 若发现上游用了某个未公开的 PP-TSM 变体, 退化到 SC-013.

## 依赖

- **前置功能**: v0.2.x 的 US6 (COS 接入 + BMN 训练 + BMN eval) 必须已合入并稳定. 本 feature 复用 `models/bmn.py + cli/eval.py + scripts/prepare_bmn_inputs.py + 4 patches`.
- **新依赖**: 无 — ffmpeg / numpy / paddle / Pillow 都已在 `requirements/base.txt`.
- **PP-TSM inference 模型**: 由 `scripts/export_pptsm_inference.py` (FR-038a) 在首次运行 `pp extract-feat` 时从 `ppTSM_k400_dense.pdparams` 自动转换得到, 缓存到 `data/raw/pretrained/`. 转换基于 `paddle.jit.to_static + paddle.jit.save`, 是确定性、幂等的; 第二次运行直接复用. 用户唯一需要的下载是 `ppTSM_k400_dense.pdparams` (FR-038).
