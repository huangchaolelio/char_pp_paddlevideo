# 乒乓球动作识别 端到端架构与流程文档

**日期**: 2026-05-13
**版本**: 对应 v0.3.0+ (commit `f291496` 时记录)
**适用范围**: 本仓库整个生命周期 — 数据准备 / 训练 / 评估 / 推理 / 持续迭代

本文档是一份**横向汇总**, 面向新加入的工程师 / 业务方 / 评审者. 不替代各
spec 的详细规格, 而是把它们组织成一张可执行的全景图.

读完本文你可以回答:
1. 这个项目能做什么
2. 模型架构是什么 (有几个模型, 各自负责什么)
3. 从一段原始视频到 timeline.json 经过了哪些步骤
4. 训练数据从哪来, 评估指标怎么算
5. 怎么在自己的视频上跑 / 怎么扩展数据 / 怎么微调
6. 哪些部分是基线 (frozen), 哪些是可优化的

---

## 0. TL;DR

```
┌────────────────────────────────────────────────────────────────────┐
│  乒乓球动作识别系统 (本仓库 v0.3.0+)                                 │
│                                                                      │
│  输入: 任意 mp4 / avi / mov / flv (任意分辨率, 任意 fps)             │
│  输出: timeline.json — 时间区间 + 14 类乒乓球动作分类 + 置信度        │
│                                                                      │
│  端到端命令 (用户视角 1 条):                                         │
│    pp infer-rawvideo --input my.mp4 --bmn-checkpoint <ckpt> \        │
│                      --prototypes <proto.npy> \                      │
│                      --output-dir out/                               │
│                                                                      │
│  典型耗时: 5 分钟视频 ≤ 5 分钟 (T4 单卡, batch_size=32 独占 GPU 时    │
│            实际 ~1.5 倍实时; 与训练共享 GPU 时退化到 0.83 倍实时)     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 1. 业务问题 & 系统能力

### 1.1 要解决的问题

给定一段乒乓球比赛 / 教学视频, 自动标出**何时发生了什么动作**, 输出一张
"动作时间轴" 让用户/教练/评审快速定位关键技术片段.

### 1.2 输入输出

| 角色 | 输入 | 输出 |
|------|------|------|
| 端到端用户 | 任意 mp4 (1-60 分钟典型) | `timeline.json` (≤ 50KB) + 可视化 mp4 (可选) |
| 数据扩充用户 | 视频目录 + GT JSON | 兼容上游训练管线的 .pkl + label_cls14.json |
| 模型微调用户 | baseline ckpt + 上一步的输出 | 新 ckpt (在 baseline 基础上 finetune) |

### 1.3 14 个动作类别 (业务定义)

源自 AI Studio 竞赛 #127 数据集 (`label_cls14_train.json`):

| id | 中文名 | 训练实例数 | LOO 原型分类准确率 |
|---:|-------|---------:|------------------:|
| 0 | 摆短     |  2003 | 71.4% |
| 1 | 拉      |  6501 | 41.0% (大类, 易被其它类误归到此) |
| 2 | 控制     |  2015 | 72.0% |
| 3 | 侧身拉   |  2697 | **88.7%** |
| 4 | 劈长     |   460 | 50.9% |
| 5 | 拧      |  1005 | **85.8%** |
| 6 | 挑      |   409 | 51.6% |
| 7 | 侧旋     |  2911 | 20.3% (易混到"拉") |
| 8 | 转不转   |   411 | 63.8% |
| 9 | 中性     |    64 | 15.6% (小类) |
| 10 | 勾球     |    23 | 0.0% (太小) |
| 11 | 普通     |   206 | 13.1% |
| 12 | 逆旋转   |   306 | 3.6% |
| 13 | 下蹲     |    43 | 55.8% |

**总计 19054 个标注**, 严重类不平衡 (max:min = 6501:23 ≈ 283:1).

---

## 2. 系统架构 (高层)

### 2.1 模型三件套

```
┌───────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│  PP-TSM           │    │  BMN               │    │  Prototype         │
│  Feature Extractor│ →  │  Temporal Localizer│ →  │  Classifier        │
│                   │    │                    │    │                    │
│  ResNet50         │    │  3D-conv 预测      │    │  14 个 2048-d 均值 │
│  + Kinetics-400   │    │  (start/end/iou)   │    │  + cosine argmax   │
│  pretrained       │    │  + NMS post-proc   │    │                    │
│                   │    │                    │    │                    │
│  pretrained       │    │  本仓库**训练**    │    │  从 GT 离线**计算**│
│  (上游 BCEBOS)    │    │  (AI Studio #127)  │    │  (无训练)          │
└───────────────────┘    └────────────────────┘    └────────────────────┘
       2048-d                  proposals                  label
       per frame (or 8)        [start_sec, end_sec,       label_name
                                score]                    cls_score
                                                          top-3
```

### 2.2 各组件职责

| 组件 | 输入 | 输出 | 状态 | 来源 |
|------|------|------|------|------|
| **PP-TSM 特征提取器** | 视频帧 (224×224) | (N, 2048) feature | 🔒 frozen (pretrained) | 上游 BCEBOS `ppTSM_k400_dense.pdparams` |
| **BMN 时序定位** | (N, 2048) feature | proposals (start, end, score) | 🟢 已训 (epoch 13/20 进行中) | 本仓库训练 v0.2.0 |
| **原型分类器** | feature 段 + 14 原型 | label_id + topk | 🟢 已构建 (v0.3.1) | 本仓库 `scripts/build_action_prototypes.py` |
| **可视化渲染** | timeline.json + 原视频 | mp4 (叠加文本) | 🟢 已实现 (v0.1.x US3) | 本仓库 `inference/visualizer.py` |

### 2.3 数据通过的形态变换

```
原始视频.mp4
     │ ① ffmpeg -r 25 -q 2 (强制重采样)
     ▼
帧序列 jpg (临时, data/raw/.tmp/extract_<run_id>/)
     │ ② PP-TSM ResNet50 forward, 静态图 Predictor
     │    每 8 帧 → 1 个 2048-d 特征向量 (seg_num=8)
     ▼
特征 pkl  {'image_feature': ndarray(N_samples, 2048) float32}
     │ ③ scripts/prepare_bmn_inputs.py::for_inference
     │    切 8s 窗口 + 线性插值到 tscale=200
     ▼
.npy 滑窗 (data/bmn_inputs/<x>/feature/<clip>_<t0>_<t1>.npy)
     │ ④ BMN forward + 12 进程 NMS 后处理
     │    (paddle.inference.Predictor + BMNMetric.accumulate)
     ▼
proposals (start_sec, end_sec, score) — ActivityNet 1.3 格式
     │ ⑤ 阈值过滤 + min_duration 过滤
     ▼
filtered proposals (典型 1-100 个 / 视频)
     │ ⑥ 对每个 proposal:
     │    a. 切对应特征段
     │    b. mean(axis=0) → 2048-d
     │    c. L2 normalize, cosine vs 14 prototypes
     │    d. argmax → label_id + cls_score + top-3
     ▼
timeline.json (rawvideo-timeline-v1)
{
  "schema": "rawvideo-timeline-v1",
  "results": [
    {start_sec, end_sec, label_id, label_name,
     score, cls_score, topk[3], rank_in_window}
  ],
  "extraction": {pp_tsm_sha256, fps, n_frames, ...},
  "bmn_inference": {checkpoint_sha256, n_proposals, ...},
  "classifier": {prototypes_path, n_classes, loo_acc, ...}
}
```

---

## 3. 数据流详解

### 3.1 训练数据来源 (一次性)

```
腾讯云 COS bucket (ap-guangzhou)
  charhuang-pp-1253960454/pp_video/
  ├── Features_competition_train.tar.gz (43.5 GB)
  │     └── 729 个 <clip_id>.pkl
  │           └── {'image_feature': ndarray(~9000, 2048)}
  │              (上游 PP-TSN 抽好的, 约每帧 1 特征)
  └── label_cls14_train.json (1.6 MB)
        └── {'fps': 25,
             'gts': [
               {'url': '<clip_id>.mp4', 'total_frames': N,
                'actions': [
                  {'label_ids': [0..13], 'label_names': [...],
                   'start_id': float_sec, 'end_id': float_sec},
                  ... 19054 entries total
                ]}, ...
             ]}
```

通过 `pp data-prepare --config configs/datasets/pingpong_competition_bmn.yaml`
拉取并解压. 凭据从 `.env` (COS_REGION/COS_BUCKET/COS_SECRET_ID/COS_SECRET_KEY).

### 3.2 训练数据准备

```
data/raw/pingpong_competition/pingpong_competition_bmn/
  ├── Features_competition_train.tar.gz   (从 COS 下载, gitignored)
  └── label_cls14_train.json              (从 COS 下载)
       │
       │ tar 流式解压 (r|gz, 单遍)
       ▼
data/clips/pingpong_competition/pingpong_competition_bmn/
  └── Features_competition_train/
      └── *.pkl (729 个)
       │
       │ scripts/prepare_bmn_inputs.py for_training
       ▼
data/bmn_inputs/pingpong_competition/
  ├── feature/ (18074 个 .npy 滑窗, 各 (200, 2048))
  ├── label_fixed.json (BMN 训练格式)
  └── label_gts.json (BMN 评估格式)
```

### 3.3 训练循环

```
.venv/bin/pp train --config configs/models/bmn_pingpong.yaml --allow-dirty
  │
  │ 内部流程:
  │   1. snapshot 业务 yaml → manifest 四元组 (commit/config_hash/seed/ckpt)
  │   2. 合并到上游 bmn_tabletennis.yaml 模板
  │   3. paddlevideo.tasks.train_model(cfg)
  │   4. 每 epoch 末写 BMN_epoch_NNNNN.pdparams
  ▼
experiments/<timestamp>-<commit>-train-bmn_pingpong/
  ├── BMN_epoch_00001.pdparams ~ BMN_epoch_00020.pdparams
  ├── BMN_best.pdparams (loss 最低 epoch 的副本)
  ├── manifest.json (章程 II 四元组 + 训练指标 summary)
  ├── config.yaml (业务 snapshot)
  ├── upstream_config.yaml (上游 yaml snapshot)
  └── log/                                  (paddle 自身日志)
```

**典型训练耗时 (T4 GPU)**:
- 1 epoch ≈ 70 分钟 (4027 steps × 1.07s/step, batch=4)
- 完整 20 epoch ≈ 24 小时
- Loss 曲线: epoch 1 起点 2.59 → epoch 13 中位 0.58 (实测)

### 3.4 原型构建 (US3 / v0.3.1, 一次性)

```
.venv/bin/python scripts/build_action_prototypes.py
  │
  │ 内部流程:
  │   1. 加载 label_cls14_train.json (729 视频 / 19054 actions)
  │   2. for each action (start_sec, end_sec, label_id):
  │        切对应视频 .pkl 中的特征段 → mean(axis=0) → 2048-d
  │   3. 按 label_id 聚合 → 14 个原型 (per-class mean)
  │   4. Leave-one-out cross-validation 评估 → 53.4% Top-1
  ▼
data/raw/pretrained/prototypes/
  ├── action_prototypes_14.npy (112 KB, (14, 2048) float32)
  └── action_prototypes_14.meta.json (准确率 + 类别计数 + 来源审计链)
```

**耗时**: ~6 分钟 (CPU only, 无 GPU 需求).

### 3.5 推理 (用户视角)

```
my_pingpong_video.mp4 (任意分辨率/fps)
  │
  │ pp infer-rawvideo --input my.mp4 --bmn-checkpoint ... --prototypes ...
  │
  │   阶段 1: 抽特征 (PP-TSM)
  │     ffmpeg -r 25 抽帧到 data/raw/.tmp/extract_<run_id>/
  │     PP-TSM forward → feature.pkl (873, 2048) for 5min 视频
  │     写 feature.meta.json (clip_id + sha256 + config_hash)
  │
  │   阶段 2: 切 BMN 滑窗
  │     prepare_bmn_inputs_for_inference: 8s 窗口 + 线性插值到 (200, 2048)
  │     写 dummy label_fixed.json (BMN dataloader 要求, annotations=[[0.0, 0.0]])
  │
  │   阶段 3: BMN 推理
  │     run_upstream_bmn_eval(gt_required=False)
  │     12 进程 NMS 后处理 → bmn_results_validation.json
  │     proposals: [{segment: [start, end], score: 0.0~1.0}, ...]
  │
  │   阶段 4: 阈值过滤
  │     score >= threshold (默认 0)
  │     end_sec - start_sec >= min_duration (默认 0.3s)
  │
  │   阶段 5: 原型分类 (新, v0.3.1)
  │     对每个过滤后的 proposal:
  │       从 feature.pkl 切对应秒段
  │       mean(axis=0) + L2 norm
  │       cosine vs 14 prototypes
  │       argmax → label_id, cls_score, top-3
  │
  │   阶段 6: 写 timeline.json (rawvideo-timeline-v1 schema)
  │
  │   阶段 7 (可选): 可视化
  │     渲染 <input>_visualized.mp4 (cv2 + drawtext, 失败不阻塞主输出)
  ▼
out/
  ├── timeline.json (主产物, ≤ 50KB)
  ├── feature.pkl (中间产物, 7MB for 5min video)
  ├── feature.meta.json
  ├── bmn_input/ (BMN 滑窗 .npy)
  ├── bmn_eval/ (BMN proposals + 后处理)
  └── <input>_visualized.mp4 (可选)
```

---

## 4. 评估指标

### 4.1 训练 / 评估 (有 GT)

`pp eval --checkpoint <ckpt> --split val` 走 BMN 时序定位评估路径
(schema = `bmn-eval-v1`):

| 指标 | 含义 | v0.2.x epoch 8 实测 |
|------|------|--------------------|
| **AR@1** | Average Recall @ 1 proposal | 29.23% |
| **AR@5** | Average Recall @ 5 proposals | 59.24% |
| **AR@10** | Average Recall @ 10 proposals | 67.91% |
| **AR@100** | Average Recall @ 100 proposals | 80.67% |
| **AUC** | Area Under AR vs AN curve | 74.63% |
| **n_proposals** | 总产生的候选区间数 | 196700 |
| **n_videos** | 评估视频数 | 1967 |

### 4.2 分类器评估 (有 GT)

通过 `scripts/build_action_prototypes.py` LOO 验证:

| 指标 | 含义 | 实测 |
|------|------|------|
| **Top-1 LOO** | Leave-one-out top-1 accuracy | 53.4% |
| **Per-class** | 每类独立准确率 | 0% - 88.7% (类不平衡) |
| **Direct** | 不剔除自身的 baseline | 53.5% (与 LOO 几乎相同, 说明原型不过拟合) |

### 4.3 推理 (无 GT, 业务用户角度)

无标注真值, 只输出**置信度**和**用户主观评估**:
- BMN proposal score (0-1): 模型对"这是个动作区间"的置信度
- Cls score (cosine similarity, -1 to 1): 模型对"这是哪类动作"的置信度
- Top-3 候选: 让用户对模糊预测有备选

业务上, 用户可以靠 visualized.mp4 直接眼检命中率.

---

## 5. 仓库结构 (按职能)

```
char_pp_prj/
├── configs/
│   ├── datasets/
│   │   ├── pingpong_public.yaml          # 001 公开示例数据 (8 类)
│   │   ├── pingpong_competition_bmn.yaml # 002 主线: AI Studio #127 (14 类) ⭐
│   │   ├── pingpong_custom.example.yaml  # 用户自定义数据集模板
│   │   └── ucf101.yaml                   # 通用大视频数据集 (备选)
│   ├── models/
│   │   ├── pp_tsm_pingpong.yaml          # 001: PP-TSM 训练分类器 (8 类)
│   │   ├── pp_tsm_extractor.yaml         # 002: PP-TSM 抽特征 ⭐
│   │   └── bmn_pingpong.yaml             # 002: BMN 训练 ⭐
│   ├── inference/
│   │   └── sliding_window.yaml           # 001 长视频滑窗推理参数
│   └── examples/
│       └── upstream_smoke.yaml           # 上游 smoke 测试参数
│
├── src/pingpong_av/
│   ├── cli/                              # 8 个 CLI 子命令 (pp <subcmd>)
│   │   ├── env_check.py
│   │   ├── data_prepare.py
│   │   ├── train.py                      # 调用 paddlevideo train_model
│   │   ├── eval.py                       # 按 model.name 分支:
│   │   │                                 #   pp_tsm → metrics-v1
│   │   │                                 #   bmn    → bmn-eval-v1
│   │   ├── infer_video.py                # 001 长视频滑窗推理
│   │   ├── infer_pkl.py                  # 001 US5 上游样例 .pkl 推理
│   │   ├── extract_feat.py               # 002 单视频抽特征 ⭐
│   │   ├── build_feature_pkls.py         # 002 批量抽特征 ⭐
│   │   └── infer_rawvideo.py             # 002 端到端原始视频推理 ⭐⭐
│   │
│   ├── extractors/                       # 002 抽特征子系统 ⭐
│   │   ├── clip_id.py                    # sha256(file_bytes)[:32]
│   │   ├── ffmpeg_frames.py              # ffmpeg 抽帧 + ffprobe 探测
│   │   ├── manifest.py                   # CSV append-only thread-safe
│   │   ├── pp_tsm_inference.py           # paddle.inference.Predictor 包装
│   │   └── action_classifier.py          # v0.3.1 原型分类器 ⭐
│   │
│   ├── models/                           # 训练用模型适配
│   │   ├── pp_tsm.py                     # 001 PP-TSM yaml 加载
│   │   ├── bmn.py                        # 002 BMN yaml 加载
│   │   └── registry.py                   # model.name → loader 映射
│   │
│   ├── data/                             # 数据集预处理
│   │   ├── public_datasets.py            # url_list/local_dir/manual/cos 模式
│   │   ├── splitter.py                   # 按 source_video_id 划分
│   │   └── list_writer.py                # 写 train/val/test.txt + meta.jsonl
│   │
│   ├── inference/                        # 推理后处理
│   │   ├── post_process.py               # 滑窗结果合并
│   │   └── visualizer.py                 # cv2 渲染 visualized.mp4
│   │
│   ├── evaluation/                       # 指标计算
│   │   ├── metrics.py                    # top1/top5/per-class
│   │   └── reporter.py                   # confusion matrix
│   │
│   ├── experiment/                       # 章程 II 实验记录
│   │   └── run_manifest.py               # commit + config_hash + seed + metrics
│   │
│   ├── upstream_adapter/                 # PaddleVideo 接入层
│   │   ├── importer.py                   # 安全 import + 错误恢复
│   │   └── trainer.py                    # train / eval / infer 三个公共 API
│   │
│   └── utils/                            # 通用工具
│       ├── config.py                     # yaml 加载 + schema 校验 + hash
│       ├── env.py                        # find_repo_root, etc.
│       ├── logging.py                    # 结构化日志
│       └── seeding.py                    # 章程 II: 全局 seed
│
├── scripts/
│   ├── bootstrap.sh                      # 一次性初始化 venv + patches
│   ├── apply_upstream_patches.sh         # 4 个 patches 幂等应用
│   ├── prepare_bmn_inputs.py             # BMN 训练/推理输入准备 (双模式)
│   ├── export_pptsm_inference.py         # PP-TSM 训练权重 → inference 模型
│   ├── build_action_prototypes.py        # v0.3.1 原型构建 ⭐
│   └── wait_and_eval.sh                  # 训练完自动 eval 的 watcher
│
├── third_party/
│   ├── PaddleVideo/                      # submodule, release/2.2.0 @ da9a8ce8
│   └── patches/                          # 4 个最小侵入 patches
│       ├── 01-paddle-fluid-removal-py311.patch
│       ├── 02-decord-lazy-import-py311.patch
│       ├── 03-inspect-getargspec-py311.patch
│       └── 04-record-tensor-scalar-py311.patch
│
├── tests/
│   ├── unit/ (89 tests)                  # 业务代码单测, 全部秒级
│   ├── integration/ (37 tests, 7 slow)   # CLI 端到端 (--runslow 启用)
│   └── fixtures/
│       ├── mini_pingpong_5s.mp4          # 245KB testsrc2 合成视频
│       └── README.md
│
├── data/                                 # 全 gitignored
│   ├── raw/
│   │   ├── pingpong_competition/         # AI Studio 数据 (43.5GB)
│   │   ├── pretrained/                   # 模型权重 + 原型
│   │   ├── test_videos/                  # 用户测试视频 (smoke run 用)
│   │   └── .tmp/                         # 抽帧临时
│   ├── clips/                            # 解压后的 .pkl
│   ├── bmn_inputs/                       # BMN 训练/推理输入
│   ├── splits/                           # 数据划分 (入库, 章程 IV)
│   └── bmn/BMN_Test_results/             # 上游 anet_prop.py 硬编码输出
│
├── experiments/                          # gitignored, 每次训练/评估一目录
│   └── <timestamp>-<commit>-<task>/
│       ├── manifest.json                 # 章程 II 四元组
│       ├── config.yaml + upstream_config.yaml
│       ├── BMN_epoch_*.pdparams
│       ├── metrics.json                  # eval 输出 (bmn-eval-v1 schema)
│       └── log/
│
├── specs/
│   ├── 001-pingpong-action-recognition/  # 80 任务, 8 个 SC, v0.1.x → v0.2.x
│   └── 002-raw-video-feature-bmn/        # 31 任务, 6 个 SC, v0.3.0
│
├── docs/                                 # 实测记录 + 架构文档 (本文档)
│   ├── v0.3.0-real-video-smoke.md        # 13 unknown 现象记录
│   ├── v0.3.1-classifier-smoke.md        # 13 → "拉" 的分类闭环
│   └── architecture-and-pipeline.md      # 本文档 ⭐
│
├── pyproject.toml                        # 入口: pp = pingpong_av.cli:main
├── requirements/{base,upstream-py311,lock}.txt
├── .env.example                          # 凭据模板 (实际 .env gitignored)
├── .specify/                             # speckit 工件
│   ├── memory/constitution.md            # 章程 v1.1.0 (8 原则)
│   ├── templates/{spec,plan,tasks}-template.md
│   └── scripts/bash/                     # 6 个 spec 流程脚本
└── README.md                             # 顶层入口
```

---

## 6. 章程 (Constitution v1.1.0) 在系统中的体现

| # | 原则 | 落地点 |
|---|------|-------|
| **I** | 规范与计划优先 | `specs/` 目录, 每个 feature 都先 spec → clarify → plan → tasks |
| **II** | 可复现实验 | `experiments/<run>/manifest.json` 必含 commit + config_hash + seed + metrics 四元组 |
| **III** | 配置驱动, 拒绝硬编码 | 所有业务参数在 `configs/`; CLI 参数仅做 yaml override |
| **IV** | 数据完整性 | clip_id = sha256(file_bytes)[:32] 跨机器一致; splits 入库; test split 不做选型 |
| **V** | 评估纪律 | `pp eval` 必出 top1/top5/per-class/macro-avg (PP-TSM) 或 AR@AN+AUC (BMN) |
| **VI** | 上游最小侵入 | submodule + 4 patches (按字母序幂等应用); 未在源码中复制粘贴上游代码 |
| **VII** | 端到端 ≤ 5 命令 | quickstart.md 实际 4 条 (curl + extract_export + infer-rawvideo + 可选 build) |
| **VIII** | 隔离 Python 3.11 | `.venv/`, requirements 全部入库, `pp env-check --strict` 把关 |

---

## 7. 用户路径 (按角色)

### 7.1 路径 A: 我只想跑端到端推理 (业务最常用)

**前提**:
- 1 张 GPU (T4 起步)
- 已 `git clone + bash scripts/bootstrap.sh` (Python 3.11 venv + 4 patches)
- 有一份本仓库 v0.2.x 训过的 BMN ckpt (或问维护者要)
- 有一份本仓库 v0.3.1 构建过的原型 (`action_prototypes_14.npy`)

**4 步**:
```bash
# 1. 下载 PP-TSM 训练权重 (148 MB, 一次性)
mkdir -p data/raw/pretrained
curl -fL -o data/raw/pretrained/ppTSM_k400_dense.pdparams \
    https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams

# 2. (首次自动) 导出 PP-TSM inference 模型 (无需手跑, pp extract-feat 自动调用)

# 3. 推理
.venv/bin/pp infer-rawvideo \
    --input my_video.mp4 \
    --bmn-checkpoint experiments/<run>/BMN_epoch_00020.pdparams \
    --prototypes data/raw/pretrained/prototypes/action_prototypes_14.npy \
    --output-dir outputs/my_run/

# 4. 看结果
cat outputs/my_run/timeline.json | python3 -m json.tool
# (或在浏览器开 outputs/my_run/<input>_visualized.mp4)
```

**典型耗时**: 5min 视频 ≤ 5min (T4); 1080p 视频 + 共享 GPU 时 ~0.83x 实时.

### 7.2 路径 B: 我有自己的视频 + GT 标注, 想扩充训练集

```bash
# 1. 把视频目录批量抽特征 + 重写 GT JSON
.venv/bin/pp build-feature-pkls \
    --videos-dir my_videos/ \
    --output-dir data/clips/my_extension/ \
    --gt-json my_label.json \
    --name my_ext

# 2. 转 BMN 训练输入
.venv/bin/python scripts/prepare_bmn_inputs.py \
    --label-json data/clips/my_extension/label_cls14_my_ext.json \
    --feature-dir data/clips/my_extension/Features_my_ext/ \
    --output-dir data/bmn_inputs/my_ext/
```

输出:
- `data/clips/my_extension/Features_my_ext/<clip_id>.pkl` 与上游格式 100% 兼容
- `data/clips/my_extension/manifest.csv` 含完整审计链
- `data/clips/my_extension/label_cls14_my_ext.json` url 已重写为 clip_id

### 7.3 路径 C: 在我自己的视频上微调基线

接 B 的输出后:

```bash
# 1. 复制 bmn yaml 改 bmn_inputs_dir
cp configs/models/bmn_pingpong.yaml configs/models/bmn_my_ext.yaml
sed -i 's|bmn_inputs_dir: null|bmn_inputs_dir: data/bmn_inputs/my_ext/|' \
    configs/models/bmn_my_ext.yaml

# 2. 从 baseline 微调
.venv/bin/pp train \
    --config configs/models/bmn_my_ext.yaml \
    --resume experiments/<baseline_run>/BMN_epoch_00020.pdparams \
    --allow-dirty

# 3. 评估
.venv/bin/pp eval \
    --checkpoint experiments/<new_run>/BMN_epoch_NNNNN.pdparams \
    --split val
```

期望: resume 后 first step loss 显著 < 从 0 训; val AR@100 在原 val set 不低于
baseline -2% (避免遗忘), 在新 val set 上有提升.

### 7.4 路径 D: 我要做完整训练流水线 (从 0 开始)

**完整 quickstart**: 见 `specs/002-raw-video-feature-bmn/quickstart.md`. 关键步骤:

```bash
# 1. 环境
git clone <repo> && cd char_pp_prj
git submodule update --init --recursive
bash scripts/bootstrap.sh

# 2. 拉数据 (从 COS) — 需要 .env 凭据
.venv/bin/pp data-prepare --config configs/datasets/pingpong_competition_bmn.yaml

# 3. 转 BMN 输入
.venv/bin/python scripts/prepare_bmn_inputs.py

# 4. 训 (~24h on T4, 20 epochs)
.venv/bin/pp train --config configs/models/bmn_pingpong.yaml --allow-dirty

# 5. 评
.venv/bin/pp eval --checkpoint experiments/<run>/BMN_epoch_00020.pdparams --split val

# 6. 构建分类原型 (~6 min)
.venv/bin/python scripts/build_action_prototypes.py

# 7. 推理 (任意视频)
.venv/bin/pp infer-rawvideo --input my.mp4 --bmn-checkpoint <ckpt> \
    --prototypes data/raw/pretrained/prototypes/action_prototypes_14.npy \
    --output-dir out/
```

---

## 8. 已知限制 & 改进方向

### 8.1 当前已知限制 (诚实记录)

| 限制 | 影响 | 缓解 |
|------|------|------|
| **PP-TSM 抽法非 1:1** | 推理特征 (N/8, 2048), 与上游训练数据 (~N, 2048) 不同尺度 | cosine 对均值池化的降采样不敏感; 实测仍工作但精度损失 ~10-20% |
| **类不平衡严重** | 6501:23 = 283:1 比例, 小类原型不可靠 | 当前小类准确率 0-15%; oversample 或 class-weighted 训练可改善 |
| **分类器仅 53% LOO** | Top-1 准确率不达业务期望 (80%+) | 003 候选: 训 MLP/LSTM head 替代原型 |
| **GPU OOM (1080p+训练共享)** | extractor 默认 batch=32 与 7GB 训练加起来超 T4 15GB | 提供 yaml override 把 batch 降到 4 + gpu_mem_mb=1500 |
| **`pp infer-rawvideo` 缺 `--batch-size`** | 用户必须改 yaml 才能调 batch | v0.3.1 patch-release 候选: 加 CLI flag |
| **BMN 仅"何时", 不"是什么"** | 原始 BMN 输出无 label | v0.3.1 已补上 (原型分类器); 003 训真 head |
| **14 类不含"快带"等细分** | 教学视频中常见动作只能映射到最近粗类 | 003: 与体育教练合作扩展 20+ 类标注 |

### 8.2 改进路径 (按优先级)

1. **003 feature: 训 MLP/LSTM 分类头** (2-3 天)
   - 用同样 19054 actions 训 14 类分类器
   - 期望 Top-1 80%+, 小类靠 weighted loss 提到 30%+
   - 输出 `models/action_classifier_v2.pdparams`, 替换原型方案

2. **`pp infer-rawvideo --batch-size N`** (30 分钟)
   - CLI 直接 override extractor yaml
   - v0.3.1 patch release

3. **数据扩展** (业务 + 标注成本)
   - 与体育教练合作扩展细类 (~30 类)
   - 加入 PP15pingskills 教学视频 (COS 上 26 个) 作多样化

4. **多模态融合** (研究)
   - 接入 audio (FootballAction 上游已有 audio_infer.py)
   - 击球声 + 视觉特征联合分类

5. **流式推理** (产品化)
   - 从文件→文件批处理转流式 (摄像头实时输入)
   - 核心改造: ffmpeg-streaming + PP-TSM batch=1 + 滑窗在线 NMS

---

## 9. 模型与权重一览

### 9.1 当前可用权重

| 名称 | 类型 | 来源 | 路径 | 大小 | 备注 |
|------|------|------|------|------|------|
| `ppTSM_k400_dense.pdparams` | 训练权重 (动态图) | BCEBOS 公开 | `data/raw/pretrained/` | 148 MB | Kinetics-400 预训练 |
| `ppTSM.pdmodel + ppTSM.pdiparams` | inference (静态图) | 本仓库导出 | `data/raw/pretrained/` | 258KB + 98MB | 含 monkey-patch 的 (feature, logits) 双输出 |
| `BMN_epoch_NNNNN.pdparams` | BMN 权重 | 本仓库训练 | `experiments/<run>/` | ~12 MB / epoch | 训练循环每 epoch 末写 |
| `action_prototypes_14.npy` | 14 类原型 | 本仓库构建 | `data/raw/pretrained/prototypes/` | 112 KB | LOO acc 53.4% |

### 9.2 章程 II 审计链

每份产物都可追溯到:
- **commit**: git rev-parse HEAD
- **config_hash**: yaml 内容 sha256[:16]
- **seed**: 训练随机种子 (默认 2026)
- **dataset_split_version**: 数据划分版本 (yaml::split_version)
- **pp_tsm_weight_sha256**: 抽特征权重身份证
- **pp_tsm_inference_sha256**: 派生 inference 文件身份证 (combined)

写入位置:
- `experiments/<run>/manifest.json` (训练 / 评估)
- `feature.meta.json` (抽特征)
- `manifest.csv` (批量抽)
- `timeline.json` (推理)

---

## 10. FAQ (常见问题)

### Q: 为什么不直接用上游 PP-TSM 做分类 (它本身就是 400 类分类器)?
A: 上游 PP-TSM 是 Kinetics-400 类别 (运动类: 跑步/跳跃/...), 与乒乓球 14 类完全不重合. 我们用 PP-TSM 的 backbone (2048-d 特征), 不用 head (logits[400]).

### Q: BMN 为什么需要 8s 滑窗?
A: 上游 BMN 设计假设 `tscale=200, dscale=200`, 即固定时长 = 200 / fps = 200/25 = 8s. 视频时长不是 8s 的整数倍时, 切多个 8s 滑窗 + 在线性插值到 200 帧.

### Q: 为什么我们的特征是 (N/8, 2048) 不是 (N, 2048)?
A: PP-TSM 的 `seg_num=8` 设计是"每 8 帧采 1 帧"作为 1 个时间段输入, 给 backbone 后输出 1 个特征. 上游 AI Studio 数据用了不同抽法 (sliding window seg_num=8 → 每帧 1 特征), 我们目前没复现这个细节. 实战中差异约 10-20% 精度.

### Q: 为什么训练数据和推理数据用了不同的特征采样率?
A: 历史原因. AI Studio 提供的预抽特征是 (~N, 2048), 与 BMN tscale=200 配套. 我们的 `pp extract-feat` 跟上游 FootballAction 的 extract_feat.py 一致 (seg_num=8). 这导致跨尺度. 如要严格对齐, 需把抽特征切成 sliding-1 模式 (每帧产 1 特征, 8 倍计算量).

### Q: 为什么不接上游 LSTM head 直接做分类?
A: 上游 FootballAction 的 LSTM head 没有公开训练好的乒乓球版本; 上游 TableTennis 应用只到 BMN 为止 (`extract_bmn_for_tabletennis.py` 是消费 .pkl, 不是分类器). 我们要么自己训 (003 feature), 要么用原型匹配 (v0.3.1, 已实现).

### Q: 如果我的 GPU 只有 8GB (3060), 能跑吗?
A: 可以. `pp extract-feat` 默认 `batch_size=32` 占 ~8GB, 改 yaml 到 `batch_size=8` 就能在 8GB 上跑. BMN 推理只占 1-2GB, 不是瓶颈. 训练不行 (BMN train batch=4 已经 7GB+, 8GB 只够推理).

### Q: 4 个 patches 都做了什么?
A: 见 `third_party/patches/README.md` 的表格. 简言之:
- 01: paddle.fluid → paddle.base (Paddle 2.6 移除了旧子包)
- 02: decord lazy import (Python 3.11 没 decord wheel)
- 03: inspect.getargspec → getfullargspec (Python 3.11 移除)
- 04: Tensor[0] → .item() (Paddle 2.6 0-d tensor 索引语义变化)

无新依赖, 无业务逻辑改动, 仅适配 Python/Paddle 版本.

---

## 11. 进一步阅读

| 主题 | 文档 |
|------|------|
| 完整规格 (001) | `specs/001-pingpong-action-recognition/{spec,plan,research,data-model,quickstart,contracts}.md` |
| 完整规格 (002) | `specs/002-raw-video-feature-bmn/{spec,plan,research,data-model,quickstart,contracts,tasks}.md` |
| 章程 | `.specify/memory/constitution.md` |
| 上游 patches | `third_party/patches/README.md` + 4 个 .patch 文件头注释 |
| v0.3.0 实测 | `docs/v0.3.0-real-video-smoke.md` |
| v0.3.1 实测 | `docs/v0.3.1-classifier-smoke.md` |
| 项目顶层 | `README.md` |
| Codebuddy 指南 | `CODEBUDDY.md` (项目状态 + 命令清单 + 章程速查) |

---

## 12. 致谢

- **上游**: PaddleVideo (release/2.2.0 @ da9a8ce8), 提供 PP-TSM / BMN 实现 + Kinetics-400 预训练权重 (BCEBOS 公开)
- **数据**: AI Studio 竞赛 #127 (乒乓球时序定位), 729 视频 + 19054 标注, 通过腾讯云 COS 共享
- **项目骨架**: speckit (`.specify/`) 的 spec → clarify → plan → tasks 工作流, 让所有决策可追溯

---

**维护者**: 持续迭代中. 如发现本文档与代码不一致, 以代码为准, 提 issue / 改文档.
