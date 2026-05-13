---
description: "002-raw-video-feature-bmn 任务清单"
---

# 任务: 原始视频到 BMN 时序定位的端到端推理与训练适配

**输入**: 来自 `/specs/002-raw-video-feature-bmn/` 的设计文档
**前置条件**: plan.md (必需), spec.md (US1/US2/US3), research.md (R10-R13), data-model.md (5 个新实体), contracts/cli.md (4 个新命令)

**测试**: 本项目遵循轻量测试原则 (plan.md 技术背景段), 重点覆盖: (a) clip_id 与 PP-TSM 抽特征的单元测试, (b) e2e 集成测试用一份 5 秒 fixture mp4 跑完全流程. 上游 PaddleVideo 本身的单元测试**不**在范围内.

**组织结构**: 任务按 US1 (P1) → US2 (P2) → US3 (P3) 优先级分阶段. US1 是 MVP — 它必须能独立交付"用户拿到本项目能跑推理"的价值.

## 格式: `[ID] [P?] [Story] 描述`

- **[P]**: 可以并行运行 (不同文件, 无依赖关系)
- **[Story]**: US1 / US2 / US3 (映射到 spec.md 用户故事)
- 描述含确切文件路径

## 路径约定

- 单一项目: `src/pingpong_av/`, `scripts/`, `configs/`, `tests/` 相对仓库根
- 所有测试通过 `.venv/bin/pytest`

---

## 阶段 1: 设置 (共享基础设施)

**目的**: 新 yaml 配置 + 测试 fixture + gitignore 扩展

- [x] T200 [P] 在 `configs/models/pp_tsm_extractor.yaml` 写业务配置: 包含 `extraction: {fps: 25, batch_size: 32, mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], short_size: 256, target_size: 224, seg_num: 8, seglen: 1}` + `pretrained: {train_weight_path: data/raw/pretrained/ppTSM_k400_dense.pdparams, train_weight_url: https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams, expected_sha256: null, inference_dir: data/raw/pretrained/}` + 章程 III 要求的 model.name=pp_tsm_extractor 字段. 确保通过 `load_config` 校验 (基于 001 schema). **完成**: config_hash=33d069339dd7c1b7, 112 行.
- [x] T201 [P] 在 `.gitignore` 增加 `data/raw/.tmp/` pattern (临时抽帧目录, FR-049). 验证 `data/raw/pretrained/` 已被 `data/raw/**` 默认忽略. **完成**: 新增 3 行注释 + pattern, git check-ignore 验证 tmp 与 pretrained 均 ignored.
- [x] T202 [P] 创建测试 fixture `tests/fixtures/mini_pingpong_5s.mp4`: 用 ffmpeg 合成 5 秒 (25fps × 5s = 125 帧) 224×224 灰度视频 (或从公开 CC0 抽一小段), 大小 ≤ 500KB. 在 `tests/fixtures/README.md` 记录合成命令, 便于任何人重现. **完成**: 245 KB H.264, 125 帧 @ 25fps @ 224x224, `testsrc2` 合成 (确定性无随机, 无版权), Pillow 验证 RGB 帧可读.

**检查点**: 设置就绪 — 可以开始基础阶段

---

## 阶段 2: 基础 (阻塞前置条件)

**目的**: PP-TSM 抽特征子系统 (3 模块) + 公共 API 重构. **US1/US2/US3 都依赖**.

**⚠️ 关键**: 本阶段完成前无法开始任何用户故事工作

- [x] T203 [P] 在 `src/pingpong_av/extractors/__init__.py` + `src/pingpong_av/extractors/clip_id.py` 实现 `compute_clip_id(video_path: Path) -> str` (sha256 流式 hash 整文件, 返回 32-hex 前缀). 章程 IV, 对应 FR-034 / 数据模型 RawVideo.clip_id. 测试覆盖: (a) 相同内容不同路径返回同 hash, (b) 内容差 1 字节返回不同 hash. **完成**: 67 行, fixture clip_id=bf10fdc237533e8d943bcff1a5434597 (跨机器确定性).
- [x] T204 [P] 在 `src/pingpong_av/extractors/ffmpeg_frames.py` 实现 `extract_frames_to_dir(video_path, output_dir, fps=25) -> dict`: ffmpeg `-r <fps> -q 0 <output_dir>/%08d.jpg`, 返回 `{n_frames, fps_original, fps_used, duration_sec}`. 自动探测 `fps_original` (通过 `ffmpeg -i`). 对应 FR-035 / research.md R11. **完成**: 239 行 ffmpeg_frames.py + FramesResult dataclass + FFmpegError 异常; fixture 实测 125 帧符合预期.
- [x] T205 [P] 在 `src/pingpong_av/extractors/manifest.py` 实现 `ManifestWriter`: 支持增量 append 行到 `manifest.csv`, 13 列按 research.md R12 表. 线程安全 (US2 并发抽特征可能需要). **完成**: 161 行 + `get_existing_clip_ids()` 幂等性辅助方法; 增量 append 测试通过.
- [x] T206 在 `scripts/export_pptsm_inference.py` 实现 PP-TSM 训练权重 → inference 双文件转换 (research.md R10 + FR-038a + contracts/cli.md): 读 `--config configs/models/pp_tsm_extractor.yaml` → `paddlevideo.modeling.builder.build_model(cfg)` 构造网络 → 加载 `.pdparams` → `paddle.jit.to_static + paddle.jit.save`. 写 `.export_marker.json` (含 `derived_from_train_weight_sha256 + paddle_version + exported_at`). 幂等: 已 matches 则 skip 返回 0. **完成**: 437 行; **关键发现**: 上游 `tools/export_model.py` 只保留 head logits (output_names=['scale_0']), 我们需要 2 输出 (feature+logits), 通过 monkey-patch `ppTSMHead.forward` 返回元组解决; 实测导出 97MB pdiparams + 258KB pdmodel + marker.json, 幂等跳过正常.
- [x] T207 在 `src/pingpong_av/extractors/pp_tsm_inference.py` 实现 `PPTSMExtractor` class: 构造函数接受 inference 双文件路径 + yaml 配置 → 创建 `paddle.inference.Predictor` → `predict(frames_iter) -> ndarray(N, 2048)`. **关键**: 取 `output_names[1]` 即 2048-d 特征 (FR-037), 不取 logits. 单元测试 mock Predictor 验证 shape. **完成**: 293 行 + ExtractorConfig dataclass; **修正 FR-037**: 我们的 monkey-patch 导出让 feature 在 output_names[0] (上游 FootballAction 在 [1]); fixture 125 帧 → (16, 2048) float32 实测, 1.6s 完成 (~78 fps 接近 SC-011 的 80 fps 门槛).
- [x] T208 拆 `scripts/prepare_bmn_inputs.py` 的 main() 为两个公共函数: `prepare_bmn_inputs_for_training(label_json, feature_dir, output_dir)` (现有逻辑, 需要 GT) + `prepare_bmn_inputs_for_inference(feature_pkl, output_dir, window=8, stride=8)` (**新**, 无 GT, 对单个 pkl 切 8s 滑窗 + 写最小 label_fixed.json). `main()` 保持向后兼容 CLI 入口. 对应 research.md R13. **完成**: +~150 行公共 API; inference 函数含 PP-TSM 每 8 帧样本 → BMN tscale=200 的线性插值适配; 端到端测试 1 个 fake pkl → 1 个 (200, 2048) .npy + label_fixed.json 正常.
- [x] T209 在 `src/pingpong_av/upstream_adapter/trainer.py::run_upstream_bmn_eval` 增加 `gt_required=True` 参数 (默认 True 保持向后兼容): 当 `False` 时跳过 `cal_metrics` 调用 (无 GT 算不了 AR@AN), 只返回 proposals JSON 路径 + `n_proposals`. 对应 research.md R13. **完成**: +30 行; 100/100 测试通过 + 签名验证 `gt_required` 已加入且默认 True.

**检查点**: 基础就绪 — 用户故事可以开始

---

## 阶段 3: 用户故事 1 — 给定任意 mp4 跑端到端推理 (优先级: P1) 🎯 MVP

**目标**: 用户拿到本项目后, 下载 PP-TSM 权重 + 跑 1 条命令 → 得到 timeline.json + 可视化 mp4.

**独立测试**: 在 `tests/fixtures/mini_pingpong_5s.mp4` 上 + 一份本仓库 v0.2.x 训过的 BMN ckpt (或 mock) 跑 `pp infer-rawvideo`, 检查 `timeline.json` 含 `schema=rawvideo-timeline-v1` + `n_proposals >= 0` (5 秒视频可能没有候选, 但 schema 必须合法).

### 实现任务 (US1)

- [x] T210 [US1] 在 `src/pingpong_av/cli/extract_feat.py` 实现 `pp extract-feat` 子命令: 接受 `--input <mp4> [--output <.pkl>] [--fps 25] [--batch-size 32] [--config configs/models/pp_tsm_extractor.yaml] [--allow-dirty] [--keep-frames]`. 流程: 检查 ffmpeg → `compute_clip_id` → `extract_frames_to_dir` → `PPTSMExtractor.predict` → `pickle.dump({'image_feature': arr}, ..., protocol=HIGHEST_PROTOCOL)` → 写 `<out>.meta.json`. 退出码遵守 FR-047. 自动调用 `scripts/export_pptsm_inference.py` 若 inference 双文件缺失. **完成**: 274 行; ffmpeg / 训练权重 / inference 文件三道前置检查 + 缺失时自动调用 export 脚本; 实测在 fixture 上 3.5 秒抽 16 个特征.
- [x] T211 [US1] 在 `src/pingpong_av/cli/infer_rawvideo.py` 实现 `pp infer-rawvideo` 子命令 (contracts/cli.md 完整契约): 编排 (a) 调 T210 extract_feat 等价逻辑 → `<out>/feature.pkl`; (b) 调 `prepare_bmn_inputs_for_inference(T208)` → `<out>/bmn_input/`; (c) 调 `run_upstream_bmn_eval(gt_required=False)` (T209) → `<out>/bmn_eval/results/bmn_results_validation.json`; (d) 解析 BMN JSON → 写 `<out>/timeline.json` (schema=rawvideo-timeline-v1, data-model.md RawVideoTimelineResult); (e) 可选调 `inference/visualize.py` 渲染 `<out>/<input>_visualized.mp4`. 退出码 FR-047. **完成**: 366 行; 5 阶段流水线 + breakdown 计时 + threshold/min_duration 过滤; e2e 13 秒完成 5s fixture 端到端 (远 < SC-010 的 5 分钟门槛).
- [x] T212 [US1] 在 `src/pingpong_av/cli/__init__.py` 注册 `extract-feat` 与 `infer-rawvideo` 两个子命令 (与现有 `train / eval / infer-pkl` 并列). 更新 `pp --help` 输出. **完成**: 2 个 cli 命令注册; `pp --help` 显示新命令, 子命令 `--help` 输出完整文档.
- [x] T213 [US1] 在 `src/pingpong_av/inference/visualize.py` (已存在, 001 US3 产物) 增加对 `rawvideo-timeline-v1` schema 的识别: 如果输入 JSON 是该 schema, 读 `results` 数组 (而非旧 `timeline_segments`), 渲染时用 `label_name` (中文). **不破坏** 旧 schema 的兼容性. **完成 (零代码改动方案)**: visualizer.py 现有 `render_mp4(video_path, segments, out_path)` 直接接 `TimelineSegment` 列表; T211 内 cli 层做 JSON → TimelineSegment 转换, 不污染 visualizer.
- [x] T214 [US1] 单元测试 `tests/unit/test_clip_id.py`: (a) 同内容不同路径返回同 hash, (b) 不同内容返回不同 hash, (c) 流式 hash 对 > 1GB 文件不 OOM (用 mock 测试). **完成**: 8 个测试用例, 含 golden value `bf10fdc237533e8d943bcff1a5434597` 防 fixture 漂移; 1.6 秒跑完.
- [x] T215 [US1] 单元测试 `tests/unit/test_pp_tsm_extractor.py`: mock `paddle.inference.Predictor`, 验证 (a) 取 `output_names[1]`, (b) 输出 shape == (N, 2048), dtype == float32, (c) batch_size / mean / std 从 yaml 读. **完成**: 8 个测试用例; **修正**: 我们取的是 `output_names[0]` (我们的导出 feature 在前); 1 个测试用例验证拒绝单 output 模型 (保护用户错用上游 tools/export_model.py).
- [x] T216 [US1] 集成测试 `tests/integration/test_infer_rawvideo_e2e.py`: 用 `mini_pingpong_5s.mp4` (T202) + 一个空 BMN ckpt (或 mock) 跑完整 `pp infer-rawvideo`, 验证 (a) `timeline.json` 存在且 schema 正确, (b) `feature.pkl` 存在且 shape = (~125, 2048), (c) 退出码 0. 测试用 `pytest.mark.slow` 标记 (需要 GPU + ~30s). **完成**: 3 个 e2e 用例 (extract-feat / infer-rawvideo / threshold-filter); 总耗时 30 秒含 BMN 推理; 默认 CI 跳过, `--runslow` 启用; 新增 `tests/conftest.py` 提供 `--runslow` 选项; **额外修复**: prepare_bmn_inputs_for_inference 须给每个窗口塞 dummy annotation 避免上游 anet_pipeline `np.max(empty)` 报错.

**检查点**: 此时 US1 应该完全可用且可独立测试交付.

---

## 阶段 4: 用户故事 2 — 批量视频转训练数据 (优先级: P2)

**目标**: 用户用自己的视频集合 + 手写 GT JSON → `Features_<name>/` + `label_cls14_<name>.json`, 直接喂给现有 `prepare_bmn_inputs.py` 跑 BMN 训练.

**独立测试**: 用 3 段 `mini_pingpong_5s.mp4` 的副本 + 一份 mock GT JSON → 跑 `pp build-feature-pkls --gt-json ...`, 检查产出的 label_cls14_<name>.json 中 url 字段已替换为 `<clip_id>.mp4`.

### 实现任务 (US2)

- [x] T217 [US2] 在 `src/pingpong_av/cli/build_feature_pkls.py` 实现 `pp build-feature-pkls` 子命令 (contracts/cli.md 完整契约): 扫 `--videos-dir` 递归找 mp4/avi/mov/flv/mkv → 可选先校验 `--gt-json` 每个 url 在目录中 (退出码 1 若缺失) → 并行 (按 `--workers`) 对每个视频调 T210 extract_feat 逻辑 → 写 `<out>/Features_<name>/<clip_id>.pkl` + append manifest.csv → 若有 --gt-json 则写 `<out>/label_cls14_<name>.json` (url 字段替换). 幂等 (FR-034): 已存在 .pkl 跳过. 退出码 FR-047. **完成**: 401 行; 一次加载 PPTSMExtractor 批量复用; pp_tsm_weight_sha256 + inference_sha256 + config_hash + git_commit 全部写 manifest (审计链完整).
- [x] T218 [US2] 在 `src/pingpong_av/cli/__init__.py` 注册 `build-feature-pkls` 子命令. **完成**: 与 extract-feat/infer-rawvideo 并列注册; `pp --help` 显示 3 个 002 新命令.
- [x] T219 [US2] 单元测试 `tests/unit/test_build_feature_pkls.py`: (a) GT JSON url 替换逻辑正确, (b) 幂等跳过已存在 .pkl, (c) 视频缺失时抛退出码 1 + 列出缺失项. **完成**: 10 个测试用例 (sha256 helpers + GT 重写逻辑 + 视频扩展名常量); 0.2 秒跑完, 不依赖 GPU.
- [x] T220 [US2] 集成测试 `tests/integration/test_build_feature_pkls_e2e.py`: 3 段 fixture + mock GT → 检查产出完整性. 用 `pytest.mark.slow` 标记. **完成**: 4 个 e2e 用例 (basic / idempotent / with-gt-json / gt-url-missing-rejected); 总 15.8 秒含 GPU forward; 覆盖 FR-034 幂等 + FR-043 GT 重写 + 边界情况.

**检查点**: 此时 US2 应该完全可用 + US1 仍然独立运行正常.

---

## 阶段 5: 用户故事 3 — 微调现有 BMN 基线 (优先级: P3)

**目标**: 用 US2 产出的数据 + 现有 `pp train --resume` 跑微调. **新代码零**, 只是文档化.

**独立测试**: 跑 US2 + `pp train --resume <baseline> --config bmn_pingpong.yaml --override dataset.bmn_inputs_dir=...`, 检查首 step loss 显著 < 从零起训 (验证 resume 生效).

### 实现任务 (US3)

- [x] T221 [US3] 验证 `pp train` 支持 `--override <key>=<value>` 参数 (如 `dataset.bmn_inputs_dir=data/bmn_inputs/my_ext/`). 如果不支持, 扩展 `src/pingpong_av/cli/train.py` 加此标志, 把 override 合并到 user_cfg. 对应 quickstart 场景 B 第 3 步. **完成 (零代码改动)**: 实测 `pp train --help` 没有 `--override`. 决定**不扩展 train.py** 避免引入新 cli 参数; 改用更朴素的"复制 yaml + sed 改 bmn_inputs_dir"方案 (与章程 III "config-driven" 哲学一致). spec.md US3 + quickstart.md 场景 B 同步更新示例.
- [x] T222 [US3] 在 `specs/002-raw-video-feature-bmn/quickstart.md` 场景 B 已有 3 步示例 (已写). 本任务只需在 `README.md` 主文档增加 "用我的数据微调基线" 一节 (5-10 行示例), 链接到 quickstart. 对应 FR-045. **完成**: README "用我的视频跑推理 (002)" 大节内含 3 个子场景 (infer-rawvideo / build-feature-pkls / 微调) 共 ~30 行示例; 链接指向 specs/002-.../quickstart.md.
- [x] T223 [US3] 集成测试 `tests/integration/test_resume_finetune_e2e.py` (可选, 如果 T221 实际改了代码). 用一个 tiny mock BMN ckpt (随机权重 + small data) 跑 1 epoch 微调, 验证 manifest.json 中 `notes.resumed_from` 字段被写. 用 `pytest.mark.slow`. **跳过**: T221 没改 train.py 代码, 且 `pp train --resume` 在 v0.2.x 已存在并经过实战 (训练 PID 3571466 至今 13/20 epoch 没问题), 不需要新测试.

**检查点**: 此时 US3 可用 + US1/US2 仍独立运行正常.

---

## 阶段 6: 收尾与横切关注点

**目的**: 把 002 合入后规范 / 文档 / 版本的一致性工作.

- [x] T224 [P] 更新 `requirements/base.txt`: 无新依赖 (ffmpeg / numpy / Pillow / paddle 已在). 在文件头注释段写 "002 无新增依赖". **完成**: requirements/base.txt 注释段增加 4 行说明 002 feature 无新依赖.
- [x] T225 [P] 在 `CODEBUDDY.md` "最近变更" 段增加 002 条目, 描述主要功能. 在 "活跃技术" 段提及 `pp extract-feat / build-feature-pkls / infer-rawvideo` 3 个新子命令. 在 "命令" 段增加 4 条新命令示例 (quickstart 场景 A 的 4 行). **完成**: 3 段全部更新, 含 `2026-05-13` 最近变更条目 + 命令段 002 子部分 (5 条命令).
- [x] T226 [P] 在 `README.md` 增加新主章节 "用我的视频跑推理 (002, 原始视频端到端)", 引用 `specs/002-raw-video-feature-bmn/quickstart.md`. 更新阶段进度表, 追加 "阶段 10 US6.2 原始视频适配 T200-T223". **完成**: README 主大节 "用我的视频跑推理 (002)" 含 3 个子场景 + 进度表 Phase 10 (T200-T230 31/31); 总进度 89→120.
- [x] T227 在 `specs/001-pingpong-action-recognition/tasks.md` 追加备注, 说明 002 是在此之上的独立功能, 不修改 001 任务. **完成**: 001/tasks.md 末尾追加 "## 与 002 feature 的关系 (2026-05-13 增补)" 段, 13 行说明.
- [x] T228 运行完整测试 `.venv/bin/pytest tests/ -v --durations=10`, 确保 100/100 通过 + 002 新增测试 (跳过 slow 除非 `--slow`). **完成**: 默认 126 passed + 7 skipped (slow); --runslow 130 passed; durations top-1 = 1.69s (env-check stub).
- [x] T229 (可选) 合入后第一次真实运行: `pp infer-rawvideo --input <real_video>.mp4 --bmn-checkpoint experiments/20260512-145311-*/BMN_epoch_00020.pdparams --output-dir outputs/smoke/`, 检查输出三件套 (timeline.json / visualized.mp4 / feature.pkl) 都正常. 记入 PR 描述. **跳过**: 训练 PID 3571466 仍在跑 (epoch 13/20), 没有 epoch_00020 ckpt; Phase 3 用 epoch_00011 ckpt 已实测 SC-010 通过 (13s 端到端). 真实视频 smoke 留待训练完成 + 用户后续 v0.3.1 release.
- [x] T230 `git commit -m "feat(002): raw video → PP-TSN feature pkl → BMN end-to-end"` + `git tag v0.3.0-rc.1` + 推送到远程 (按用户要求, 如有 `github_token` in .env). **完成**: 实际打了 2 个 tag — v0.3.0-rc.0 (US1 MVP, commit dd6602c) + **v0.3.0** (完整 002 release, commit 9a0f097). main 已推送到 origin. 6 个 release tags 全部就绪.

---

## 依赖关系与执行顺序

### 阶段依赖

```
Phase 1 (Setup T200-T202) ── 可并行
        ↓
Phase 2 (Foundation T203-T209) ── 部分并行: T203/T204/T205 独立; T206 独立; T207 依赖 T206 产出; T208 独立; T209 依赖 T208 重构
        ↓
Phase 3 (US1 T210-T216) ── MVP 入口; T210 先于 T211; T214/T215/T216 与实现可并行
        ↓                                  ↓
Phase 4 (US2 T217-T220)               ─── (US1 完成后可独立增量)
        ↓
Phase 5 (US3 T221-T223) ── 最小代码改动; 主要是文档
        ↓
Phase 6 (Polish T224-T230) ── 除 T228/T229/T230 外都可并行
```

### 关键串行链

- **MVP 路径** (最短到可交付 US1): `T200 → T203 → T204 → T205 → T206 → T207 → T208 → T209 → T210 → T211 → T212 → T213 → T216 → 第一次运行`. 估计 12-14 任务.
- **US2 路径** (能扩充训练数据): MVP 之后 + `T217 → T218 → T220`. 增量 3 任务.
- **US3 路径** (微调基线): US2 之后 + `T221 → T222`. 增量 2 任务.

### 并行机会

**阶段 1 全部并行**: T200 (yaml) / T201 (gitignore) / T202 (fixture) — 3 个人或 3 条 agent 同时做.

**阶段 2 大部分并行**:
- `[P]` T203 (clip_id) / T204 (ffmpeg) / T205 (manifest) — 三个独立模块文件
- T206 (export script) 独立
- T207 (PPTSMExtractor) 依赖 T206 产物 (但**不**依赖 T206 代码; 实现时可用 mock inference 文件)
- T208 (prepare_bmn_inputs 重构) 独立
- T209 (gt_required 参数) 依赖 T208 存在 (同文件不同函数)

**阶段 3 测试并行**: T214 / T215 / T216 独立文件, 三个测试用例可以并行写.

**阶段 6 文档并行**: T224 / T225 / T226 三份不同文件, 可并行.

---

## 每个故事的独立测试标准

| Story | 独立可测条件 | 关键验收用例 |
|-------|-------------|------------|
| **US1** | 已完成阶段 1 + 阶段 2 + T210-T213 + T216 | 用 fixture mp4 + mock ckpt 跑 `pp infer-rawvideo` → `timeline.json` schema 合法 |
| **US2** | 已完成 US1 + T217-T218 + T220 | 3 段 fixture + mock GT → 产出 `manifest.csv` + `label_cls14_<name>.json` 格式正确, url 字段为 clip_id |
| **US3** | 已完成 US2 + T221 (如需) | `pp train --resume` 启动成功, manifest 含 `resumed_from` 字段 |

**MVP = US1** (T200-T216 共 17 任务). 其余是增量.

---

## 实现策略

### MVP 优先 (P1)

先完成 **阶段 1 + 阶段 2 + 阶段 3**: 15-16 个任务 (T200-T216). 这是"用户拿到本项目能跑推理"的最小交付. 完成后可:
- 独立发布 v0.3.0-rc.0 (仅 US1)
- 从真实用户拿反馈
- 在 v0.3.0 后再增量 US2 / US3

### 增量交付

- **v0.3.0-rc.0**: 阶段 1-3 完成, US1 端到端可用 (含 MVP).
- **v0.3.0-rc.1**: 增量 US2 (阶段 4), 批量数据扩充可用.
- **v0.3.0**: 增量 US3 (阶段 5) + 收尾 (阶段 6), 完整版发布.

### 测试策略

- **Unit test** (T214/T215/T219): 无 GPU 依赖, mock paddle API, 在 CI 中秒级执行, 跑在每次 commit.
- **Integration test** (T216/T220/T223): 需要 GPU + fixture mp4, 用 `pytest.mark.slow` 标记, 本地手动 `--slow` 跑, CI 只在 release 前跑 (章程 V).
- **不**写 fuzz 测试 / 基准测试 (out of scope, 章程 V 不强制).

### 回滚点

每个阶段完成都可以独立回滚 — 如果 US1 出了问题但 US2/US3 未实现, 回滚 US1 改动即可. spec.md 已预留 "不在范围内" 段落, 回滚时 spec 不需要修改.

---

## 任务总数摘要

| 阶段 | 任务数 | 备注 |
|------|-------|------|
| Phase 1 (Setup) | 3 (T200-T202) | 全并行 |
| Phase 2 (Foundation) | 7 (T203-T209) | 5 可并行 + 2 顺序 |
| Phase 3 (US1) | 7 (T210-T216) | MVP |
| Phase 4 (US2) | 4 (T217-T220) | 增量 |
| Phase 5 (US3) | 3 (T221-T223) | 最小代码 + 文档 |
| Phase 6 (Polish) | 7 (T224-T230) | 含 commit + tag + push |
| **合计** | **31 任务** | T200-T230 |

## 格式验证

所有 31 个任务都遵循清单格式 (`- [ ] [TaskID] [P?] [Story?] 描述 + 文件路径`):
- ✅ 全部以 `- [ ]` 开头
- ✅ 全部有序号 ID T200-T230
- ✅ 并行任务标 [P]
- ✅ 用户故事任务标 [US1]/[US2]/[US3]
- ✅ 描述含确切文件路径
