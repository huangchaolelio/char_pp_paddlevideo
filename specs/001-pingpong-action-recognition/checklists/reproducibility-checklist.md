# 可复现性回归 checklist (T074, 章程 II)

**目的**: 章程 II "可复现实验" 要求同 `seed` + 同 `config_hash` + 同 `commit` + 同 `dataset_split_version` 的两次运行产出**字节级或数值级一致**的指标 . 本 checklist 在以下条件就绪时由维护者**逐项勾选**:

- AI Studio 竞赛 #127 的乒乓球数据已手动下载到 `data/raw/pingpong_public/` 且 `.ready` 哨兵存在
- decord 3.11 兼容性已解决 (T056 patch 02 应用)
- GPU 可用且 `pp env-check --strict` 全绿

代码层面的可复现性回归已在 `tests/unit/test_reproducibility.py` 自动化覆盖 (12 个测试 ✓).
本 checklist 关注端到端 (data-prepare + train + eval) 的真实回归.

---

## 准备工作

- [ ] **A0** — git 工作区干净 (`git status` 无 modified / untracked)
- [ ] **A1** — submodule 指针 == `da9a8ce8f0beba020727c7e7b6c50308d32df76f` (`git submodule status`)
- [ ] **A2** — `.venv/bin/pp env-check --strict` 输出 exit 0 (Python 3.11.x + paddle 2.6.2 + paddlevideo from submodule)
- [ ] **A3** — `data/raw/pingpong_public/.ready` 存在且数据目录非空
- [ ] **A4** — 上次回归结果已 archive (避免污染本次比较)

## 第一次运行 (Run-A)

- [ ] **B1** — 记录起始时刻: `date -Iseconds > /tmp/run_a_start`
- [ ] **B2** — `pp data-prepare --config configs/datasets/pingpong_public.yaml --force` exit 0
- [ ] **B3** — 记录 `data/splits/{train,val,test}.txt` 各自的 sha256:
      ```
      sha256sum data/splits/*.txt > /tmp/run_a_splits.sha
      ```
- [ ] **B4** — `pp train --config configs/models/pp_tsm_pingpong.yaml --seed 2026` exit 0
- [ ] **B5** — 记录 `experiments/<run_id_A>/manifest.json` 路径与四元组 (config_hash / commit / seed / dataset_split_version)
- [ ] **B6** — `pp eval --checkpoint experiments/<run_id_A>/checkpoints/best.pdparams --split val` exit 0
- [ ] **B7** — 备份 `experiments/<run_id_A>/metrics.json` 到 `/tmp/run_a_metrics.json`

## 第二次运行 (Run-B, 完全相同参数)

- [ ] **C1** — git 工作区仍然干净 (与 A0 一致)
- [ ] **C2** — `pp data-prepare --config configs/datasets/pingpong_public.yaml --force` exit 0
- [ ] **C3** — `sha256sum data/splits/*.txt > /tmp/run_b_splits.sha`
- [ ] **C4** — `pp train --config configs/models/pp_tsm_pingpong.yaml --seed 2026` exit 0
- [ ] **C5** — `pp eval --checkpoint experiments/<run_id_B>/checkpoints/best.pdparams --split val` exit 0

## 比对断言 (章程 II 验收)

- [ ] **D1** — splits 文件 sha256 完全一致 (`diff /tmp/run_a_splits.sha /tmp/run_b_splits.sha` 应无输出)
- [ ] **D2** — Run-A 与 Run-B 的 manifest.json `config_hash` 字段相同
- [ ] **D3** — Run-A 与 Run-B 的 manifest.json `commit` 字段相同 (没人 push 新 commit)
- [ ] **D4** — Run-A 与 Run-B 的 manifest.json `seed` == 2026
- [ ] **D5** — Run-A 与 Run-B 的 manifest.json `dataset_split_version` 相同
- [ ] **D6** — Run-A 与 Run-B 的 `metrics.json` 中 `top1`, `top5`, `macro_avg.f1` 差异在 ±0.5 个百分点内
        (PaddleVideo 的 GPU 算子可能引入极小非确定性, 但应远小于业务波动)

## 失败处理

任何一条 D# 不达标:
1. 检查 git status 是否有未提交修改 (典型: 临时改了 yaml 又忘了 revert)
2. 检查 `manifest.notes` 是否含 `allow_dirty: true` — 若有, 本次结果**不得**作为正式指标 (章程 II)
3. 检查 CUDA/cuDNN 版本是否被系统更新 (manifest.cuda_version 字段会显式记录)
4. 在 PR 描述中开 issue, 引用本 checklist 的具体不达标项

## 当前状态

> **2026-05-12**: 本 checklist 处于"待激活"状态. 阻塞项:
>   - PaddleVideo 上游不提供公开数据集 URL (R2 修正版), 需要用户手动 AI Studio 注册
>   - decord 0.4.x 无 Python 3.11 wheel, T056 待 patch
>
> 代码层面的可复现性已被 `tests/unit/test_reproducibility.py` 全自动化覆盖 (12/12 通过).
> 端到端的真实回归需要等上述两个阻塞项解决.
