# PaddleVideo 上游兼容性补丁 (`third_party/patches/`)

本目录用于维护本项目对**上游 PaddleVideo submodule** 的最小化补丁集. 严格遵守章程
原则 VI (上游兼容与最小侵入) 与原则 VIII (隔离的 Python 3.11 环境).

---

## 背景

- 上游仓库: `https://github.com/PaddlePaddle/PaddleVideo.git`
- 接入分支: `release/2.2.0` (固定 commit, 见仓库根 `.gitmodules` 与 `README.md`)
- 接入方式: Git submodule, 路径 `third_party/PaddleVideo/`
- 上游若与本项目锁定的 Python 3.11 或本仓库的工程约定不兼容, **不得**直接修改 submodule
  入库版本; 必须通过本目录下的 patch 文件修复.

---

## 命名规范

每个补丁是一个独立文件, 文件名格式为:

```
NN-<short-symptom>.patch
```

- `NN` 是两位十进制数字 (`01`, `02`, ..., `99`), 决定**应用顺序**.
- `<short-symptom>` 用 kebab-case 描述补丁要修的核心问题 (例如 `collections-mapping-py311`).
- 扩展名 `.patch` (而非 `.diff`), 内容为 `git format-patch` 或 unified diff 格式.

示例:

```
01-collections-mapping-py311.patch
02-numpy-bool-alias-deprecation.patch
03-decord-import-fallback-to-pyav.patch
```

---

## 每个补丁的强制元信息

补丁文件**第一行注释块**必须包含以下字段, 缺一不可:

```
# Patch:        <补丁的简短标题>
# Reason:       <为什么需要打: 触发的错误信息 / 上游 issue 号 / 内部 ticket>
# Upstream PR:  <若上游已合并 / 已开 PR 的链接; 没有则填 "none">
# Tested on:    <验证补丁有效的具体上游 commit SHA, 通常等于本项目固定的 submodule SHA>
# Removable when: <什么条件下可以删除该补丁: 例如"上游 PR #1234 合并后" / "升级到 v2.X 后">
```

例:

```diff
# Patch:        collections.Mapping → collections.abc.Mapping for Python 3.11
# Reason:       Python 3.10+ removed collections.Mapping; PaddleVideo data loaders still import it.
#               Triggers `ImportError: cannot import name 'Mapping' from 'collections'`.
# Upstream PR:  https://github.com/PaddlePaddle/PaddleVideo/pull/XXXX  (assumed merged in develop)
# Tested on:    da9a8ce8f0beba020727c7e7b6c50308d32df76f  (release/2.2.0 tip)
# Removable when: 上游 cherry-pick 修复到 release/2.2.x 或本项目升级到含修复的更高 release.
--- a/paddlevideo/loader/builder.py
+++ b/paddlevideo/loader/builder.py
@@ -10,7 +10,7 @@
-from collections import Mapping
+from collections.abc import Mapping
```

---

## 应用方式

`scripts/apply_upstream_patches.sh` (在 T011 实现) 会按文件名升序对 `third_party/PaddleVideo/`
的工作区**幂等**地应用本目录下所有 `.patch` 文件:

1. 跳过已应用的补丁 (通过 `git apply --check` 探测);
2. 失败时立即退出非零, 由 bootstrap 流程感知;
3. 应用补丁产生的修改**不入库** (submodule 工作区是 detached HEAD, 修改不影响父仓库
   追踪的 submodule SHA), 这正是章程 VI 所要求的"不污染上游"行为.

---

## 当何时新增 / 删除补丁

**新增** 补丁的触发条件 (严格):

- 在隔离 `.venv` 中导入 `paddlevideo` 或运行其示例时报错, 且错误根因可定位到 Python 3.11
  与上游代码的兼容性差异.
- 上游某行为与本项目 quickstart / contracts 定义的接口冲突, 必须用最小化 diff 修复.
- 上述两种以外的"个人风格" / "代码整洁度"修改一律拒绝, 这是章程 VI 的硬约束.

**删除** 补丁的触发条件:

- 上游已合并对应修复, 升级 submodule 到包含修复的 commit 后, 该补丁的 `Removable when`
  条件被满足 → 删除补丁文件并在 commit message 中引用上游 PR.
- 删除后必须重新跑一次 quickstart 全流程验证, 确保确实不再依赖该补丁.

---

## 关联制品

- 章程: `.specify/memory/constitution.md` (原则 VI, 原则 VIII)
- 计划: `specs/001-pingpong-action-recognition/plan.md`
- 研究: `specs/001-pingpong-action-recognition/research.md` (R1: submodule 锁定; R3: 3.11 适配策略)
- bootstrap: `scripts/bootstrap.sh` (T010), 调用 `apply_upstream_patches.sh` (T011)

---

## 当前补丁清单

| 文件 | 标题 | 触发场景 | 删除条件 |
|------|------|---------|---------|
| [`01-paddle-fluid-removal-py311.patch`](01-paddle-fluid-removal-py311.patch) | paddle.fluid → paddle.base / paddle.nn.functional | Paddle 2.6 移除 paddle.fluid 子包, 上游 release/2.2.0 多处 import 旧位置 | 升级到含修复的更高 PaddleVideo release; 或 fork 上游做合并 |
| [`02-decord-lazy-import-py311.patch`](02-decord-lazy-import-py311.patch) | decord 延迟 import | decord 0.4.x 无 Python 3.11 wheel; 上游 3 个 pipeline 文件在模块顶层 eager import | 上游切换为 lazy import 或升级到有 3.11 wheel 的 decord 版本 |
| [`03-inspect-getargspec-py311.patch`](03-inspect-getargspec-py311.patch) | inspect.getargspec → getfullargspec | Python 3.11 删除 inspect.getargspec; 上游 BMN 训练经过 build_optimizer (Adam) 时触发 | 上游升级到 3.11 兼容版本 |
| [`04-record-tensor-scalar-py311.patch`](04-record-tensor-scalar-py311.patch) | record.AverageMeter Tensor[0] → .item() | paddle 2.6+ 0-d Tensor.numpy() 返回 0-d ndarray, [0] 索引抛 IndexError; 每个训练 step 触发 | 上游升级到任意安全索引方式 |
