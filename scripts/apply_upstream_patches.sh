#!/usr/bin/env bash
# ============================================================
# apply_upstream_patches.sh — 把 third_party/patches/*.patch 应用到 PaddleVideo submodule
# ============================================================
# 章程对齐:
#   - 章程 VI:  patch 仅在 submodule 工作区生效, 不修改入库的 submodule 指针;
#               不直接改上游源码, 升级 submodule 时丢弃过时 patch 即可.
#   - 章程 VIII: 一切兼容性修复都通过 patch + 适配层, 不通过降低 Python 版本绕过.
#
# 行为:
#   1. 按文件名顺序 (NN-*.patch) 升序遍历;
#   2. 对每个 patch 先 `git apply --check` 探测; 已应用则跳过 (幂等);
#   3. 失败立即退出, 退出码非零, 由 bootstrap 流程感知;
#   4. 仅作用于 third_party/PaddleVideo/ 工作区.
# ============================================================

set -euo pipefail

# 解析仓库根 (不依赖被调用时的 PWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PATCHES_DIR="$REPO_ROOT/third_party/patches"
SUBMODULE_DIR="$REPO_ROOT/third_party/PaddleVideo"

if [[ ! -d "$SUBMODULE_DIR/.git" && ! -f "$SUBMODULE_DIR/.git" ]]; then
    echo "ERROR: PaddleVideo submodule 未初始化于 $SUBMODULE_DIR" >&2
    echo "       请先运行: git submodule update --init --recursive" >&2
    exit 2
fi

if [[ ! -d "$PATCHES_DIR" ]]; then
    echo "ERROR: 找不到 patches 目录: $PATCHES_DIR" >&2
    exit 2
fi

cd "$SUBMODULE_DIR"

# 收集所有 .patch 文件并按文件名排序
mapfile -t PATCH_FILES < <(find "$PATCHES_DIR" -maxdepth 1 -type f -name '*.patch' | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
    echo "INFO: $PATCHES_DIR 下没有 .patch 文件, 跳过 (这是预期: 当前 patch 清单为空)."
    exit 0
fi

APPLIED=0
SKIPPED=0
FAILED=0

for patch_file in "${PATCH_FILES[@]}"; do
    name="$(basename "$patch_file")"

    # 探测: 是否已经应用过
    if git apply --reverse --check "$patch_file" >/dev/null 2>&1; then
        echo "SKIP   $name  (已应用)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # 探测: 是否能干净应用
    if ! git apply --check "$patch_file" >/dev/null 2>&1; then
        echo "FAIL   $name  (无法干净应用; 上游可能已变化, 需要更新或丢弃此 patch)" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    # 实际应用
    if git apply "$patch_file"; then
        echo "APPLY  $name"
        APPLIED=$((APPLIED + 1))
    else
        echo "FAIL   $name  (apply 失败)" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "Summary: applied=$APPLIED  skipped=$SKIPPED  failed=$FAILED"

if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: 有 patch 应用失败. 请查看 third_party/patches/README.md 修复或更新对应 patch." >&2
    exit 1
fi

exit 0
