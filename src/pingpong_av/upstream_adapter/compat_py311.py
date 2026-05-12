"""上游 PaddleVideo 在 Python 3.11 下的运行时兼容层 (章程 VIII).

本模块**仅**承载那些通过修改 sys.modules / 提前打 monkey-patch 即可解决、不需要
触动 submodule 文件的兼容修复. 一切需要修改文件内容才能修复的问题都应**进入**
``third_party/patches/*.patch`` 走 :file:`scripts/apply_upstream_patches.sh` 流程.

调用约定:
    业务入口 (``cli/train.py`` / ``cli/eval.py`` 等) 在调用 paddlevideo 之前, 先调用
    :func:`apply_runtime_patches`. 该函数**幂等**, 多次调用无副作用.

当前内容:
    占位骨架. 在执行 ``T033`` (上游 smoke 测试) 阶段如果暴露出 3.11 不兼容的运行时
    问题, 在此函数中按需添加最小化补丁. 每个补丁应附 ``# REASON / # REMOVABLE WHEN`` 注释,
    与 ``third_party/patches/README.md`` 的元信息约定一致.
"""

from __future__ import annotations

_PATCHES_APPLIED = False


def apply_runtime_patches() -> None:
    """对 import paddlevideo 之前需要的运行时兼容进行 monkey-patch.

    幂等: 第一次调用执行实际逻辑, 之后调用直接返回.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    # ---- 占位: 当前没有运行时 patch ----
    # 在此添加形如:
    #
    #   # REASON: PaddleVideo 在某模块中 `from collections import Mapping`,
    #   #         3.10+ 已移除该 alias, 触发 ImportError.
    #   # REMOVABLE WHEN: 上游 issue #XXXX 修复后或本项目升级到含修复的 release.
    #   import collections
    #   import collections.abc
    #   if not hasattr(collections, "Mapping"):
    #       collections.Mapping = collections.abc.Mapping
    #
    # 每条 patch 必须 (1) 幂等 (二次调用不重复), (2) 范围最小, (3) 带 REASON / REMOVABLE WHEN.

    _PATCHES_APPLIED = True


def reset_patches_for_testing() -> None:
    """仅供单元测试使用: 重置已应用标记, 让 :func:`apply_runtime_patches` 可再次执行.

    生产代码**不应**调用此函数.
    """
    global _PATCHES_APPLIED
    _PATCHES_APPLIED = False
