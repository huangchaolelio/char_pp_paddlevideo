"""上游 PaddleVideo 单点导入保障 (章程 VI).

业务代码**统一**通过本模块拿到可用的 ``paddlevideo`` 模块, 而**不直接** import
``third_party/PaddleVideo`` 的内部路径. 这样实现"逻辑上单点接入, 物理上隔离"的目标:

- **优先**使用已经通过 `pip install -e third_party/PaddleVideo` 安装到 .venv 的版本;
- **兜底**: 若 import 失败, 把 submodule 目录加入 ``sys.path`` 后重试一次.
- 两种途径都失败时给出**带修复指引**的明确错误, 提示运行 bootstrap.

不在本模块的范围:
- 训练 / 评估 / 推理的具体调用 (那是 ``upstream_adapter.trainer`` 的职责);
- Python 3.11 兼容补丁的应用 (那是 ``upstream_adapter.compat_py311`` 的职责).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from pingpong_av.utils.env import find_repo_root

__all__ = ["UpstreamImportError", "ensure_paddlevideo_on_path"]


class UpstreamImportError(ImportError):
    """无法以任何方式导入上游 PaddleVideo."""


def _submodule_dir(repo_root: Path) -> Path:
    return repo_root / "third_party" / "PaddleVideo"


def ensure_paddlevideo_on_path(repo_root: Path | None = None) -> ModuleType:
    """确保 ``paddlevideo`` 可被 import, 并返回该模块对象.

    优先级:
        1. 直接 ``import paddlevideo`` (假定已通过 ``pip install -e ...`` 安装).
        2. 若失败, 把 ``<repo>/third_party/PaddleVideo`` 加入 ``sys.path`` 头部后再次尝试.

    抛出:
        UpstreamImportError: 两种途径都失败.
    """
    # ---- 第 1 步: 直接 import ----
    try:
        return importlib.import_module("paddlevideo")
    except ImportError as first_err:
        first_repr = f"{type(first_err).__name__}: {first_err}"

    # ---- 第 2 步: 把 submodule 目录加入 sys.path 后兜底 ----
    root = repo_root or find_repo_root()
    sub = _submodule_dir(root)

    if not sub.is_dir():
        raise UpstreamImportError(
            "无法 import paddlevideo, 且 submodule 目录不存在: "
            f"{sub}. 请先运行 `git submodule update --init --recursive` 拉取上游, "
            "然后 `bash scripts/bootstrap.sh` 完成安装."
        )

    # PaddleVideo 仓库根布局: third_party/PaddleVideo/paddlevideo/
    if not (sub / "paddlevideo").is_dir():
        raise UpstreamImportError(
            "submodule 目录存在但不含 paddlevideo 包: "
            f"{sub}. submodule 可能被错误初始化, 请检查 .gitmodules 与 submodule 状态."
        )

    sub_str = str(sub.resolve())
    if sub_str not in sys.path:
        # 插入到首位以便优先于其他可能的同名包
        sys.path.insert(0, sub_str)

    # 清理可能被部分加载残留的 paddlevideo 模块, 强制重新解析
    for mod_name in list(sys.modules):
        if mod_name == "paddlevideo" or mod_name.startswith("paddlevideo."):
            sys.modules.pop(mod_name, None)

    try:
        return importlib.import_module("paddlevideo")
    except ImportError as second_err:
        second_repr = f"{type(second_err).__name__}: {second_err}"
        raise UpstreamImportError(
            "无法 import paddlevideo (即便加入 sys.path 之后也失败).\n"
            f"  - 直接 import 失败: {first_repr}\n"
            f"  - 经 sys.path 兜底失败: {second_repr}\n"
            "请确认: (1) submodule 已 init; (2) bootstrap.sh 已成功执行 "
            "(含 `pip install -e third_party/PaddleVideo`); "
            "(3) third_party/patches/ 下的 3.11 兼容补丁已应用."
        ) from second_err
