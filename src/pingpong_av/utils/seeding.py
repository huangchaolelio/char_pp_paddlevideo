"""统一随机种子设置 (FR-018, 章程 II).

调用 :func:`set_seed` 可一次性把种子注入到所有相关随机源:
- Python ``random`` 标准库
- ``numpy``
- ``paddle`` (含 GPU)
- 进程级环境变量 ``PYTHONHASHSEED`` (新进程才会生效, 仅作记录)

paddle 不可用时**优雅降级**: 仍设置 random 与 numpy, 不抛错; 这样 ``pp env-check`` 在
paddle 未装的环境下也能调用 set_seed 而不致命 (env-check 本身要在 import paddle 失败的
情况下仍能给出诊断).

不在本模块的范围:
- cudnn 确定性策略 (与 paddle 版本相关; 训练入口在调用 paddle 时按需设置).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedingResult:
    seed: int
    sources_set: list[str]      # 实际生效的随机源列表
    sources_skipped: list[str]  # 因依赖缺失而跳过的随机源


def set_seed(seed: int) -> SeedingResult:
    """同步设置 random / numpy / paddle 的随机种子.

    返回 :class:`SeedingResult` 便于 manifest / 日志记录哪些随机源真正被设置.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须是 int, 实际为 {type(seed).__name__}: {seed!r}")
    if seed < 0:
        raise ValueError(f"seed 必须是非负整数, 实际为 {seed}")

    set_: list[str] = []
    skipped: list[str] = []

    # Python 标准库 random — 永远可用
    random.seed(seed)
    set_.append("random")

    # PYTHONHASHSEED — 仅对新启动的子进程生效; 这里设置以保持一致性记录
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    set_.append("PYTHONHASHSEED")

    # numpy
    try:
        import numpy as np

        np.random.seed(seed)
        set_.append("numpy")
    except Exception:  # ImportError 或 numpy 内部错误
        skipped.append("numpy")

    # paddle (CPU + 所有可见 GPU)
    try:
        import paddle

        paddle.seed(seed)
        # paddle.seed 在新版本中已会同步设置 GPU; 老版本需要 paddle.framework.seed(...)
        set_.append("paddle")
    except Exception:
        skipped.append("paddle")

    return SeedingResult(seed=seed, sources_set=set_, sources_skipped=skipped)
