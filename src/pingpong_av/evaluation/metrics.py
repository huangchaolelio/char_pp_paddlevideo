"""评估指标 (章程 V: 评估纪律).

提供四类指标, 全部基于 numpy + scikit-learn 实现, **不**依赖 paddle:

- :func:`compute_topk` — Top-K 准确率 (FR-011, 章程 V 必含 top1/top5)
- :func:`compute_per_class` — 每类 precision/recall/f1/support (章程 V 必含)
- :func:`compute_macro_avg` — 宏平均 (章程 V: 类别不平衡时必含)
- :func:`compute_imbalance_warning` — 类别不平衡探测 (T072 衍生)

输入约定:
    logits: ``np.ndarray[N, C]`` — 模型输出 (logits 或 softmax 概率, 二者对 argmax/topk 等价)
    labels: ``np.ndarray[N]``    — 整数标签, 在 ``[0, C-1]``

不在本模块的范围:
- JSON / PNG 写盘 (那是 :mod:`pingpong_av.evaluation.reporter`)
- 与 paddle 张量的转换 (调用方在 :mod:`upstream_adapter.trainer` 中已转 numpy)
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "compute_topk",
    "compute_per_class",
    "compute_macro_avg",
    "compute_imbalance_warning",
]


# --------------------------------------------------------------------------------------
# Top-K 准确率
# --------------------------------------------------------------------------------------


def compute_topk(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """计算 Top-K 准确率: 真值标签是否在每个样本的 top-K 预测之中.

    返回 ``[0.0, 1.0]`` 的浮点数. 当 k > 类别数时, 等价于 1.0.
    """
    _validate_inputs(logits, labels)
    if k <= 0:
        raise ValueError(f"k 必须 > 0, 实际为 {k}")

    n_samples, n_classes = logits.shape
    if n_samples == 0:
        return 0.0

    eff_k = min(k, n_classes)
    # argpartition 比完全排序快: 取每行 top-k 的索引集合 (不要求 k 个之间有序)
    topk_idx = np.argpartition(-logits, kth=eff_k - 1, axis=1)[:, :eff_k]
    # broadcast 比较: labels[:, None] 形状 (N,1); topk_idx 形状 (N,k)
    hits = np.any(topk_idx == labels[:, None], axis=1)
    return float(hits.mean())


# --------------------------------------------------------------------------------------
# 每类指标 (precision/recall/f1/support)
# --------------------------------------------------------------------------------------


def compute_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """每类 precision / recall / f1 / support, 字段名与 data-model.md metrics-v1 对齐.

    返回:
        ``{class_name: {"precision": float, "recall": float, "f1": float, "support": int}}``,
        即便某类在 labels 中 support=0 也会出现 (此时 precision/recall/f1 都为 0).
    """
    _validate_inputs(logits, labels)
    if not class_names:
        raise ValueError("class_names 不能为空")
    n_classes = logits.shape[1]
    if len(class_names) != n_classes:
        raise ValueError(
            f"class_names 长度 ({len(class_names)}) 与 logits 第二维 ({n_classes}) 不一致"
        )

    if labels.size == 0:
        return {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for name in class_names}

    preds = logits.argmax(axis=1)

    # 用 sklearn 的 precision_recall_fscore_support; labels=range(n_classes) 强制返回所有类
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        out[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return out


# --------------------------------------------------------------------------------------
# 宏平均
# --------------------------------------------------------------------------------------


def compute_macro_avg(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    """宏平均 = 每类指标的算术平均 (不按 support 加权).

    章程 V 要求类别不平衡时**必须**附加宏平均, 因此本函数总是返回结果, 由调用方决定
    是否输出.

    在没有任何类别有 support>0 时返回全 0.
    """
    if not per_class:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    n = len(per_class)
    p_sum = sum(v["precision"] for v in per_class.values())
    r_sum = sum(v["recall"] for v in per_class.values())
    f_sum = sum(v["f1"] for v in per_class.values())
    return {
        "precision": float(p_sum / n),
        "recall": float(r_sum / n),
        "f1": float(f_sum / n),
    }


# --------------------------------------------------------------------------------------
# 类别不平衡探测 (T072 横切)
# --------------------------------------------------------------------------------------


def compute_imbalance_warning(
    per_class: dict[str, dict[str, float]],
    *,
    ratio_threshold: float = 5.0,
) -> dict[str, Any]:
    """探测 support 最大类与最小类的比例, 超过阈值即标记 imbalance_warning.

    章程 V: 类别不平衡时**必须**附加宏平均 — 本函数返回的标志由 reporter 决定写入与否.

    返回:
        ``{"imbalance_warning": bool, "max_support": int, "min_support": int,
            "max_min_ratio": float}``
    """
    supports = [v["support"] for v in per_class.values() if v["support"] > 0]
    if not supports:
        return {
            "imbalance_warning": False,
            "max_support": 0,
            "min_support": 0,
            "max_min_ratio": 0.0,
        }
    max_s = max(supports)
    min_s = min(supports)
    ratio = max_s / max(min_s, 1)
    return {
        "imbalance_warning": ratio > ratio_threshold,
        "max_support": int(max_s),
        "min_support": int(min_s),
        "max_min_ratio": float(ratio),
    }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _validate_inputs(logits: np.ndarray, labels: np.ndarray) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits 必须是 2D [N, C], 实际形状 {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels 必须是 1D [N], 实际形状 {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits 与 labels 行数不一致: {logits.shape[0]} vs {labels.shape[0]}"
        )
    if labels.size > 0:
        if labels.min() < 0 or labels.max() >= logits.shape[1]:
            raise ValueError(
                f"labels 范围越界: [{labels.min()}, {labels.max()}], "
                f"应在 [0, {logits.shape[1] - 1}]"
            )
