"""把指标计算结果写入实验目录, 按 data-model.md ``metrics-v1`` schema (章程 V).

产出 (放在 ``experiments/<run_id>/``):

- ``metrics.json``        — 完整结构化指标
- ``confusion_matrix.png`` — 混淆矩阵热图 (matplotlib 生成)

调用方:
    :mod:`pingpong_av.cli.eval` 在执行评估后调用 :func:`write_metrics_json`,
    后者协调 ``metrics.py`` 计算指标 + 写文件.

不在本模块的范围:
- 模型前向 (``upstream_adapter.trainer.run_upstream_eval``)
- 写入 manifest (那是 ``cli.eval`` 用 ``finalize`` 完成)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pingpong_av.evaluation.metrics import (
    compute_imbalance_warning,
    compute_macro_avg,
    compute_per_class,
    compute_topk,
)
from pingpong_av.utils.logging import get_logger

__all__ = ["write_metrics_json", "build_metrics_payload", "render_confusion_matrix"]

_log = get_logger(__name__)


def build_metrics_payload(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    checkpoint: str | Path,
    split: str,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    """根据 logits + labels 计算所有章程 V 必需指标, 返回 metrics-v1 schema dict."""

    n_samples = int(labels.shape[0])

    topk_metrics: dict[str, float] = {}
    for k in topk:
        topk_metrics[f"top{k}"] = compute_topk(logits, labels, k=k)

    per_class = compute_per_class(logits, labels, class_names)
    macro_avg = compute_macro_avg(per_class)
    imbalance = compute_imbalance_warning(per_class)

    payload: dict[str, Any] = {
        "schema": "metrics-v1",
        "checkpoint": str(checkpoint),
        "split": split,
        "n_samples": n_samples,
        # 章程 V 必出: top1 / top5
        **topk_metrics,
        "macro_avg": macro_avg,
        "per_class": per_class,
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }
    if imbalance["imbalance_warning"]:
        payload["imbalance_warning"] = True
        payload["imbalance_detail"] = imbalance

    return payload


def write_metrics_json(
    payload: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """把 metrics payload 写入 ``out_path`` (典型为 ``<run_dir>/metrics.json``).

    覆盖已有同名文件 (eval 重跑场景), 由 cli.eval 的 ``--rerun`` 闸门控制.
    """
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")
    _log.info(
        "metrics.json written",
        extra={
            "path": str(out),
            "n_samples": payload.get("n_samples"),
            "top1": payload.get("top1"),
            "top5": payload.get("top5"),
        },
    )
    return out


# --------------------------------------------------------------------------------------
# 混淆矩阵 PNG
# --------------------------------------------------------------------------------------


def render_confusion_matrix(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    normalize: bool = True,
    title: str | None = None,
) -> Path:
    """渲染混淆矩阵热图, 写入 ``out_path``.

    参数:
        normalize: True 时按真值归一化 (每行除以 row sum), 在类别不平衡时更易阅读.

    返回:
        实际写入的 PNG 路径.

    若 matplotlib 不可用 (极少, 因为已在 requirements/base.txt 中), 抛 ImportError.
    """
    if labels.size == 0:
        # 兜底: 写一个占位空白图, 避免后续读 PNG 失败
        _write_placeholder(out_path)
        return Path(out_path).resolve()

    preds = logits.argmax(axis=1)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
        if 0 <= true_idx < n and 0 <= pred_idx < n:
            cm[true_idx, pred_idx] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        display = cm / row_sums
    else:
        display = cm.astype(float)

    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")  # 无显示器环境
    import matplotlib.pyplot as plt

    fig_size = max(6.0, 0.6 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(display, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else display.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 轴刻度与名称
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=max(6, 10 - n // 4))
    ax.set_yticklabels(class_names, fontsize=max(6, 10 - n // 4))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix"))

    # 在格子上标数 (类别 ≤ 20 时); 否则太挤就省略
    if n <= 20:
        thresh = display.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = display[i, j]
                ax.text(
                    j, i,
                    f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=max(5, 9 - n // 3),
                )

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _log.info("confusion matrix rendered", extra={"path": str(out), "n_classes": n})
    return out


def _write_placeholder(out_path: str | Path) -> None:
    """labels 为空时写一个最小占位 PNG, 避免后续 IO 失败."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    # 一个 1x1 透明 PNG (8 字节签名 + IHDR + IDAT + IEND)
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
