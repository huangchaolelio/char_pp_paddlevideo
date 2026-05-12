"""evaluation.reporter 集成测试 (T072 章程 V).

验证:
- 类别平衡时, metrics.json 含 macro_avg 但**不**附 imbalance_warning
- 类别严重不平衡时, metrics.json 含 imbalance_warning=true + imbalance_detail
- 章程 V 必出字段 (top1, top5, per_class, macro_avg, n_samples, schema)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pingpong_av.evaluation.reporter import build_metrics_payload, write_metrics_json


def _make_logits(labels: np.ndarray, n_classes: int, *, accuracy: float = 0.9) -> np.ndarray:
    """构造给定准确率的 logits (确定性, 不用 RNG 避免脆弱)."""
    n = len(labels)
    n_correct = int(n * accuracy)
    preds = labels.copy()
    # 把后 n - n_correct 个改错 (循环到下一类)
    for i in range(n_correct, n):
        preds[i] = (preds[i] + 1) % n_classes
    logits = np.zeros((n, n_classes))
    logits[np.arange(n), preds] = 5.0
    return logits


# --------------------------------------------------------------------------------------
# 章程 V 必出字段
# --------------------------------------------------------------------------------------


def test_metrics_payload_contains_constitution_v_required_fields(tmp_path: Path) -> None:
    labels = np.array([0, 1, 2] * 10)  # 平衡: 每类 10
    logits = _make_logits(labels, n_classes=3, accuracy=0.9)
    payload = build_metrics_payload(
        logits=logits, labels=labels,
        class_names=["serve", "forehand", "backhand"],
        checkpoint=tmp_path / "fake.pdparams",
        split="test",
        topk=(1, 5),
    )
    # schema 标识
    assert payload["schema"] == "metrics-v1"
    # 章程 V 硬要求的指标全在
    for k in ("top1", "top5", "macro_avg", "per_class", "n_samples", "split"):
        assert k in payload, f"缺少 {k}: {sorted(payload.keys())}"
    # macro_avg 子字段
    for k in ("precision", "recall", "f1"):
        assert k in payload["macro_avg"]
    # per_class 含所有类
    for cls in ("serve", "forehand", "backhand"):
        assert cls in payload["per_class"]


# --------------------------------------------------------------------------------------
# 平衡场景: 不应附 imbalance_warning
# --------------------------------------------------------------------------------------


def test_balanced_classes_no_imbalance_warning(tmp_path: Path) -> None:
    """每类样本数相同 (max/min ratio=1.0), 不应附 imbalance_warning."""
    labels = np.array([0, 1, 2] * 20)  # 每类 20 个
    logits = _make_logits(labels, n_classes=3, accuracy=0.8)
    payload = build_metrics_payload(
        logits=logits, labels=labels,
        class_names=["a", "b", "c"],
        checkpoint="fake.pdparams", split="val",
    )
    # 平衡时不应有 imbalance_warning 字段
    assert "imbalance_warning" not in payload or payload["imbalance_warning"] is False


# --------------------------------------------------------------------------------------
# 不平衡场景: 必须附 imbalance_warning + imbalance_detail
# --------------------------------------------------------------------------------------


def test_imbalanced_classes_attaches_warning_and_detail(tmp_path: Path) -> None:
    """类别 0 有 100 个样本, 类别 1 只有 3 个 → ratio=33.3 > 5.0 阈值."""
    labels = np.concatenate([
        np.zeros(100, dtype=int),
        np.ones(3, dtype=int),
        np.full(50, 2, dtype=int),
    ])
    logits = _make_logits(labels, n_classes=3, accuracy=0.85)
    payload = build_metrics_payload(
        logits=logits, labels=labels,
        class_names=["majority", "minority", "medium"],
        checkpoint="fake.pdparams", split="test",
    )
    assert payload.get("imbalance_warning") is True, payload
    detail = payload["imbalance_detail"]
    assert detail["max_support"] == 100
    assert detail["min_support"] == 3
    assert detail["max_min_ratio"] == pytest.approx(100 / 3, rel=1e-3)


def test_imbalance_warning_persists_through_json_roundtrip(tmp_path: Path) -> None:
    """write_metrics_json → read_back: imbalance_warning 字段不丢."""
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(3, dtype=int)])
    logits = _make_logits(labels, n_classes=2, accuracy=0.9)
    payload = build_metrics_payload(
        logits=logits, labels=labels,
        class_names=["a", "b"],
        checkpoint="fake.pdparams", split="test",
    )
    out = tmp_path / "metrics.json"
    write_metrics_json(payload, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["imbalance_warning"] is True
    assert loaded["imbalance_detail"]["max_support"] == 100
    assert loaded["imbalance_detail"]["min_support"] == 3
    # macro_avg 在不平衡时尤其重要 (章程 V)
    assert "macro_avg" in loaded
    assert all(k in loaded["macro_avg"] for k in ("precision", "recall", "f1"))
