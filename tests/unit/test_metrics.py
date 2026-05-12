"""evaluation/metrics.py 单元测试 (章程 V).

覆盖范围:
- ``compute_topk``: 完美 / 全错 / 中间 / k > C / 空输入
- ``compute_per_class``: precision/recall/f1/support 数值正确性, 空 support 类别
- ``compute_macro_avg``: 算术平均不按 support 加权
- ``compute_imbalance_warning``: 阈值边界
- 输入校验: 形状 / 范围越界
"""

from __future__ import annotations

import numpy as np
import pytest

from pingpong_av.evaluation.metrics import (
    compute_imbalance_warning,
    compute_macro_avg,
    compute_per_class,
    compute_topk,
)


# --------------------------------------------------------------------------------------
# compute_topk
# --------------------------------------------------------------------------------------


def test_topk_perfect_predictions() -> None:
    """argmax 完全等于 labels → top1 = 1.0."""
    logits = np.array([[5.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0],
                       [0.0, 0.0, 5.0]])
    labels = np.array([0, 1, 2])
    assert compute_topk(logits, labels, k=1) == pytest.approx(1.0)
    assert compute_topk(logits, labels, k=2) == pytest.approx(1.0)


def test_topk_all_wrong_top1() -> None:
    """argmax 全错 → top1 = 0.0; top2 命中其他类 → 1.0."""
    logits = np.array([[5.0, 4.0, 0.0],
                       [4.0, 5.0, 0.0]])
    labels = np.array([1, 0])  # 真值是 secondary 选项
    assert compute_topk(logits, labels, k=1) == pytest.approx(0.0)
    assert compute_topk(logits, labels, k=2) == pytest.approx(1.0)


def test_topk_partial_correct() -> None:
    """3 个样本中 2 个 top-1 正确 → top1 = 2/3."""
    logits = np.array([[5.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0],
                       [5.0, 0.0, 0.0]])  # 第三个错
    labels = np.array([0, 1, 2])
    assert compute_topk(logits, labels, k=1) == pytest.approx(2 / 3)


def test_topk_k_greater_than_classes_clips_to_one() -> None:
    """k 超过类别数时, 等价于全部命中 → 1.0."""
    logits = np.random.randn(10, 3)
    labels = np.random.randint(0, 3, size=(10,))
    assert compute_topk(logits, labels, k=100) == pytest.approx(1.0)


def test_topk_empty_input_returns_zero() -> None:
    logits = np.zeros((0, 3))
    labels = np.zeros((0,), dtype=np.int64)
    assert compute_topk(logits, labels, k=1) == 0.0


def test_topk_invalid_k_raises() -> None:
    logits = np.zeros((3, 3))
    labels = np.zeros((3,), dtype=np.int64)
    with pytest.raises(ValueError, match=">"):
        compute_topk(logits, labels, k=0)


# --------------------------------------------------------------------------------------
# compute_per_class
# --------------------------------------------------------------------------------------


def test_per_class_perfect_predictions() -> None:
    """完美预测 → 每类 precision=recall=f1=1.0."""
    logits = np.eye(3) * 10
    labels = np.array([0, 1, 2])
    out = compute_per_class(logits, labels, ["a", "b", "c"])
    for cls in ("a", "b", "c"):
        assert out[cls]["precision"] == pytest.approx(1.0)
        assert out[cls]["recall"] == pytest.approx(1.0)
        assert out[cls]["f1"] == pytest.approx(1.0)
        assert out[cls]["support"] == 1


def test_per_class_includes_zero_support_classes() -> None:
    """没有 support 的类也应出现在结果中, support=0, 指标=0."""
    logits = np.array([[5.0, 0.0, 0.0]] * 3)  # 全部预测为 a
    labels = np.array([0, 0, 0])
    out = compute_per_class(logits, labels, ["a", "b", "c"])
    assert "b" in out and out["b"]["support"] == 0
    assert "c" in out and out["c"]["support"] == 0
    assert out["a"]["support"] == 3


def test_per_class_basic_confusion_matrix() -> None:
    """构造已知混淆矩阵, 校验 precision/recall 数值."""
    # 真值: 0,0,1,1; 预测: 0,1,1,1 → 类0: TP=1 FN=1; 类1: TP=2 FP=1
    # precision_0 = 1/1 = 1.0  recall_0 = 1/2 = 0.5
    # precision_1 = 2/3 ≈ 0.667  recall_1 = 2/2 = 1.0
    logits = np.array([
        [5.0, 0.0],   # pred 0
        [0.0, 5.0],   # pred 1
        [0.0, 5.0],   # pred 1
        [0.0, 5.0],   # pred 1
    ])
    labels = np.array([0, 0, 1, 1])
    out = compute_per_class(logits, labels, ["c0", "c1"])
    assert out["c0"]["precision"] == pytest.approx(1.0)
    assert out["c0"]["recall"] == pytest.approx(0.5)
    assert out["c1"]["precision"] == pytest.approx(2 / 3)
    assert out["c1"]["recall"] == pytest.approx(1.0)
    assert out["c0"]["support"] == 2
    assert out["c1"]["support"] == 2


def test_per_class_class_names_length_must_match() -> None:
    logits = np.zeros((3, 3))
    labels = np.zeros((3,), dtype=np.int64)
    with pytest.raises(ValueError, match="不一致"):
        compute_per_class(logits, labels, ["a", "b"])  # 只给 2 个名字


def test_per_class_empty_input() -> None:
    out = compute_per_class(np.zeros((0, 3)), np.zeros((0,), dtype=np.int64), ["a", "b", "c"])
    assert all(v["support"] == 0 for v in out.values())


# --------------------------------------------------------------------------------------
# compute_macro_avg
# --------------------------------------------------------------------------------------


def test_macro_avg_arithmetic_mean() -> None:
    per_class = {
        "a": {"precision": 1.0, "recall": 0.5, "f1": 0.6, "support": 2},
        "b": {"precision": 0.0, "recall": 1.0, "f1": 0.4, "support": 100},
    }
    out = compute_macro_avg(per_class)
    # macro 不按 support 加权
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.75)
    assert out["f1"] == pytest.approx(0.5)


def test_macro_avg_empty() -> None:
    assert compute_macro_avg({}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# --------------------------------------------------------------------------------------
# compute_imbalance_warning
# --------------------------------------------------------------------------------------


def test_imbalance_warning_below_threshold() -> None:
    per_class = {
        "a": {"precision": 0, "recall": 0, "f1": 0, "support": 100},
        "b": {"precision": 0, "recall": 0, "f1": 0, "support": 50},
    }
    out = compute_imbalance_warning(per_class, ratio_threshold=5.0)
    assert out["imbalance_warning"] is False
    assert out["max_min_ratio"] == pytest.approx(2.0)


def test_imbalance_warning_above_threshold() -> None:
    per_class = {
        "a": {"precision": 0, "recall": 0, "f1": 0, "support": 1000},
        "b": {"precision": 0, "recall": 0, "f1": 0, "support": 100},
        "c": {"precision": 0, "recall": 0, "f1": 0, "support": 10},
    }
    out = compute_imbalance_warning(per_class, ratio_threshold=5.0)
    assert out["imbalance_warning"] is True
    assert out["max_support"] == 1000
    assert out["min_support"] == 10
    assert out["max_min_ratio"] == pytest.approx(100.0)


def test_imbalance_warning_ignores_zero_support_classes() -> None:
    """零样本类别不应参与不平衡判定 (否则永远报警)."""
    per_class = {
        "a": {"precision": 0, "recall": 0, "f1": 0, "support": 100},
        "b": {"precision": 0, "recall": 0, "f1": 0, "support": 0},  # 零样本类
    }
    out = compute_imbalance_warning(per_class, ratio_threshold=5.0)
    assert out["imbalance_warning"] is False
    assert out["max_support"] == 100
    assert out["min_support"] == 100


# --------------------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------------------


def test_logits_must_be_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_topk(np.zeros(5), np.zeros(5, dtype=np.int64), k=1)


def test_labels_must_be_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        compute_topk(np.zeros((5, 3)), np.zeros((5, 1), dtype=np.int64), k=1)


def test_logits_labels_row_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="行数不一致"):
        compute_topk(np.zeros((5, 3)), np.zeros(4, dtype=np.int64), k=1)


def test_label_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="越界"):
        compute_topk(np.zeros((3, 3)), np.array([0, 1, 5]), k=1)
