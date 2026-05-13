"""action_classifier.py: BMN proposal → 14 类 label (原型匹配, training-free).

对应 003 feature (简化方案, 见 scripts/build_action_prototypes.py 头部注释).

用法:
    >>> clf = ActionPrototypeClassifier(prototype_path, classes_meta)
    >>> # 对每个 BMN proposal:
    >>> label_id, label_name, cls_score, topk = clf.classify_segment(
    ...     feature_arr=image_feature_arr,    # (N_samples, 2048) 视频的 PP-TSM 特征
    ...     start_sec=12.4, end_sec=16.0,
    ...     duration_sec=279.0,               # 视频总长 (用于秒→样本索引换算)
    ... )

设计 (research.md R10 + spec.md US1):
    - 原型用上游 AI Studio 数据 (~每帧 1 特征) 构建; 见 scripts/build_action_prototypes.py
    - 本仓库 pp extract-feat 用 seg_num=8 (每 8 帧 1 特征), 数学上是降采样后的同质特征
    - cosine similarity 对此降采样**不敏感** (均值后再归一化); 因此可直接复用原型
    - 实测准确率: 上游同尺度 LOO 53.4%; 跨尺度 (seg_num=8 → 1:1) 预计仍 > 35%
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ActionPrototypeClassifier", "ProtoClassifyResult"]


@dataclass
class ProtoClassifyResult:
    """一次分类的结果."""

    label_id: int                    # argmax 0..n_classes-1
    label_name: str                  # 中文 (e.g. "侧旋")
    cls_score: float                 # cosine similarity, [-1, 1]
    topk: list[tuple[int, str, float]]   # top-k 排序: [(id, name, score), ...]


class ActionPrototypeClassifier:
    """加载预计算的 14 类原型 + 提供 segment → label 分类."""

    def __init__(
        self,
        prototypes_path: Path | str,
        classes_meta: list[dict],
        *,
        topk: int = 3,
    ) -> None:
        """
        Args:
            prototypes_path: ``data/raw/pretrained/prototypes/action_prototypes_14.npy`` (14, 2048) float32.
            classes_meta: list of {id, name, display_name}; 取 display_name 作 label_name.
            topk: 返回 top-k 候选数.
        """
        prototypes_path = Path(prototypes_path)
        if not prototypes_path.is_file():
            raise FileNotFoundError(
                f"原型文件不存在: {prototypes_path}\n"
                f"  请先运行: .venv/bin/python scripts/build_action_prototypes.py"
            )
        self._prototypes = np.load(prototypes_path).astype(np.float32)   # (n_classes, 2048)
        # L2 normalize 一次, 后续 cosine 就是 dot
        norms = np.linalg.norm(self._prototypes, axis=1, keepdims=True) + 1e-8
        self._prototypes_normalized = self._prototypes / norms

        self._classes_meta = list(classes_meta)
        self._n_classes = self._prototypes.shape[0]
        self._topk = max(1, int(topk))

        # 校验类别表与原型矩阵一致
        if len(self._classes_meta) != self._n_classes:
            raise ValueError(
                f"classes_meta 长度 ({len(self._classes_meta)}) 与原型类数 "
                f"({self._n_classes}) 不一致"
            )

        # 校验 .meta.json (如果存在) 与本次加载一致
        meta_path = prototypes_path.with_suffix(".meta.json")
        if meta_path.is_file():
            try:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._meta = {}
        else:
            self._meta = {}

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def metadata(self) -> dict[str, Any]:
        """返回 .meta.json 内容 (准确率 / class_counts / 来源等)."""
        return dict(self._meta)

    def classify_segment(
        self,
        feature_arr: np.ndarray,
        start_sec: float,
        end_sec: float,
        duration_sec: float,
    ) -> ProtoClassifyResult:
        """对视频中 [start_sec, end_sec] 段做分类.

        Args:
            feature_arr: (N_samples, 2048) 整段视频的 PP-TSM 特征.
            start_sec / end_sec: 区间秒数.
            duration_sec: 视频总长 (用于换算秒→样本索引).

        Returns:
            :class:`ProtoClassifyResult`.
        """
        if feature_arr.ndim != 2 or feature_arr.shape[1] != self._prototypes.shape[1]:
            raise ValueError(
                f"特征维度不匹配: expected (N, {self._prototypes.shape[1]}), "
                f"got {feature_arr.shape}"
            )

        N = feature_arr.shape[0]
        if duration_sec <= 0:
            duration_sec = max(1.0, float(N))   # 退化路径
        feat_per_sec = N / duration_sec

        i0 = int(round(start_sec * feat_per_sec))
        i1 = int(round(end_sec * feat_per_sec))
        i1 = max(i1, i0 + 1)
        i1 = min(i1, N)
        i0 = max(0, i0)
        if i1 <= i0:
            i0, i1 = 0, min(1, N)

        slice_feat = feature_arr[i0:i1]   # (T, 2048)
        feat_mean = slice_feat.mean(axis=0).astype(np.float32)

        # cosine similarity
        norm = np.linalg.norm(feat_mean) + 1e-8
        feat_norm = feat_mean / norm
        sims = feat_norm @ self._prototypes_normalized.T   # (n_classes,)

        order = np.argsort(-sims)
        topk_ids = order[: self._topk].tolist()

        # 构造 topk 列表
        topk: list[tuple[int, str, float]] = []
        for idx in topk_ids:
            name = self._get_display_name(idx)
            topk.append((int(idx), name, float(sims[idx])))

        top1_id = int(order[0])
        return ProtoClassifyResult(
            label_id=top1_id,
            label_name=self._get_display_name(top1_id),
            cls_score=float(sims[top1_id]),
            topk=topk,
        )

    def _get_display_name(self, idx: int) -> str:
        if 0 <= idx < len(self._classes_meta):
            c = self._classes_meta[idx]
            return c.get("display_name") or c.get("name") or f"class_{idx}"
        return f"class_{idx}"
