"""单片段推理 (FR-013, data-model.md ``clip-prediction-v1`` schema).

职责:
- 给定 ``checkpoint`` + ``video_path``, 调用上游适配层得到 ``[num_classes]`` 概率向量;
- 取 Top-K, 解析对应 ``ActionClass`` 名称;
- 返回 :data:`PredictionResult` dict, 严格遵循 data-model.md ``clip-prediction-v1`` schema.

本模块**不**做 IO (写文件), 由 :mod:`cli.infer_clip` 决定是 stdout 还是写文件.

不在本模块的范围:
- 长视频滑窗 (那是 :mod:`pingpong_av.inference.sliding_window`, T057);
- 文件可读性预校验 (那是 cli 层的职责, FR-016);
- 后处理可视化 (那是 :mod:`pingpong_av.inference.visualizer`, T061+).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pingpong_av.utils.logging import get_logger

__all__ = ["infer_clip", "duration_seconds"]

_log = get_logger(__name__)


def infer_clip(
    *,
    checkpoint: str | Path,
    upstream_config: str | Path,
    video_path: str | Path,
    class_names: list[str],
    config_hash: str,
    topk: int = 5,
) -> dict[str, Any]:
    """对单段视频做一次推理, 返回 ``clip-prediction-v1`` schema 的 dict.

    参数:
        checkpoint: 训练得到的 ``.pdparams`` 路径.
        upstream_config: 上游格式的 YAML (通常是 ``experiments/<run>/upstream_config.yaml``;
                         由 train 阶段 snapshot 而来).
        video_path: 输入视频片段.
        class_names: 类别名称列表, 顺序对应 logits 第二维.
        config_hash: 训练时记录的 config_hash, 写入 result.model.config_hash 便于追溯.
        topk: 返回的 top-K 数量 (默认 5; 若 num_classes < topk 自动 clip 到 num_classes).

    返回:
        ``clip-prediction-v1`` 格式 dict (data-model.md):
            {
              "schema": "clip-prediction-v1",
              "input": {"video_path": ..., "duration_sec": ...},
              "model": {"checkpoint": ..., "config_hash": ...},
              "topk": [{"id": ..., "name": ..., "score": ...}, ...],
              "produced_at": "<ISO8601 UTC>"
            }

    抛出:
        :class:`pingpong_av.upstream_adapter.trainer.UpstreamRuntimeError`: 上游推理失败.
        FileNotFoundError: video / checkpoint / upstream_config 不存在.
    """
    from pingpong_av.upstream_adapter.trainer import run_upstream_infer

    checkpoint = Path(checkpoint).resolve()
    upstream_config = Path(upstream_config).resolve()
    video_path = Path(video_path).resolve()

    # 上游推理: 返回 [num_classes] 向量 (logits 或 softmax 概率)
    raw_scores = run_upstream_infer(
        upstream_config,
        checkpoint,
        video_path,
    )
    raw_scores = np.asarray(raw_scores).reshape(-1)

    # 决定是否需要 softmax: 上游不同模型 head 行为不一; 简单以"是否非负且和≈1"判定
    scores = _ensure_probabilities(raw_scores)

    n_classes = scores.shape[0]
    if len(class_names) != n_classes:
        raise ValueError(
            f"class_names 长度 ({len(class_names)}) 与模型输出维度 ({n_classes}) 不一致; "
            "请确认使用的 checkpoint 与 class_names 来自同一次训练."
        )

    eff_k = max(1, min(topk, n_classes))
    # argpartition 取无序 topk; 然后按 score 排序
    cand_idx = np.argpartition(-scores, kth=eff_k - 1)[:eff_k]
    cand_idx = cand_idx[np.argsort(-scores[cand_idx])]

    topk_list = [
        {
            "id": int(idx),
            "name": class_names[int(idx)],
            "score": float(scores[int(idx)]),
        }
        for idx in cand_idx
    ]

    duration = duration_seconds(video_path)

    return {
        "schema": "clip-prediction-v1",
        "input": {
            "video_path": str(video_path),
            "duration_sec": duration,
        },
        "model": {
            "checkpoint": str(checkpoint),
            "config_hash": config_hash,
        },
        "topk": topk_list,
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _ensure_probabilities(scores: np.ndarray) -> np.ndarray:
    """如果 scores 已经是概率 (非负且和≈1), 直接返回; 否则做一次稳定 softmax."""
    if scores.size == 0:
        return scores
    if (scores >= 0).all() and 0.99 <= float(scores.sum()) <= 1.01:
        return scores
    # 数值稳定 softmax
    s = scores - scores.max()
    exp = np.exp(s)
    return exp / exp.sum()


def duration_seconds(video_path: Path) -> float | None:
    """读取视频时长 (秒). 失败 / 不可解析时返回 None (不抛错, 由 CLI 决定怎么处理).

    优先用 PyAV (本项目业务依赖中已含 av); 失败则回退 OpenCV; 都失败则 None.
    """
    try:
        import av

        with av.open(str(video_path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                # av container.duration 单位是 AV_TIME_BASE = 1e6
                return float(container.duration / 1_000_000)
    except Exception as exc:  # 上层会决定是否记录
        _log.debug("av 读取时长失败: %s", exc)

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and n > 0:
                return float(n / fps)
    except Exception as exc:
        _log.debug("cv2 读取时长失败: %s", exc)

    return None
