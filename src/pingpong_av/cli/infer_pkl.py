"""``pp infer-pkl`` 子命令: 用上游官方 VideoSwin TableTennis 模型推理一个 .pkl 样例.

输入 pkl 格式 (上游 `example_tennis.pkl` 实测):
    (video_name: str, labels: dict, frames: list[bytes])
    - frames: 每元素是一帧 JPEG 字节流 (1280x720 RGB)
    - labels: 含 '正反手' / '动作类型' / '发球' 三组任务的真值标签

推理流程:
    1. 反序列化 pkl, 取出 frames JPEG bytes 列表
    2. JPEG 解码 → numpy RGB
    3. 均匀采样 num_seg=32 帧 (与上游 videoswin_tabletennis.yaml runtime_cfg.test.num_seg 对齐)
    4. Resize (short_size=256) + CenterCrop 224 + Normalize (ImageNet 均值方差)
    5. Stack 成 [1, 3, T, H, W] 喂给上游 RecognizerTransformer
    6. 取 softmax(logits), 输出 Top-K + 与 pkl 内 GT 标签对照

退出码:
    0  成功
    1  用户输入错 (pkl / checkpoint 不存在)
    2  环境问题 (上游不可导)
    4  运行时失败 (模型加载 / 推理异常)
"""

from __future__ import annotations

import io
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np

from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


# ImageNet 标准化参数 (上游 PIPELINE.test 也是这一组)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run(
    *,
    pkl_path: str,
    checkpoint: str,
    topk: int,
    num_seg: int,
    output_path: str | None,
) -> int:
    """执行 infer-pkl. 返回应当作为进程退出码使用的整数."""

    # ---- 1. 校验输入 ----
    pkl = Path(pkl_path).resolve()
    if not pkl.is_file():
        click.echo(f"ERROR: pkl 文件不存在: {pkl}", err=True)
        return 1
    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        click.echo(
            f"ERROR: VideoSwin_tennis checkpoint 不存在: {ckpt}\n"
            "请先下载 (380MB):\n"
            "  mkdir -p data/raw/pingpong_public/checkpoints\n"
            "  curl -fL -o data/raw/pingpong_public/checkpoints/VideoSwin_tennis.pdparams \\\n"
            "    https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_tennis.pdparams",
            err=True,
        )
        return 1
    if topk < 1:
        click.echo(f"ERROR: --topk 必须 ≥ 1, 实际 {topk}", err=True)
        return 1
    if num_seg < 1:
        click.echo(f"ERROR: --num-seg 必须 ≥ 1, 实际 {num_seg}", err=True)
        return 1

    # ---- 2. 反序列化 pkl ----
    try:
        with pkl.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        click.echo(f"ERROR: pkl 反序列化失败: {type(exc).__name__}: {exc}", err=True)
        return 1
    video_name, gt_labels, frame_bytes_list = _parse_tennis_pkl(obj)
    if frame_bytes_list is None:
        click.echo(
            f"ERROR: pkl 格式不识别 (期望 (name, labels_dict, list[bytes])): "
            f"{type(obj).__name__}",
            err=True,
        )
        return 1

    click.echo(
        f"[infer-pkl] video={video_name!r}  frames={len(frame_bytes_list)}  "
        f"gt_labels={gt_labels}",
        err=True,
    )

    # ---- 3. 解码 + 采样 + 预处理 ----
    try:
        tensor = _prepare_input_tensor(
            frame_bytes_list, num_seg=num_seg, image_size=224, short_side=256,
        )
    except Exception as exc:
        click.echo(f"ERROR: 帧预处理失败: {type(exc).__name__}: {exc}", err=True)
        return 4
    click.echo(
        f"[infer-pkl] sampled {num_seg} frames → tensor shape {tensor.shape} "
        f"(range [{tensor.min():.3f}, {tensor.max():.3f}])",
        err=True,
    )

    # ---- 4. 加载模型 + 推理 ----
    try:
        from pingpong_av.models.videoswin_tennis import (
            TABLETENNIS_CLASS_NAMES,
            TabletennisModelError,
            load_videoswin_tennis_model,
        )
    except ImportError as exc:
        click.echo(f"ERROR: 无法 import 模型模块: {exc}", err=True)
        return 2

    try:
        model, up_cfg = load_videoswin_tennis_model(checkpoint=ckpt)
    except TabletennisModelError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        return 1
    except Exception as exc:
        click.echo(f"ERROR: 模型加载失败: {type(exc).__name__}: {exc}", err=True)
        return 4

    try:
        import paddle
        model.eval()
        x = paddle.to_tensor(tensor)
        with paddle.no_grad():
            # 上游 RecognizerTransformer.test_step 期望 list[Tensor]
            out = model([x], mode="test")
        scores = out.numpy().reshape(-1)
    except Exception as exc:
        click.echo(f"ERROR: 模型推理失败: {type(exc).__name__}: {exc}", err=True)
        return 4

    probs = _ensure_probs(scores)
    n_classes = probs.shape[0]
    eff_k = min(topk, n_classes)
    top_idx = np.argpartition(-probs, kth=eff_k - 1)[:eff_k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]

    class_names = TABLETENNIS_CLASS_NAMES[:n_classes] if len(TABLETENNIS_CLASS_NAMES) >= n_classes \
        else [f"class_{i}" for i in range(n_classes)]

    topk_list = [
        {"id": int(i), "name": class_names[int(i)], "score": float(probs[int(i)])}
        for i in top_idx
    ]

    # ---- 5. 比对 ground truth (如果有 '动作类型' 标签) ----
    gt_action = gt_labels.get("动作类型") if isinstance(gt_labels, dict) else None
    gt_action_name = class_names[gt_action] if isinstance(gt_action, int) and 0 <= gt_action < n_classes else None
    pred_top1 = topk_list[0]
    hit = (gt_action == pred_top1["id"]) if gt_action is not None else None

    payload = {
        "schema": "pkl-prediction-v1",
        "input": {
            "pkl_path": str(pkl),
            "video_name": video_name,
            "n_frames_in_pkl": len(frame_bytes_list),
            "n_frames_sampled": num_seg,
        },
        "model": {
            "checkpoint": str(ckpt),
            "framework": str(up_cfg.MODEL.framework),
            "backbone": str(up_cfg.MODEL.backbone.name),
            "head": str(up_cfg.MODEL.head.name),
            "num_classes": int(up_cfg.MODEL.head.num_classes),
        },
        "ground_truth": gt_labels,
        "ground_truth_action_id": gt_action,
        "ground_truth_action_name": gt_action_name,
        "prediction": {
            "topk": topk_list,
            "top1_match_gt": hit,
        },
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }

    if output_path:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(f"✓ 结果已写入: {out}", err=True)

    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    # 人类摘要
    click.echo("", err=True)
    click.echo(f"📊 Top-{eff_k} 预测:", err=True)
    for i, item in enumerate(topk_list, 1):
        marker = " ← GT" if gt_action is not None and item["id"] == gt_action else ""
        click.echo(f"  {i}. {item['name']} (id={item['id']})  prob={item['score']:.4f}{marker}", err=True)
    if hit is not None:
        click.echo("", err=True)
        if hit:
            click.echo(f"✅ Top-1 与 GT 一致 (action_id={gt_action})", err=True)
        else:
            click.echo(
                f"⚠ Top-1 (id={pred_top1['id']}) 与 GT (id={gt_action}) 不一致 — "
                "样例 pkl 只是上游 inference 演示, 不代表模型整体准确率.",
                err=True,
            )
    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _parse_tennis_pkl(obj: Any) -> tuple[str, dict[str, Any], list[bytes] | None]:
    """容错解析上游 example_tennis.pkl: (name, labels_dict, frame_bytes_list)."""
    if not isinstance(obj, (tuple, list)) or len(obj) < 3:
        return ("?", {}, None)
    name, labels, frames = obj[0], obj[1], obj[2]
    if not isinstance(frames, (list, tuple)) or not all(isinstance(b, (bytes, bytearray)) for b in frames):
        return (str(name), {}, None)
    labels_dict = labels if isinstance(labels, dict) else {}
    return (str(name), labels_dict, list(frames))


def _prepare_input_tensor(
    frame_bytes_list: list[bytes],
    *,
    num_seg: int,
    image_size: int,
    short_side: int,
) -> np.ndarray:
    """JPEG bytes → 均匀采样 → resize → center crop → normalize → [1, 3, T, H, W] float32."""
    from PIL import Image

    n = len(frame_bytes_list)
    if n == 0:
        raise ValueError("pkl 中没有任何帧")
    if n <= num_seg:
        # 不够采样 num_seg 帧时, 用线性插值索引 (可能重复)
        idx = np.linspace(0, n - 1, num_seg).round().astype(int).tolist()
    else:
        # 均匀采样 num_seg 个索引 (与 yaml UniformCrop 风格一致)
        idx = np.linspace(0, n - 1, num_seg).round().astype(int).tolist()

    frames = []
    for i in idx:
        img = Image.open(io.BytesIO(frame_bytes_list[i])).convert("RGB")
        # Scale: 短边到 short_side
        w, h = img.size
        if h < w:
            new_h = short_side
            new_w = int(round(w * short_side / h))
        else:
            new_w = short_side
            new_h = int(round(h * short_side / w))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # CenterCrop image_size
        left = max(0, (new_w - image_size) // 2)
        top = max(0, (new_h - image_size) // 2)
        img = img.crop((left, top, left + image_size, top + image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        # 转为 [3, H, W]
        arr = arr.transpose(2, 0, 1)
        frames.append(arr)

    # Stack: [T, 3, H, W] → 转为 [N=1, C=3, T, H, W] (上游 SwinTransformer3D 期望此格式)
    arr_t = np.stack(frames, axis=0)              # [T, 3, H, W]
    arr_t = arr_t.transpose(1, 0, 2, 3)           # [3, T, H, W]
    arr_t = arr_t[np.newaxis, ...]                # [1, 3, T, H, W]
    return arr_t.astype(np.float32)


def _ensure_probs(scores: np.ndarray) -> np.ndarray:
    """logits → softmax 兜底."""
    if scores.size == 0:
        return scores
    if (scores >= 0).all() and 0.99 <= float(scores.sum()) <= 1.01:
        return scores
    s = scores - scores.max()
    exp = np.exp(s)
    return exp / exp.sum()
