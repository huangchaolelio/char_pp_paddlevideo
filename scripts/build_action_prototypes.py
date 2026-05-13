"""scripts/build_action_prototypes.py: 从 AI Studio 竞赛 #127 数据构建 14 类原型 (training-free).

对应 003 feature (label-classifier-head, 简化版 — 原型匹配方案):
    BMN 输出 `(start_sec, end_sec, score)` proposals → 我们用 PP-TSM 特征 + 14 类原型
    做 cosine similarity argmax → 给 proposal 加 `label_id + label_name + cls_score`.

数据来源:
    label_cls14_train.json (729 视频 / 19054 actions / 14 类)
    + Features_competition_train/<clip_id>.pkl (PP-TSM/PP-TSN 2048-d 特征)

输出:
    data/raw/pretrained/prototypes/action_prototypes_14.npy        # (14, 2048) float32
    data/raw/pretrained/prototypes/action_prototypes_14.meta.json  # 准确率 + class_counts

实测准确率 (leave-one-out, 19054 actions):
    Top-1: 53.4%   远 > 随机 7.1%
    Big classes (>1000 instances): 41-89% (侧身拉 89%, 拧 86%, 控制 72%)
    Small classes (<100): 0-56% (类不平衡严重, 期望)

退出码:
    0  成功
    1  数据缺失 (GT json / Features dir 不存在)
    2  环境问题 (numpy 不可用 — 应该不会)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 14-class action prototypes from AI Studio data")
    parser.add_argument(
        "--gt-json",
        type=str,
        default="data/raw/pingpong_competition/pingpong_competition_bmn/label_cls14_train.json",
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        default="data/clips/pingpong_competition/pingpong_competition_bmn/Features_competition_train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/pretrained/prototypes",
    )
    parser.add_argument("--n-classes", type=int, default=14)
    parser.add_argument("--feat-dim", type=int, default=2048)
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="跳过 leave-one-out 校验 (大数据集快速重建用)",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_json)
    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.output_dir)

    if not gt_path.is_absolute():
        gt_path = _REPO_ROOT / gt_path
    if not feat_dir.is_absolute():
        feat_dir = _REPO_ROOT / feat_dir
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    if not gt_path.is_file():
        print(f"ERROR: GT JSON 不存在: {gt_path}", file=sys.stderr)
        return 1
    if not feat_dir.is_dir():
        print(f"ERROR: 特征目录不存在: {feat_dir}", file=sys.stderr)
        return 1

    print(f"GT JSON:  {gt_path}")
    print(f"Feat dir: {feat_dir}")
    print(f"Output:   {out_dir}")
    print()

    # ---- Step 1: 加载 GT ----
    print("=== Step 1: 加载 GT ===")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    fps = gt["fps"]
    print(f"  fps: {fps}")
    print(f"  videos: {len(gt['gts'])}")

    # ---- Step 2: 抽取所有 action 特征 ----
    print("\n=== Step 2: 抽取所有 action 特征 ===")
    pkl_cache: dict[str, np.ndarray] = {}
    action_feats: list[np.ndarray] = []
    action_labels: list[int] = []
    n_processed = 0
    n_skipped = 0

    for g_idx, g in enumerate(gt["gts"]):
        url = g["url"]
        total_frames = g["total_frames"]
        clip_id = Path(url).stem

        pkl_path = feat_dir / f"{clip_id}.pkl"
        if not pkl_path.is_file():
            n_skipped += len(g.get("actions", []))
            continue

        if clip_id not in pkl_cache:
            with pkl_path.open("rb") as f:
                arr = pickle.load(f)["image_feature"]
            pkl_cache[clip_id] = arr

        arr = pkl_cache[clip_id]
        n_feat = arr.shape[0]
        duration_sec = total_frames / fps
        feat_per_sec = n_feat / duration_sec   # ≈ fps (上游约每帧一特征)

        for a in g.get("actions", []):
            start_sec = float(a["start_id"])
            end_sec = float(a["end_id"])
            labels = a.get("label_ids", [])
            if not labels:
                n_skipped += 1
                continue
            label = int(labels[0])

            i0 = int(round(start_sec * feat_per_sec))
            i1 = int(round(end_sec * feat_per_sec))
            i1 = max(i1, i0 + 1)
            i1 = min(i1, n_feat)
            i0 = max(0, i0)

            if i1 <= i0:
                n_skipped += 1
                continue

            slice_feat = arr[i0:i1]
            feat_mean = slice_feat.mean(axis=0).astype(np.float32)

            action_feats.append(feat_mean)
            action_labels.append(label)
            n_processed += 1

        # LRU-like 缓存清理 (避免 OOM)
        if g_idx % 100 == 99:
            keep_keys = list(pkl_cache.keys())[-20:]
            pkl_cache = {k: v for k, v in pkl_cache.items() if k in keep_keys}
            if g_idx % 200 == 199:
                print(f"  [{g_idx+1}/{len(gt['gts'])}] processed {n_processed} actions...")

    print(f"\n  ✓ processed: {n_processed}  skipped: {n_skipped}")

    if n_processed == 0:
        print("ERROR: 没抽到任何 action 特征 (检查 GT json 与 feat-dir)", file=sys.stderr)
        return 1

    X = np.stack(action_feats)
    y = np.array(action_labels, dtype=np.int32)
    print(f"  X shape: {X.shape}  y shape: {y.shape}")

    # ---- Step 3: 构建原型 ----
    print(f"\n=== Step 3: 构建 {args.n_classes} 类原型 (各类均值) ===")
    n_classes = int(args.n_classes)
    prototypes = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    class_counts = np.zeros(n_classes, dtype=np.int32)
    sums = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        class_counts[c] = int(mask.sum())
        if mask.sum() > 0:
            sums[c] = X[mask].sum(axis=0)
            prototypes[c] = sums[c] / mask.sum()
    print(f"  类分布: {class_counts.tolist()}")

    # ---- Step 4: LOO 验证 ----
    if not args.skip_loo:
        print(f"\n=== Step 4: Leave-one-out 验证 ===")
        def l2norm(x):
            n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
            return x / n

        Xn = l2norm(X)
        Pn = l2norm(prototypes)
        preds_direct = (Xn @ Pn.T).argmax(axis=1)
        acc_direct = float((preds_direct == y).mean())
        print(f"  Direct: acc={acc_direct:.4f}")

        preds_loo = np.zeros(len(y), dtype=np.int32)
        for i in range(len(y)):
            c_true = y[i]
            proto_loo = prototypes.copy()
            if class_counts[c_true] > 1:
                proto_loo[c_true] = (sums[c_true] - X[i]) / (class_counts[c_true] - 1)
            sim = Xn[i] @ l2norm(proto_loo).T
            preds_loo[i] = sim.argmax()
        acc_loo = float((preds_loo == y).mean())
        print(f"  LOO: acc={acc_loo:.4f}")

        per_class_acc: list[float] = []
        for c in range(n_classes):
            mask = (y == c)
            if mask.sum() > 0:
                cls_acc = float((preds_loo[mask] == c).mean())
            else:
                cls_acc = 0.0
            per_class_acc.append(cls_acc)
    else:
        acc_direct = float("nan")
        acc_loo = float("nan")
        per_class_acc = []

    # ---- Step 5: 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"action_prototypes_{n_classes}.npy", prototypes)
    meta = {
        "schema": "action-prototypes-v1",
        "n_classes": n_classes,
        "feat_dim": int(X.shape[1]),
        "n_actions_used": int(n_processed),
        "class_counts": class_counts.tolist(),
        "accuracy_direct": acc_direct,
        "accuracy_leave_one_out": acc_loo,
        "per_class_accuracy_loo": per_class_acc,
        "produced_from": str(gt_path),
        "feature_source": str(feat_dir),
        "feature_sampling_note": (
            "上游 AI Studio 数据: 约每帧 1 个 PP-TSN 特征 (1:1). "
            "推理时若用本仓库的 pp extract-feat (seg_num=8, 每 8 帧 1 特征), "
            "需先做时间维线性插值上采样到 1:1 才能用此原型分类."
        ),
    }
    meta_path = out_dir / f"action_prototypes_{n_classes}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ✓ saved: {out_dir}/action_prototypes_{n_classes}.npy ({prototypes.nbytes/1024:.1f}KB)")
    print(f"  ✓ saved: {meta_path}")
    print(f"\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
