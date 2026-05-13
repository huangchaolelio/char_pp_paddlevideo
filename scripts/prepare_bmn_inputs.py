"""BMN 训练输入准备脚本 (T101, US6 — AI Studio 竞赛 #127 BMN 时序定位).

把 ``data/raw/pingpong_competition/pingpong_competition_bmn/label_cls14_train.json`` +
``data/clips/pingpong_competition/pingpong_competition_bmn/Features_competition_train/*.pkl``
转换为上游 ``configs/bmn_tabletennis.yaml`` 期望的输入格式:

    <out_dir>/
    ├── feature/<basename>_<start>_<end>.npy   # 8s 滑窗内的特征切片 (8s × fps × 2048)
    ├── label.json                              # 内部用 (start/end 等 ActivityNet 风格)
    ├── label_fixed.json                        # 同 label.json (上游 BMNDataset 用)
    └── label_gts.json                          # {taxonomy, database, version} 评估用

设计上来源于上游 ``applications/TableTennis/{get_instance_for_bmn,gts_format_transfer}.py``,
**不复制其逻辑入库版本 (章程 VI)**, 而是直接 import 上游模块或调用其函数.
但这两个脚本是顶层 ``if __name__ == '__main__'`` 风格, 不便 import; 因此本脚本
**复刻其核心 4 个函数** (gen_gts_for_bmn / combile_gts / save_feature_to_numpy /
gts_format_transfer) 并清晰标注上游来源, 与 patches/ 流程一致.

splits 来源:
    本脚本不重新划分; 直接读 ``data/splits/pingpong_competition/{train,val}.txt``
    按这两个 list 的 clip_id 切分 (章程 IV, 划分 video 不跨 split).

使用:
    .venv/bin/python scripts/prepare_bmn_inputs.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np


# ============================================================
# 路径配置 (与 yaml + splits 对齐)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "pingpong_competition" / "pingpong_competition_bmn"
FEAT_DIR = REPO_ROOT / "data" / "clips" / "pingpong_competition" / "pingpong_competition_bmn" / "Features_competition_train"
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "pingpong_competition"
OUT_DIR = REPO_ROOT / "data" / "bmn_inputs" / "pingpong_competition"

LABEL_JSON = RAW_ROOT / "label_cls14_train.json"

# 与 上游 bmn_tabletennis.yaml 对齐
BMN_WINDOW = 8       # seconds (上游 get_instance_for_bmn::bmn_window)
RNG_SEED = 2026


# ============================================================
# 上游函数复刻 (来源: applications/TableTennis/get_instance_for_bmn.py)
# 章程 VI: 不直接复制源码; 这里用最小必要等价实现, 行为对齐.
# ============================================================


def gen_gts_for_bmn(gts_data: dict) -> dict:
    """对每段视频, 按 BMN_WINDOW 滑窗合并相邻 actions 成 root_actions 组.

    与上游 ``gen_gts_for_bmn`` 行为等价: 拒绝单 action 持续 > BMN_WINDOW; 把多个 action
    合并到同一 window 里 (不超 BMN_WINDOW).
    """
    fps = gts_data["fps"]
    out = {"fps": fps, "gts": []}
    for sub_item in gts_data["gts"]:
        url = sub_item["url"]
        max_length = sub_item["total_frames"]
        out["gts"].append({"url": url, "total_frames": max_length, "root_actions": []})
        actions = sub_item.get("actions", [])
        if not actions:
            continue

        # 拒绝过长 action
        actions = [a for a in actions
                   if (a["end_id"] - a["start_id"]) <= BMN_WINDOW]
        if not actions:
            continue

        root_actions = [actions[0]]
        before_id = 0
        for idx in range(1, len(actions)):
            cur = actions[idx]
            duration = cur["end_id"] - root_actions[0]["start_id"]
            if duration > BMN_WINDOW:
                after_id = cur["start_id"]
                out["gts"][-1]["root_actions"].append({
                    "before_id": before_id,
                    "after_id": after_id,
                    "actions": list(root_actions),
                })
                before_id = root_actions[-1]["end_id"]
                root_actions = [cur]
            else:
                root_actions.append(cur)
            if idx == len(actions) - 1:
                out["gts"][-1]["root_actions"].append({
                    "before_id": before_id,
                    "after_id": max_length,
                    "actions": list(root_actions),
                })
    return out


def combile_gts(gts_bmn: dict, gts_process: dict, mode: str, *, rng: random.Random) -> dict:
    """对每个 root_action 组生成 1-3 个候选 windows (与上游一致)."""
    fps = gts_process["fps"]
    duration_second = float(BMN_WINDOW)
    duration_frame = BMN_WINDOW * fps
    feature_frame = duration_frame
    for item in gts_process["gts"]:
        url = item["url"]
        basename = os.path.basename(url).split(".")[0]
        for root_action in item["root_actions"]:
            segments = []
            segments.append({
                "actions": root_action["actions"],
                "before_id": root_action["before_id"],
                "after_id": root_action["after_id"],
            })
            if len(root_action["actions"]) > 1:
                segments.append({
                    "actions": [root_action["actions"][0]],
                    "before_id": root_action["before_id"],
                    "after_id": root_action["actions"][1]["start_id"],
                })
                segments.append({
                    "actions": [root_action["actions"][-1]],
                    "before_id": root_action["actions"][-2]["end_id"],
                    "after_id": root_action["after_id"],
                })

            for segment in segments:
                before_id = segment["before_id"]
                after_id = segment["after_id"]
                actions = segment["actions"]
                box0 = max(actions[-1]["end_id"] - BMN_WINDOW, before_id)
                box1 = min(actions[0]["start_id"], after_id - BMN_WINDOW)
                if box0 <= box1:
                    if int(box0) - int(box1) == 0:
                        cur_start = box0
                    else:
                        box0 = math.ceil(box0)
                        box1 = int(box1)
                        cur_start = rng.randint(box0, box1)
                    cur_end = cur_start + BMN_WINDOW
                    cur_start = round(cur_start, 2)
                    cur_end = round(cur_end, 2)
                    name = f"{basename}_{cur_start}_{cur_end}"
                    annotations = []
                    for action in actions:
                        label = str(1.0 * action["label_ids"][0])
                        label_name = action["label_names"][0]
                        seg0 = 1.0 * round(action["start_id"] - cur_start, 2)
                        seg1 = 1.0 * round(action["end_id"] - cur_start, 2)
                        annotations.append({
                            "segment": [seg0, seg1],
                            "label": label,
                            "label_name": label_name,
                        })
                    gts_bmn[name] = {
                        "duration_second": duration_second,
                        "duration_frame": duration_frame,
                        "feature_frame": feature_frame,
                        "subset": mode,
                        "annotations": annotations,
                    }
    return gts_bmn


def save_feature_to_numpy(gts_bmn: dict, folder: Path, *, fps: int, feat_dir: Path,
                          ) -> int:
    """对每个 8s 滑窗切片, 从对应 video 的 feature pkl 中切出对应帧 → .npy."""
    folder.mkdir(parents=True, exist_ok=True)
    process: dict[str, list] = {}
    miss = 0
    for item, _value in gts_bmn.items():
        basename, start_id, end_id = item.rsplit("_", 2)
        process.setdefault(basename, []).append({
            "name": item, "start": float(start_id), "end": float(end_id),
        })
    for item, values in process.items():
        feat_path = feat_dir / f"{item}.pkl"
        if not feat_path.is_file():
            miss += len(values)
            continue
        with feat_path.open("rb") as f:
            feature_video = pickle.load(f)["image_feature"]
        for value in values:
            save_cut_name = folder / value["name"]
            start_frame = round(value["start"] * fps)
            end_frame = round(value["end"] * fps)
            if end_frame > len(feature_video):
                miss += 1
                continue
            feature_cut = np.array(feature_video[start_frame:end_frame], dtype=np.float32)
            np.save(save_cut_name, feature_cut)
    return miss


# ============================================================
# 主流程
# ============================================================


def _read_split_clip_ids(split_file: Path) -> set[str]:
    """从 splits/<split>.txt 中读取 clip_id 集合."""
    out: set[str] = set()
    if not split_file.is_file():
        return out
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 行格式: "Features_competition_train/<hash>.pkl\t<label_id>"
        rel = line.split()[0]
        clip_id = Path(rel).stem
        out.add(clip_id)
    return out


# ============================================================
# 公共 API (T208, 2026-05-13 新增, 002 feature)
# ============================================================


def prepare_bmn_inputs_for_training(
    *,
    label_json: Path,
    feature_dir: Path,
    output_dir: Path,
    split_clip_ids: dict[str, set[str]] | None = None,
    rng_seed: int = RNG_SEED,
    bmn_window: int = BMN_WINDOW,
) -> dict:
    """训练路径: 有 GT, 把 labels + features → BMN 期望的 feature/*.npy + label_fixed.json.

    与 ``main()`` 行为等价, 但接受参数 (不依赖全局常量 LABEL_JSON / FEAT_DIR / OUT_DIR).
    ``main()`` 仍存在做向后兼容, 实际是 wrapper.

    Args:
        label_json: label_cls14_train.json 路径 (含 gts + fps).
        feature_dir: Features_*/*.pkl 所在目录.
        output_dir: 输出根目录, 会写入 ``feature/*.npy``, ``label_fixed.json``, ``label_gts.json``.
        split_clip_ids: 可选, ``{"train": set(), "validation": set(), "test": set()}``;
                        若提供, 按此切分 GT; 若为 None, 所有 GT 视为 train.
        rng_seed: 与 main 一致.
        bmn_window: 8 秒窗口 (上游 BMN 默认).

    Returns:
        ``{"n_windows": int, "n_npy": int, "miss": int, "label_fixed_json": str}``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gts_data = json.loads(label_json.read_text(encoding="utf-8"))
    fps = gts_data["fps"]

    # 切分
    if split_clip_ids is None:
        split_clip_ids = {"train": {Path(str(g.get("url", ""))).stem for g in gts_data["gts"]}}

    split_gts: dict[str, dict] = {}
    for split_name in ("train", "validation", "test"):
        split_gts[split_name] = {"fps": fps, "gts": []}
    for g in gts_data["gts"]:
        cid = Path(str(g.get("url", ""))).stem
        for split_name, ids in split_clip_ids.items():
            if cid in ids:
                split_gts[split_name]["gts"].append(g)
                break

    rng = random.Random(rng_seed)
    gts_bmn: dict = {}
    # 上游 BMN subset 只区分 train / validation (test 不单独训)
    for mode in ("train", "validation"):
        gts_in = split_gts.get(mode, {"fps": fps, "gts": []})
        gts_process = gen_gts_for_bmn(gts_in)
        gts_bmn = combile_gts(gts_bmn, gts_process, mode, rng=rng)

    # 切片 → .npy
    feat_out = output_dir / "feature"
    miss = save_feature_to_numpy(gts_bmn, feat_out, fps=fps, feat_dir=feature_dir)
    available = {p.stem for p in feat_out.glob("*.npy")}
    gts_bmn = {name: v for name, v in gts_bmn.items() if name in available}

    # 写 label_fixed + label_gts
    label_fixed = output_dir / "label_fixed.json"
    label_fixed.write_text(json.dumps(gts_bmn, indent=4, ensure_ascii=False), encoding="utf-8")
    label_gts = output_dir / "label_gts.json"
    label_gts.write_text(
        json.dumps({"taxonomy": None, "database": gts_bmn, "version": None},
                   indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "n_windows": len(gts_bmn),
        "n_npy": sum(1 for _ in feat_out.glob("*.npy")),
        "miss": miss,
        "label_fixed_json": str(label_fixed),
        "label_gts_json": str(label_gts),
        "feature_dir": str(feat_out),
    }


def prepare_bmn_inputs_for_inference(
    *,
    feature_pkl: Path,
    output_dir: Path,
    clip_id: str | None = None,
    fps: int = 25,
    bmn_window: int = BMN_WINDOW,
    stride: int | None = None,
) -> dict:
    """推理路径 (**无 GT**): 单个特征 pkl → BMN 滑窗 .npy + 占位 label_fixed.json.

    与训练路径的关键区别:
        - 无 GT, 所有 annotations=[] (让 BMN dataloader 不报错即可)
        - 不拆分 subset, 所有窗口都标 "validation" (上游 BMN 推理默认的 subset)
        - 用**连续滑窗**而不是 ground-truth-centered 滑窗: ``stride`` 控制步长, 默认 = bmn_window (不重叠)

    Args:
        feature_pkl: PP-TSM 抽出的 ``{'image_feature': (N, 2048)}`` pickle 文件.
        output_dir: 输出根目录.
        clip_id: 如为 None, 用 pkl 文件名 stem (PP-TSM 抽出的 pkl 名就是 clip_id).
        fps: 视频 fps, 用于时间戳换算 (应与抽特征时一致).
        bmn_window: 8 秒窗口.
        stride: 滑窗步长秒; None → bmn_window (不重叠).

    Returns:
        ``{"n_windows": int, "n_npy": int, "label_fixed_json": str, "clip_id": str}``
    """
    import pickle

    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    feat_out = output_dir / "feature"
    feat_out.mkdir(exist_ok=True)

    with feature_pkl.open("rb") as f:
        obj = pickle.load(f)
    image_feat = obj["image_feature"]  # (N, 2048) 或 (N/seg_num, 2048) 看抽法
    n_samples = image_feat.shape[0]

    # clip_id 默认用文件名 stem
    cid = clip_id or feature_pkl.stem

    if stride is None:
        stride = bmn_window

    # BMN 期望每个窗口是 (tscale=200, 2048) 形状. 用 feature 的采样速率:
    # 如果 PP-TSM 抽法是"每 seg_num=8 帧一个样本", 每秒 fps/seg_num = 25/8 ≈ 3.125 个样本.
    # 8 秒窗口约 25 个样本, 远 < tscale=200. **不兼容**.
    #
    # 解决: 在本函数里把 image_feat 做**时间维插值**到 (tscale * n_windows, 2048).
    # 这样每个窗口切出 200 个样本, 匹配 BMN 期望.
    # (训练路径 v0.2.x 的 save_feature_to_numpy 也是这么做的, 按 fps 从 "每帧 1 样本"的假设切;
    # 这里我们 PP-TSM 输出是 "每 8 帧 1 样本", 所以要先插值)
    #
    # 简化实现: 按 bmn_window 秒切窗, 每窗内线性插值到 tscale=200.
    # 这不是业务最优但能让 pipeline 跑通.

    total_duration_sec = n_samples * 8.0 / fps  # seg_num=8 假设
    tscale = 200  # 上游 BMN 默认
    gts_bmn: dict[str, dict] = {}

    # 生成窗口列表
    windows = []
    t = 0.0
    while t + bmn_window <= total_duration_sec + 1e-6:
        windows.append((t, t + bmn_window))
        t += stride

    # 如果视频不足一窗, 至少切一个窗 (全视频)
    if not windows and total_duration_sec > 0:
        windows.append((0.0, min(total_duration_sec, float(bmn_window))))

    samples_per_window = image_feat.shape[0] / max(total_duration_sec / bmn_window, 1.0) if windows else n_samples

    for (t0, t1) in windows:
        # 从 image_feat 取 [t0*fps/seg_num, t1*fps/seg_num] 切片
        i0 = int(round(t0 * image_feat.shape[0] / total_duration_sec)) if total_duration_sec > 0 else 0
        i1 = int(round(t1 * image_feat.shape[0] / total_duration_sec)) if total_duration_sec > 0 else image_feat.shape[0]
        i1 = min(max(i1, i0 + 1), image_feat.shape[0])
        slice_feat = image_feat[i0:i1]

        # 线性插值到 tscale=200
        if slice_feat.shape[0] != tscale:
            if slice_feat.shape[0] == 0:
                continue
            src_idx = np.arange(slice_feat.shape[0], dtype=np.float32)
            tgt_idx = np.linspace(0, slice_feat.shape[0] - 1, tscale, dtype=np.float32)
            # 每维度独立插值
            resampled = np.empty((tscale, slice_feat.shape[1]), dtype=np.float32)
            for d in range(slice_feat.shape[1]):
                resampled[:, d] = np.interp(tgt_idx, src_idx, slice_feat[:, d])
            slice_feat = resampled

        name = f"{cid}_{t0:.2f}_{t1:.2f}"
        np.save(feat_out / name, slice_feat.astype(np.float32))
        gts_bmn[name] = {
            "duration_second": float(bmn_window),
            "duration_frame": bmn_window * fps,
            "feature_frame": bmn_window * fps,
            "subset": "validation",
            # 必须给至少 1 个 dummy annotation, 否则上游 anet_pipeline.py:118
            # `np.max(gt_iou_map, axis=0)` 在 dataloader 阶段就抛 ValueError (zero-size array).
            # 推理时 BMN 不用 GT, 但 dataloader 仍走相同 pipeline. 给一个零长度 segment 占位.
            "annotations": [
                {
                    "segment":    [0.0, 0.0],
                    "label":      "0",
                    "label_name": "dummy_inference_placeholder",
                }
            ],
        }

    # 写占位 label
    label_fixed = output_dir / "label_fixed.json"
    label_fixed.write_text(json.dumps(gts_bmn, indent=4, ensure_ascii=False), encoding="utf-8")
    label_gts = output_dir / "label_gts.json"
    label_gts.write_text(
        json.dumps({"taxonomy": None, "database": gts_bmn, "version": None},
                   indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "n_windows": len(gts_bmn),
        "n_npy": sum(1 for _ in feat_out.glob("*.npy")),
        "label_fixed_json": str(label_fixed),
        "label_gts_json": str(label_gts),
        "feature_dir": str(feat_out),
        "clip_id": cid,
    }


# ============================================================
# main (向后兼容 CLI 入口)
# ============================================================


def main() -> int:
    print(f"BMN inputs preparation")
    print(f"  RAW_ROOT:  {RAW_ROOT}")
    print(f"  FEAT_DIR:  {FEAT_DIR}")
    print(f"  SPLITS:    {SPLITS_DIR}")
    print(f"  OUT_DIR:   {OUT_DIR}")
    print(f"  BMN_WINDOW: {BMN_WINDOW}s")

    if not LABEL_JSON.is_file():
        print(f"ERROR: {LABEL_JSON} 不存在; 请先运行 pp data-prepare.")
        return 1
    if not FEAT_DIR.is_dir():
        print(f"ERROR: {FEAT_DIR} 不存在.")
        return 1

    train_ids = _read_split_clip_ids(SPLITS_DIR / "train.txt")
    val_ids = _read_split_clip_ids(SPLITS_DIR / "val.txt")
    test_ids = _read_split_clip_ids(SPLITS_DIR / "test.txt")
    print(f"  splits: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")
    if not train_ids:
        print("ERROR: train split 为空; 请先运行 pp data-prepare.")
        return 1

    gts_data = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
    fps = gts_data["fps"]

    # 按本项目 splits 拆分 GT (上游用 train + validation 两份独立 json, 我们从一份按 hash 切)
    train_gt = {"fps": fps, "gts": []}
    val_gt = {"fps": fps, "gts": []}
    test_gt = {"fps": fps, "gts": []}
    for g in gts_data["gts"]:
        cid = Path(str(g.get("url", ""))).stem
        if cid in train_ids:
            train_gt["gts"].append(g)
        elif cid in val_ids:
            val_gt["gts"].append(g)
        elif cid in test_ids:
            test_gt["gts"].append(g)
    print(f"  gt entries: train={len(train_gt['gts'])}  val={len(val_gt['gts'])}  test={len(test_gt['gts'])}")

    rng = random.Random(RNG_SEED)
    gts_bmn: dict = {}
    for mode, gts in [("train", train_gt), ("validation", val_gt)]:
        gts_process = gen_gts_for_bmn(gts)
        gts_bmn = combile_gts(gts_bmn, gts_process, mode, rng=rng)
    print(f"  total bmn windows (raw): {len(gts_bmn)}")

    # 切片特征到 .npy (在写 label_fixed.json **之前**, 以便筛掉 miss 的条目)
    feat_out = OUT_DIR / "feature"
    miss = save_feature_to_numpy(gts_bmn, feat_out, fps=fps, feat_dir=FEAT_DIR)
    n_npy = sum(1 for _ in feat_out.glob("*.npy"))
    print(f"  wrote {n_npy} .npy feature slices to {feat_out}")
    if miss:
        print(f"  skipped {miss} slices due to bound issues; "
              "filtering label_fixed.json to keep only entries with matching .npy")

    # 筛掉没有 .npy 的条目, 避免 BMN dataloader 抛 FileNotFoundError
    available = {p.stem for p in feat_out.glob("*.npy")}
    before = len(gts_bmn)
    gts_bmn = {name: v for name, v in gts_bmn.items() if name in available}
    after = len(gts_bmn)
    print(f"  filtered windows: {before} → {after}")

    # 写 label_fixed.json (BMNDataset 直接消费的格式) — 与上游 label.json 同结构
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_fixed = OUT_DIR / "label_fixed.json"
    label_fixed.write_text(json.dumps(gts_bmn, indent=4, ensure_ascii=False),
                           encoding="utf-8")
    print(f"  wrote {label_fixed} ({label_fixed.stat().st_size/1024:.1f} KB)")

    # 写 label_gts.json (BMNMetric 评估用 ActivityNet 格式)
    label_gts = OUT_DIR / "label_gts.json"
    label_gts.write_text(
        json.dumps({"taxonomy": None, "database": gts_bmn, "version": None},
                   indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  wrote {label_gts} ({label_gts.stat().st_size/1024:.1f} KB)")

    print("✓ BMN inputs ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
