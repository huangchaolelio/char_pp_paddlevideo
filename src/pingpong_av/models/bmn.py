"""BMN (Boundary-Matching Network) 时序定位模型配置加载器.

把本项目业务配置 ``configs/datasets/pingpong_competition_bmn.yaml`` (描述数据)
+ ``configs/models/bmn_pingpong.yaml`` (描述模型, 仅极少必填字段) 合并到上游
``third_party/PaddleVideo/applications/TableTennis/configs/bmn_tabletennis.yaml``
模板, 写入临时 yaml 供 ``run_upstream_train`` 调用.

设计理念 (与 :mod:`pingpong_av.models.pp_tsm` 一致):
- 业务 yaml 不复制上游字段; 只覆盖必要的路径与 epochs
- 每次调用都生成一份 fresh yaml (config_hash 仍由原业务 yaml 计算)
- 不修改上游 yaml 入库版本 (章程 VI)

输入路径覆盖 (强制):
    PIPELINE.train.load_feat.feat_path  → data/bmn_inputs/.../feature
    PIPELINE.valid.load_feat.feat_path  → 同上
    PIPELINE.test.load_feat.feat_path   → 同上
    DATASET.{train,valid,test}.file_path → data/bmn_inputs/.../label_fixed.json
    METRIC.file_path                     → 同上
    METRIC.ground_truth_filename         → data/bmn_inputs/.../label_gts.json
    epochs                               → user_cfg.train.epochs
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import yaml

from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

__all__ = ["load_bmn_config"]

_log = get_logger(__name__)


# 上游 BMN yaml (release/2.2.0)
_UPSTREAM_YAML_REL = Path(
    "third_party/PaddleVideo/applications/TableTennis/configs/bmn_tabletennis.yaml"
)


def load_bmn_config(
    user_cfg: dict[str, Any],
    *,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    splits_dir: Path | None = None,  # noqa: ARG001 — 与 pp_tsm loader 签名一致, BMN 不用
) -> tuple[Path, dict[str, Any]]:
    """加载 BMN 上游配置, 用业务 user_cfg 字段覆盖路径与 epochs.

    Args:
        user_cfg: 已合并的业务配置 (model + dataset + train sections).
                  期望:
                    user_cfg["model"]["bmn_inputs_dir"]: BMN 输入数据根目录
                                                         (默认 data/bmn_inputs/<dataset>/)
                    user_cfg["train"]["epochs"]:         训练轮数
                    user_cfg["dataset"]["batch_size"]:   batch_size (可选)
        repo_root: 仓库根 (None 时自动探测).
        output_dir: 临时 yaml 落盘目录 (None 时用 tempdir; 训练时传 manifest 目录).

    Returns:
        (yaml 文件路径, 合并后的 dict)
    """
    repo_root = repo_root or find_repo_root()
    upstream_yaml = repo_root / _UPSTREAM_YAML_REL
    if not upstream_yaml.is_file():
        raise FileNotFoundError(
            f"上游 BMN yaml 不存在: {upstream_yaml}; "
            "请确认 third_party/PaddleVideo submodule 已 init."
        )

    with upstream_yaml.open("r", encoding="utf-8") as f:
        merged = yaml.safe_load(f)

    # ---- 路径覆盖 ----
    model_cfg = user_cfg.get("model") or {}
    bmn_inputs_dir = model_cfg.get("bmn_inputs_dir")
    if not bmn_inputs_dir:
        # 默认: 跟随 dataset.name
        ds_name = (user_cfg.get("dataset") or {}).get("name") or "pingpong_competition"
        bmn_inputs_dir = repo_root / "data" / "bmn_inputs" / ds_name
    bmn_inputs_dir = Path(bmn_inputs_dir).resolve()

    feature_path = bmn_inputs_dir / "feature"
    label_fixed = bmn_inputs_dir / "label_fixed.json"
    label_gts = bmn_inputs_dir / "label_gts.json"

    if not feature_path.is_dir():
        raise FileNotFoundError(
            f"BMN feature 目录不存在: {feature_path}; "
            "请先运行 scripts/prepare_bmn_inputs.py."
        )
    if not label_fixed.is_file():
        raise FileNotFoundError(
            f"BMN label_fixed.json 不存在: {label_fixed}; "
            "请先运行 scripts/prepare_bmn_inputs.py."
        )

    # PIPELINE.<split>.load_feat.feat_path
    for split in ("train", "valid", "test"):
        sect = merged.get("PIPELINE", {}).get(split)
        if sect and "load_feat" in sect:
            sect["load_feat"]["feat_path"] = str(feature_path)

    # DATASET.<split>.file_path
    for split in ("train", "valid", "test"):
        sect = merged.get("DATASET", {}).get(split)
        if sect:
            sect["file_path"] = str(label_fixed)

    # METRIC paths
    metric = merged.setdefault("METRIC", {})
    metric["file_path"] = str(label_fixed)
    metric["ground_truth_filename"] = str(label_gts)

    # batch_size + epochs
    train_cfg = user_cfg.get("train") or {}
    if "epochs" in train_cfg:
        merged["epochs"] = int(train_cfg["epochs"])
    ds_cfg = user_cfg.get("dataset") or {}
    if "batch_size" in ds_cfg:
        merged.setdefault("DATASET", {})["batch_size"] = int(ds_cfg["batch_size"])

    # output_dir
    out_root = output_dir or Path(tempfile.mkdtemp(prefix="bmn_cfg_"))
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "bmn_pingpong_upstream.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)

    _log.info(
        "BMN upstream config materialized",
        extra={
            "yaml": str(out_path),
            "feat_path": str(feature_path),
            "n_npy": sum(1 for _ in feature_path.glob("*.npy")),
            "epochs": merged.get("epochs"),
        },
    )
    return out_path, merged
