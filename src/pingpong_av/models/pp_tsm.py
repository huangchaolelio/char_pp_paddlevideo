"""PP-TSM 配置薄封装: 把本项目业务视角的 YAML 转换为上游 PaddleVideo 原生格式 (章程 VI).

设计目标:
- 业务侧 (``configs/models/pp_tsm_pingpong.yaml``) 用对人友好的字段名 (model.name,
  train.epochs 等), 由本模块转换为上游需要的大写嵌套结构 (MODEL/DATASET/PIPELINE/
  OPTIMIZER 等).
- **不**让业务代码依赖上游 YAML 的具体 key 命名习惯, 升级上游时改一处.
- 类别数 ``num_classes`` 自动从 ``classes`` 字段长度注入, 避免两处维护.

单一公开入口:
    :func:`load_pp_tsm_config(user_cfg, dataset_cfg=None) -> Path`
    返回**写到磁盘**的上游格式 YAML 路径 (放在临时位置), 由 :mod:`upstream_adapter.trainer`
    传给上游 ``get_config``.

不在本模块的范围:
- Kinetics 预训练权重的下载 (走上游本身的 url 解析).
- 训练/评估循环 (那是 ``upstream_adapter.trainer``).
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from pingpong_av.utils.config import ConfigError
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

__all__ = ["load_pp_tsm_config", "build_upstream_dict", "PPTSMConfigError"]

_log = get_logger(__name__)

# 上游 release/2.2.0 中本项目使用的"视频格式 PP-TSM" 模板. T042 不复制其内容,
# 而是在生成时**读取它作为基线**, 再用本项目字段做 override.
_UPSTREAM_BASE_RELPATH = Path(
    "third_party/PaddleVideo/configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml"
)


class PPTSMConfigError(ConfigError):
    """PP-TSM 配置合并失败."""


# --------------------------------------------------------------------------------------
# 公共入口
# --------------------------------------------------------------------------------------


def load_pp_tsm_config(
    user_cfg: Mapping[str, Any],
    *,
    dataset_cfg: Mapping[str, Any] | None = None,
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    repo_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """合并本项目业务配置与上游 PP-TSM 模板, 写出上游格式 YAML.

    参数:
        user_cfg: 通过 :func:`pingpong_av.utils.config.load_config` 加载的业务配置 (含
                  ``classes``, ``model``, ``train``, ``eval``, ``pipeline``, ``seed`` 等).
        dataset_cfg: 可选的独立 dataset 配置. 若提供, 则与 ``user_cfg`` 中的 classes
                     做一致性校验 (章程 III: 防漂移).
        splits_dir: ``data/splits/`` 目录; train.txt / val.txt / test.txt 的位置. 默认
                    通过 :func:`find_repo_root` 推断.
        output_dir: 训练产物 (checkpoint/log) 目录. 通常是 ``experiments/<run_id>/``.
                    会被注入上游的 ``output_dir`` 字段.
        repo_root: 仓库根. 默认自动查找.

    返回:
        ``(merged_yaml_path, merged_dict)``. merged_yaml_path 指向一个临时 YAML 文件
        (调用方负责清理); merged_dict 是写入文件的相同内容, 便于上层 manifest snapshot.

    抛出:
        :class:`PPTSMConfigError`: 配置不兼容 / 上游模板缺失 / 类别表不一致.
    """
    if user_cfg.get("model", {}).get("name") != "pp_tsm":
        raise PPTSMConfigError(
            f"load_pp_tsm_config 只接受 model.name='pp_tsm', 实际为 "
            f"{user_cfg.get('model', {}).get('name')!r}"
        )

    classes = user_cfg.get("classes")
    if not classes:
        raise PPTSMConfigError("user_cfg 缺少 classes (章程 III)")
    num_classes = len(classes)

    # 类别表一致性校验 (章程 III: 防漂移)
    if dataset_cfg is not None:
        ds_classes = dataset_cfg.get("classes")
        if ds_classes:
            _ensure_classes_match(classes, ds_classes)

    repo_root = repo_root or find_repo_root()
    upstream_base = repo_root / _UPSTREAM_BASE_RELPATH
    if not upstream_base.is_file():
        raise PPTSMConfigError(
            f"上游 PP-TSM 模板不存在: {upstream_base}; "
            "请确认 submodule 已 init."
        )

    # 1) 读上游模板作为基线
    base = yaml.safe_load(upstream_base.read_text(encoding="utf-8"))

    # 2) 用本项目字段做 override
    merged = build_upstream_dict(
        base=base,
        user_cfg=user_cfg,
        num_classes=num_classes,
        splits_dir=splits_dir,
        output_dir=output_dir,
        repo_root=repo_root,
    )

    # 3) 写到临时位置
    tmp_dir = Path(tempfile.mkdtemp(prefix="pp_tsm_cfg_"))
    out_path = tmp_dir / "pp_tsm_pingpong_upstream.yaml"
    out_path.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    _log.info(
        "PP-TSM upstream config materialized",
        extra={"path": str(out_path), "num_classes": num_classes},
    )
    return out_path, merged


# --------------------------------------------------------------------------------------
# 合并逻辑 (无副作用, 便于单元测试)
# --------------------------------------------------------------------------------------


def build_upstream_dict(
    *,
    base: dict[str, Any],
    user_cfg: Mapping[str, Any],
    num_classes: int,
    splits_dir: str | Path | None,
    output_dir: str | Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    """把 base (上游模板) + user_cfg 合并成上游格式 dict.

    覆盖规则 (按字段优先级):
        - 类别数: user_cfg.classes 长度 → MODEL.head.num_classes
        - 帧数:   user_cfg.pipeline.num_segments → PIPELINE.{train,valid,test}.sample.num_seg
        - 输入:   user_cfg.pipeline.{image_size, short_side, mean, std} → PIPELINE.*.transform
        - batch:  user_cfg.train.batch_size → DATASET.batch_size
                  user_cfg.eval.test_batch_size → DATASET.test_batch_size
        - 优化器: user_cfg.train.optimizer.* + lr.* → OPTIMIZER.*
        - epoch:  user_cfg.train.epochs → epochs
        - 数据路径: splits_dir → DATASET.{train,valid,test}.file_path
                                + DATASET.*.data_prefix = 空 (paths 是绝对路径写在 list 里)
        - output_dir: → 顶层 output_dir
        - log_interval: user_cfg.logging.log_interval → log_interval
    """
    merged = copy.deepcopy(base)

    # ---- MODEL ----
    head = merged.setdefault("MODEL", {}).setdefault("head", {})
    head["num_classes"] = num_classes
    if "drop_ratio" in head and "head" in user_cfg.get("model", {}):
        # user_cfg.model.head 可选覆盖
        head_user = user_cfg["model"].get("head", {}) or {}
        if "dropout_ratio" in head_user:
            head["drop_ratio"] = float(head_user["dropout_ratio"])

    backbone = merged["MODEL"].setdefault("backbone", {})
    backbone_user = user_cfg.get("model", {}).get("backbone", {}) or {}
    if "depth" in backbone_user:
        backbone["depth"] = int(backbone_user["depth"])
    if "pretrained" in backbone_user and backbone_user["pretrained"]:
        backbone["pretrained"] = str(backbone_user["pretrained"])

    # ---- PIPELINE: num_segments / image_size / short_side / mean / std ----
    pipe = user_cfg.get("pipeline", {}) or {}
    num_segments = int(pipe.get("num_segments", 8))
    image_size = int(pipe.get("image_size", 224))
    short_side = int(pipe.get("short_side", 256))
    mean = list(pipe.get("mean", [0.485, 0.456, 0.406]))
    std = list(pipe.get("std", [0.229, 0.224, 0.225]))

    for split_key in ("train", "valid", "test"):
        sect = merged.get("PIPELINE", {}).get(split_key)
        if not sect:
            continue
        if "sample" in sect:
            sect["sample"]["num_seg"] = num_segments
        # 强制把 decode backend 切换为 'cv2' (Python 3.11 兼容).
        # 上游模板默认是 'decord', 但 decord 0.4.x 没有 3.11 wheel; 见
        # third_party/patches/02-decord-lazy-import-py311.patch 与 research.md R3.
        if "decode" in sect and isinstance(sect["decode"], dict):
            sect["decode"]["backend"] = "cv2"
        # 修改 transform 数组中的 Scale/CenterCrop/MultiScaleCrop/RandomCrop/Normalization
        for op in sect.get("transform", []):
            if not isinstance(op, dict):
                continue
            for op_name, op_args in op.items():
                if op_args is None:
                    continue
                if op_name == "Scale" and "short_size" in op_args:
                    op_args["short_size"] = short_side
                elif op_name in ("MultiScaleCrop", "CenterCrop", "RandomCrop") and "target_size" in op_args:
                    # MultiScaleCrop 用 short_side, 其他用 image_size
                    if op_name == "MultiScaleCrop":
                        op_args["target_size"] = short_side
                    else:
                        op_args["target_size"] = image_size
                elif op_name == "Normalization":
                    op_args["mean"] = mean
                    op_args["std"] = std

    # ---- DATASET ----
    train = user_cfg.get("train", {}) or {}
    eval_cfg = user_cfg.get("eval", {}) or {}
    ds = merged.setdefault("DATASET", {})
    if "batch_size" in train:
        ds["batch_size"] = int(train["batch_size"])
    if "num_workers" in train:
        ds["num_workers"] = int(train["num_workers"])
    if "test_batch_size" in eval_cfg:
        ds["test_batch_size"] = int(eval_cfg["test_batch_size"])

    # 切换到 list 文件 (file_path) — splits_dir 必须存在或后续 data-prepare 会写入
    splits_dir_p = Path(splits_dir) if splits_dir else (repo_root / "data" / "splits")
    splits_dir_p = splits_dir_p.resolve()

    for split_name, list_basename in (("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")):
        sub = ds.setdefault(split_name, {})
        sub["format"] = "VideoDataset"
        sub["file_path"] = str(splits_dir_p / list_basename)
        # data_prefix 留空: list 文件中已写绝对路径, 上游不需要再拼前缀
        sub["data_prefix"] = ""

    # ---- OPTIMIZER ----
    opt_user = train.get("optimizer", {}) or {}
    lr_user = train.get("lr", {}) or {}
    opt = merged.setdefault("OPTIMIZER", {})
    if "name" in opt_user:
        opt["name"] = opt_user["name"]
    if "momentum" in opt_user:
        opt["momentum"] = float(opt_user["momentum"])
    if "weight_decay" in opt_user:
        opt.setdefault("weight_decay", {})
        opt["weight_decay"]["name"] = "L2"
        opt["weight_decay"]["value"] = float(opt_user["weight_decay"])

    if lr_user:
        sub_lr = opt.setdefault("learning_rate", {})
        sub_lr["iter_step"] = bool(sub_lr.get("iter_step", True))
        if lr_user.get("scheduler") == "CosineAnnealing":
            sub_lr["name"] = "CustomWarmupCosineDecay"
        sub_lr["max_epoch"] = int(train.get("epochs", sub_lr.get("max_epoch", 80)))
        if "warmup_epochs" in lr_user:
            sub_lr["warmup_epochs"] = int(lr_user["warmup_epochs"])
        if "warmup_start_lr" in lr_user:
            sub_lr["warmup_start_lr"] = float(lr_user["warmup_start_lr"])
        if "base_lr" in lr_user:
            sub_lr["cosine_base_lr"] = float(lr_user["base_lr"])

    # ---- 顶层 ----
    if "epochs" in train:
        merged["epochs"] = int(train["epochs"])
    log = user_cfg.get("logging", {}) or {}
    if "log_interval" in log:
        merged["log_interval"] = int(log["log_interval"])

    if output_dir is not None:
        merged["output_dir"] = str(Path(output_dir).resolve())

    # PreciseBN 不适用于小数据集; 关闭以免内存压力
    merged.pop("PRECISEBN", None)

    return merged


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _ensure_classes_match(model_classes: list, dataset_classes: list) -> None:
    """章程 III 一致性闸门: 模型配置和数据集配置中的 classes 必须等价.

    "等价"定义: 同样的长度, 同样的 (id, name) 对集合.
    display_name 与 description 等额外字段可以不同.
    """
    def _key(c: dict) -> tuple[int, str]:
        return (int(c["id"]), str(c["name"]))

    a = sorted(_key(c) for c in model_classes)
    b = sorted(_key(c) for c in dataset_classes)
    if a != b:
        diff_a = [x for x in a if x not in b]
        diff_b = [x for x in b if x not in a]
        raise PPTSMConfigError(
            "model 配置与 dataset 配置的 classes 不一致 (章程 III 防漂移):\n"
            f"  model 独有: {diff_a}\n"
            f"  dataset 独有: {diff_b}\n"
            "请同步两份配置, 或重新派生 model 配置."
        )
