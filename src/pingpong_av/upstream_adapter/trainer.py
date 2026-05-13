"""上游 PaddleVideo 训练 / 评估 / 推理调用适配层 (章程 VI).

本模块是业务代码与上游 PaddleVideo **唯一**的函数级接入点. 我们不复用上游的
``main.py`` 命令行入口 (它是个 argparse 脚本), 而是直接调用上游暴露的 Python API:

- ``paddlevideo.utils.get_config(config_path, overrides=None)`` — 加载上游配置
- ``paddlevideo.tasks.train_model(cfg, weights, parallel, validate)`` — 训练循环
- ``paddlevideo.tasks.test_model(cfg, weights, parallel)`` — 测试循环

这样设计的好处:
1. 业务 CLI (``pp train`` / ``pp eval``) 可以在调用前后统一做 manifest 记录、种子、
   异常捕获、退出码映射, 而不受上游脚本行为的束缚.
2. 升级上游时只需要维护本模块一处的函数签名兼容性.

本文件 (T017) 实现 :func:`run_upstream_train`;
:func:`run_upstream_eval` / :func:`run_upstream_infer` 在 T018 追加.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pingpong_av.upstream_adapter.compat_py311 import apply_runtime_patches
from pingpong_av.upstream_adapter.importer import ensure_paddlevideo_on_path
from pingpong_av.utils.logging import get_logger
from pingpong_av.utils.seeding import set_seed

__all__ = [
    "run_upstream_train",
    "run_upstream_eval",
    "run_upstream_bmn_eval",
    "run_upstream_infer",
    "UpstreamRuntimeError",
]

_log = get_logger(__name__)


def _get_paddlevideo_api() -> tuple[Any, Any, Any]:
    """返回 (get_config, train_model, test_model); 其中 test_model 仅在 T018 使用.

    延迟到调用时才 import, 避免 `pingpong_av` 包导入期强依赖 paddlevideo.
    """
    apply_runtime_patches()
    ensure_paddlevideo_on_path()
    from paddlevideo.tasks import test_model, train_model
    from paddlevideo.utils import get_config

    return get_config, train_model, test_model


def run_upstream_train(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    seed: int,
    resume: str | Path | None = None,
    weights: str | Path | None = None,
    validate: bool = True,
    amp: bool = False,
    overrides: list[str] | None = None,
) -> None:
    """调用上游 PaddleVideo 的 ``train_model`` 执行一次训练.

    参数:
        config_path: 上游格式的训练配置 YAML. 由 ``pingpong_av.models.pp_tsm.load_pp_tsm_config``
                     (T042) 负责从本项目配置生成一份兼容上游的版本.
        output_dir: 本次 run 的根目录 (例如 ``experiments/<run_id>/``). 上游会把
                     日志与 checkpoint 写入 cfg.output_dir 所指向的位置; 本函数通过
                     override 注入该路径.
        seed: 随机种子 (FR-018, 章程 II). 调用前由 :func:`set_seed` 统一注入.
        resume: 从现有 checkpoint 恢复训练 (FR-010). 透传为上游 ``weights`` 参数
                (PaddleVideo 将其作为初始化权重路径, 同时开启优化器状态恢复).
        weights: 初始化权重, 与 resume 的区别: resume 表示"继续同一次训练",
                 weights 表示"从预训练权重初始化冷启动" (FR-012). 二者不能同时指定.
        validate: 训练中是否在验证集上评估 (上游默认 True). 对应 FR-009.
        amp: 是否启用自动混合精度.
        overrides: 额外的配置覆写列表, 形如 ``["train.batch_size=32", ...]``;
                   业务代码通常不应使用此通道, 业务级调参走 ``configs/*.yaml`` (章程 III).

    异常:
        ImportError / UpstreamImportError: 上游不可导; 指引运行 bootstrap.
        RuntimeError / 其他: 上游训练循环内抛出的错误, 原样透传.

    副作用:
        * 调用 :func:`set_seed` 统一设置 random/numpy/paddle 种子.
        * 上游可能向 ``cfg.output_dir`` 写入 checkpoint 与日志.
    """
    if resume is not None and weights is not None:
        raise ValueError("resume 与 weights 不能同时指定: resume 表示继续训练, weights 表示冷启动初始化.")

    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"训练配置文件不存在: {config_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置种子 (章程 II)
    seed_result = set_seed(seed)
    _log.info(
        "seed set before train",
        extra={"seed": seed, "sources_set": seed_result.sources_set,
               "sources_skipped": seed_result.sources_skipped},
    )

    # 透传给上游: 注入 output_dir + 其他必要 override
    all_overrides: list[str] = list(overrides or [])
    all_overrides.append(f"output_dir={output_dir}")

    get_config, train_model, _ = _get_paddlevideo_api()

    cfg = get_config(str(config_path), overrides=all_overrides, show=False) \
        if _accepts_show_kw(get_config) else get_config(str(config_path), overrides=all_overrides)

    # 判断并行模式 — 本项目 MVP 默认单 GPU, 不走分布式初始化
    import paddle
    parallel = False
    world_size = 1
    try:
        from paddlevideo.utils import get_dist_info
        _, world_size = get_dist_info()
        parallel = world_size != 1
        if parallel:
            paddle.distributed.init_parallel_env()
    except ImportError:
        # 某些上游版本没有 get_dist_info; 按单卡处理
        pass

    # 关于 resume vs weights 的传参: 上游 train_model 只有 `weights` 参数,
    # 传 resume 路径进去由上游自行根据 cfg 中的 `resume_epoch` 等决定是否恢复优化器状态.
    effective_weights: str | None = None
    if resume is not None:
        effective_weights = str(Path(resume).resolve())
        _log.info("resuming training from checkpoint", extra={"resume": effective_weights})
    elif weights is not None:
        effective_weights = str(Path(weights).resolve())
        _log.info("initializing training from pretrained weights", extra={"weights": effective_weights})

    _log.info(
        "calling upstream train_model",
        extra={"config": str(config_path), "output_dir": str(output_dir),
               "world_size": world_size, "parallel": parallel, "validate": validate, "amp": amp},
    )

    train_kwargs: dict[str, Any] = {
        "weights": effective_weights,
        "parallel": parallel,
        "validate": validate,
    }
    # amp 参数是较新上游才加入的; 用 inspect 保险地注入
    if _has_kwarg(train_model, "amp"):
        train_kwargs["amp"] = amp

    train_model(cfg, **train_kwargs)
    _log.info("upstream train_model returned cleanly")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _has_kwarg(fn: Any, name: str) -> bool:
    """上游函数签名在不同 release 间有变动; 此 helper 保险地检测关键字是否被接受."""
    try:
        import inspect

        sig = inspect.signature(fn)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


def _accepts_show_kw(fn: Any) -> bool:
    """上游 get_config 某些版本支持 ``show=False`` 静默加载, 老版本不支持."""
    return _has_kwarg(fn, "show")


# --------------------------------------------------------------------------------------
# T018: 评估 + 单片段推理适配
# --------------------------------------------------------------------------------------


class UpstreamRuntimeError(RuntimeError):
    """上游 PaddleVideo 在运行时抛出的错误, 由适配层统一包装."""


def _load_upstream_config(config_path: str | Path, overrides: list[str] | None = None) -> Any:
    """薄封装, 隔离 get_config 的版本差异."""
    apply_runtime_patches()
    ensure_paddlevideo_on_path()
    from paddlevideo.utils import get_config  # 延迟 import

    overrides = list(overrides or [])
    if _accepts_show_kw(get_config):
        return get_config(str(config_path), overrides=overrides, show=False)
    return get_config(str(config_path), overrides=overrides)


def run_upstream_eval(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    split_file: str | Path | None = None,
    batch_size: int | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """对给定 checkpoint 在指定 split 上做前向, 返回 logits + labels.

    与直接调用上游 ``test_model`` 的**关键区别**: 本函数**不**复用上游的 Metric 机制
    (上游 Metric 只打印, 不返回数值, 且其 schema 因模型而异). 取而代之, 我们手动
    跑 data_loader + model.forward 的 eval 循环, 把每个 batch 的输出拼接成 ``logits``
    与 ``labels`` 返回; 指标计算由 ``pingpong_av.evaluation.metrics`` (T047) 独立完成.
    这样保证章程 V 要求的 top1/top5/per-class/macro-avg 输出 schema 是本项目自己的, 不
    受上游 Metric 变动影响.

    参数:
        config_path: 与训练时同一份配置 (通常从 ``experiments/<run>/config.yaml`` 读取).
        checkpoint: ``.pdparams`` 路径.
        split_file: 如提供, 通过 override 把 ``DATASET.test.file_path`` 指向该文件;
                    允许 ``pp eval`` 使用 ``data/splits/test.txt`` 而非 cfg 默认值.
        batch_size: 如提供, override ``DATASET.test_batch_size``.
        overrides: 额外 override.

    返回:
        ``{"logits": np.ndarray[N, C], "labels": np.ndarray[N], "class_names": list[str] | None}``

    抛出:
        :class:`UpstreamRuntimeError`: 上游 model / data 构建或前向出错.
    """
    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置不存在: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    all_overrides: list[str] = list(overrides or [])
    if split_file is not None:
        all_overrides.append(f"DATASET.test.file_path={Path(split_file).resolve()}")
    if batch_size is not None:
        all_overrides.append(f"DATASET.test_batch_size={int(batch_size)}")

    cfg = _load_upstream_config(config_path, overrides=all_overrides)

    import numpy as np
    import paddle
    from paddle.io import DataLoader  # noqa: F401 (some upstream versions use custom wrapper)

    try:
        from paddlevideo.loader.builder import build_dataloader, build_dataset
        from paddlevideo.modeling.builder import build_model
        from paddlevideo.utils import load
    except ImportError as exc:
        raise UpstreamRuntimeError(
            f"无法导入上游 paddlevideo 必需的子模块: {exc}. "
            "请确认 submodule 已 init, bootstrap.sh 已成功完成 editable 安装."
        ) from exc

    # 1. 构建模型并加载权重
    if getattr(cfg.MODEL.backbone, "pretrained", None):
        cfg.MODEL.backbone.pretrained = ""  # 禁用 pretrain 初始化, 以 checkpoint 为准
    model = build_model(cfg.MODEL)
    model.eval()

    state_dicts = load(str(checkpoint))
    model.set_state_dict(state_dicts)
    _log.info("checkpoint loaded", extra={"checkpoint": str(checkpoint)})

    # 2. 构建 dataset + dataloader
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    bsz = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device("gpu" if paddle.device.cuda.device_count() > 0 else "cpu")
    dataloader_setting = dict(
        batch_size=bsz,
        num_workers=cfg.DATASET.get("test_num_workers", cfg.DATASET.get("num_workers", 0)),
        places=places,
        drop_last=False,
        shuffle=False,
    )
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # 3. 前向循环, 收集 logits + labels
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    n_batches = 0
    try:
        with paddle.no_grad():
            for batch_id, data in enumerate(data_loader):
                outputs = model(data, mode="test")
                # PaddleVideo 的 model(..., mode='test') 约定: 返回预测 logits (可能是 softmax 概率)
                # 形状 [N, C]. labels 通常在 data[-1] 或 data[1], 取决于 pipeline.
                logits = _to_numpy(outputs)
                label = _extract_labels(data)
                logits_chunks.append(logits)
                labels_chunks.append(label)
                n_batches += 1
    except Exception as exc:
        raise UpstreamRuntimeError(
            f"上游评估前向循环第 {n_batches} 个 batch 后出错: {exc}"
        ) from exc

    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 0))
    labels = np.concatenate(labels_chunks, axis=0) if labels_chunks else np.empty((0,), dtype=np.int64)

    class_names: list[str] | None = None
    # 优先从 cfg 的 head 或 dataset 元信息读类别名; 没有则由上层从本项目 dataset yaml 拿
    head_cfg = getattr(cfg.MODEL, "head", None)
    if head_cfg is not None and "class_names" in head_cfg:
        class_names = list(head_cfg["class_names"])

    _log.info(
        "eval forward done",
        extra={"n_samples": int(labels.shape[0]), "n_classes": int(logits.shape[1]) if logits.size else 0},
    )
    return {"logits": logits, "labels": labels, "class_names": class_names}


def run_upstream_bmn_eval(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    result_path: Path | None = None,
    output_path: Path | None = None,
    label_gts_path: str | Path | None = None,
    subset: str = "validation",
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """BMN 时序定位的评估循环.

    与 :func:`run_upstream_eval` 的关键差异:
        - PP-TSM 输出 logits[N, C]; BMN 输出 (start_score, end_score, confidence_map) 三组张量,
          经过 NMS-like 后处理后才能得到 proposal 列表. 上游 ``BMNMetric.accumulate()`` 已经做了
          这一切, 并把 ActivityNet 1.3 风格的结果写到 ``cfg.METRIC.result_path/bmn_results_<subset>.json``.
        - 上游 Metric 的指标 (AR@1/5/10/100) 仅通过 ``logger.info`` 输出, 没有返回值.
          本函数在调用上游 ``test_model`` 后, 主动**再调用一次 ``cal_metrics``** 拿到 numpy 数组,
          以便上层 cli 写入结构化 metrics.json.

    参数:
        config_path: 训练时的 upstream_config.yaml snapshot (或 BMN 数据集 yaml 走完 load_bmn_config
                     的产出).
        checkpoint: ``.pdparams`` 路径.
        result_path: 若指定, 覆盖 cfg.METRIC.result_path (此目录会被 BMNMetric 写入
                     ``bmn_results_<subset>.json``); 推荐传入 ``<run_dir>/bmn_eval/``.
        output_path: 若指定, 覆盖 cfg.METRIC.output_path (BMN per-video 候选区间的中间产物).
        label_gts_path: 若指定, 覆盖 cfg.METRIC.ground_truth_filename.
        subset: BMN Metric 的 subset 标签 (默认 'validation', 与 prepare_bmn_inputs.py 一致).

    返回:
        {
          "ar@1":   float,
          "ar@5":   float,
          "ar@10":  float,
          "ar@100": float,
          "average_nr_proposals": ndarray (sorted),
          "average_recall":       ndarray,
          "recall_per_tiou":      ndarray (10 tIoU thresholds × len(nr_proposals)),
          "result_path":          str,
          "bmn_results_json":     str,
          "n_videos_evaluated":   int,
          "n_proposals":          int,
        }
    """
    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置不存在: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    cfg = _load_upstream_config(config_path, overrides=[])

    # 覆盖 METRIC 字段, 让 BMNMetric 把结果落到我们指定的目录
    metric = cfg.setdefault("METRIC", {})
    if subset:
        metric["subset"] = subset
    if result_path is not None:
        result_path = Path(result_path).resolve()
        result_path.mkdir(parents=True, exist_ok=True)
        metric["result_path"] = str(result_path)
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        metric["output_path"] = str(output_path)
    if label_gts_path is not None:
        metric["ground_truth_filename"] = str(Path(label_gts_path).resolve())

    # 上游 test_model 通常要求 cfg.MODEL.framework == 'BMNLocalizer'; 我们这里假设 cfg 已对齐 (训练 + eval 同源)
    try:
        _, _, test_model = _get_paddlevideo_api()
        # 注意: 上游 BMNMetric 通过 @METRIC.register 装饰, 该装饰器返回 None,
        # 导致 ``paddlevideo.metrics.bmn_metric.BMNMetric`` 模块属性值为 None.
        # 必须经 registry 拿真实类.
        import paddlevideo.metrics.bmn_metric  # noqa: F401 -- trigger registration side effect
        from paddlevideo.metrics.registry import METRIC as _METRIC
        BMNMetric = _METRIC.get("BMNMetric")
        if BMNMetric is None:
            raise UpstreamRuntimeError("METRIC registry 中没有 'BMNMetric'; 上游版本异常.")
    except ImportError as exc:
        raise UpstreamRuntimeError(
            f"无法导入上游 BMN 测试 API: {exc}; 请确认 patches 已应用、bootstrap 完成."
        ) from exc

    _log.info(
        "BMN test_model start",
        extra={
            "checkpoint": str(checkpoint),
            "subset": metric.get("subset"),
            "result_path": metric.get("result_path"),
        },
    )

    # 预创建上游 anet_prop.py:167 硬编码的 verbose=True 输出目录, 避免
    # FileNotFoundError: 'data/bmn/BMN_Test_results/auc_result.txt'
    try:
        from pingpong_av.utils.env import find_repo_root as _frr
        _auc_dir = _frr() / "data" / "bmn" / "BMN_Test_results"
    except Exception:
        _auc_dir = Path("data/bmn/BMN_Test_results")
    _auc_dir.mkdir(parents=True, exist_ok=True)

    bmn_results_json = Path(metric["result_path"]) / f"bmn_results_{metric.get('subset','validation')}.json"

    if reuse_existing and bmn_results_json.is_file():
        _log.info(
            "BMN test_model skipped (reuse_existing=True; using cached predictions)",
            extra={"bmn_results_json": str(bmn_results_json)},
        )
    else:
        try:
            test_model(cfg, weights=str(checkpoint), parallel=False)
        except Exception as exc:
            raise UpstreamRuntimeError(
                f"上游 BMN test_model 失败: {type(exc).__name__}: {exc}"
            ) from exc

    # 上游已经把 ActivityNet 1.3 风格 JSON 写到 result_path; 重新调用 cal_metrics 拿数值
    if not bmn_results_json.is_file():
        raise UpstreamRuntimeError(
            f"上游 BMNMetric 未写出预期 JSON: {bmn_results_json}. "
            "可能是后处理多进程 crash; 请查看上一层 stdout/stderr."
        )

    import numpy as np
    # 重新跑 cal_metrics (轻量, 只读 JSON, 不需要 GPU); 用与上游一致的 tiou_thresholds
    # BMNMetric 是 BaseMetric 子类, 实例化时需要 file_path / ground_truth_filename / subset / output_path /
    # result_path 等; 我们重新构造一个轻量实例只为 cal_metrics
    try:
        bmn_metric = BMNMetric(
            data_size=0,
            batch_size=1,
            tscale=int(metric.get("tscale", 200)),
            dscale=int(metric.get("dscale", 200)),
            file_path=str(metric["file_path"]),
            ground_truth_filename=str(metric["ground_truth_filename"]),
            subset=metric.get("subset", "validation"),
            output_path=str(metric["output_path"]),
            result_path=str(metric["result_path"]),
        )
    except Exception as exc:
        raise UpstreamRuntimeError(f"无法重建 BMNMetric 实例: {exc}") from exc

    try:
        avg_nr_proposals, avg_recall, recall = bmn_metric.cal_metrics(
            str(metric["ground_truth_filename"]),
            str(bmn_results_json),
            max_avg_nr_proposals=100,
            tiou_thresholds=np.linspace(0.5, 0.95, 10),
            subset=metric.get("subset", "validation"),
        )
    except FileNotFoundError as exc:
        # 上游 anet_prop.py:167 在 verbose=True 时硬编码写 'data/bmn/BMN_Test_results/auc_result.txt';
        # 该路径不存在时抛 FileNotFoundError. 我们事先创建一次然后重试 (与 patch 替代).
        if "auc_result.txt" in str(exc) or "BMN_Test_results" in str(exc):
            from pingpong_av.utils.env import find_repo_root as _frr
            try:
                fallback = _frr() / "data" / "bmn" / "BMN_Test_results"
            except Exception:
                fallback = Path("data/bmn/BMN_Test_results")
            fallback.mkdir(parents=True, exist_ok=True)
            _log.warning(
                "BMN cal_metrics: created upstream-hardcoded dir to satisfy verbose=True path",
                extra={"dir": str(fallback)},
            )
            avg_nr_proposals, avg_recall, recall = bmn_metric.cal_metrics(
                str(metric["ground_truth_filename"]),
                str(bmn_results_json),
                max_avg_nr_proposals=100,
                tiou_thresholds=np.linspace(0.5, 0.95, 10),
                subset=metric.get("subset", "validation"),
            )
        else:
            raise UpstreamRuntimeError(f"BMNMetric.cal_metrics 失败: {exc}") from exc
    except Exception as exc:
        raise UpstreamRuntimeError(f"BMNMetric.cal_metrics 失败: {exc}") from exc

    # avg_recall 是按 max_avg_nr_proposals=100 输出, 形状 [100,], 索引 0/4/9/99 即 AR@1/5/10/100
    def _ar_at(idx: int) -> float:
        try:
            return float(np.mean(recall[:, idx])) * 100.0
        except (IndexError, ValueError):
            return float("nan")

    # 计 count
    import json as _json
    with bmn_results_json.open("r", encoding="utf-8") as f:
        bmn_json = _json.load(f)
    n_videos = len(bmn_json.get("results", {}))
    n_proposals = sum(len(v) for v in bmn_json.get("results", {}).values())

    out = {
        "ar@1": _ar_at(0),
        "ar@5": _ar_at(4),
        "ar@10": _ar_at(9),
        "ar@100": _ar_at(-1),
        "average_nr_proposals": avg_nr_proposals.tolist() if hasattr(avg_nr_proposals, "tolist") else list(avg_nr_proposals),
        "average_recall":       avg_recall.tolist()       if hasattr(avg_recall, "tolist")       else list(avg_recall),
        "result_path":     str(metric["result_path"]),
        "bmn_results_json": str(bmn_results_json),
        "n_videos_evaluated": n_videos,
        "n_proposals":        n_proposals,
        "subset":             metric.get("subset", "validation"),
    }
    _log.info(
        "BMN eval done",
        extra={"ar@1": out["ar@1"], "ar@100": out["ar@100"],
               "n_videos": n_videos, "n_proposals": n_proposals},
    )
    return out


def run_upstream_infer(
    config_path: str | Path,
    checkpoint: str | Path,
    video_path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> "np.ndarray":  # noqa: F821
    """对单个视频片段做一次推理, 返回一个 softmax / logits 向量 ``[num_classes]``.

    内部通过构造一个只含单样本的临时 file_list, 复用上游的 test pipeline 来采样帧与预处理,
    这样可以确保与训练/评估一致的输入分布. 输出保持 **概率形式** (若模型 head 已做 softmax),
    否则调用方应用 softmax.

    使用场景: ``pp infer-clip`` 与 ``pp infer-video`` (滑窗每个窗口).

    异常:
        * 视频不可读 → FileNotFoundError
        * 推理过程失败 → :class:`UpstreamRuntimeError`
    """
    import tempfile
    import numpy as np
    import paddle

    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 构造单行 list 文件 (label 占位 0, 不参与推理结果)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(f"{video_path}\t0\n")
        tmp_path = tmp.name

    try:
        result = run_upstream_eval(
            config_path, checkpoint,
            split_file=tmp_path, batch_size=1,
            overrides=overrides,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logits = result["logits"]
    if logits.size == 0 or logits.shape[0] == 0:
        raise UpstreamRuntimeError(f"单视频推理没有产出任何 logits, 视频: {video_path}")
    return logits[0]  # [num_classes]


# --------------------------------------------------------------------------------------
# low-level helpers for T018
# --------------------------------------------------------------------------------------


def _to_numpy(outputs: Any) -> "np.ndarray":  # noqa: F821
    """把 paddle.Tensor / tuple / dict 的输出标准化为 numpy [N, C]."""
    import numpy as np
    import paddle

    if isinstance(outputs, paddle.Tensor):
        return outputs.numpy()
    if isinstance(outputs, (list, tuple)):
        # 约定: 第 0 项是 logits
        return _to_numpy(outputs[0])
    if isinstance(outputs, dict):
        for key in ("logits", "prob", "output"):
            if key in outputs:
                return _to_numpy(outputs[key])
    if isinstance(outputs, np.ndarray):
        return outputs
    raise UpstreamRuntimeError(
        f"无法将上游输出转为 numpy: 类型 {type(outputs).__name__}. "
        "可能是上游签名变化, 需要在 _to_numpy 中适配."
    )


def _extract_labels(batch: Any) -> "np.ndarray":  # noqa: F821
    """从 PaddleVideo 的 batch 数据中提取 label 张量为 numpy [N]."""
    import numpy as np
    import paddle

    if isinstance(batch, (list, tuple)):
        # PaddleVideo 多数 pipeline 将 label 放在 batch[-1] 或 batch[1]
        candidates = [batch[-1]]
        if len(batch) >= 2:
            candidates.append(batch[1])
        for c in candidates:
            if isinstance(c, paddle.Tensor) and c.dtype in (paddle.int64, paddle.int32):
                return c.numpy().reshape(-1)
    if isinstance(batch, dict) and "label" in batch:
        lab = batch["label"]
        if isinstance(lab, paddle.Tensor):
            return lab.numpy().reshape(-1)
    raise UpstreamRuntimeError(
        f"无法从 batch 中提取 label; batch 类型 {type(batch).__name__}. "
        "可能需要在 _extract_labels 中适配新的 pipeline 格式."
    )
