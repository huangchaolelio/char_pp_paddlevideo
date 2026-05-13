"""``pp eval`` 子命令 (FR-011, 章程 IV/V).

编排:
    1. 校验 ``--checkpoint`` 文件存在, 自动从同 run 目录回找 ``config.yaml`` (snapshot);
    2. 章程 IV 闸门: ``--split=test`` 且 metrics.json 已存在 → 必须 ``--rerun``, 否则退出 3;
    3. 调用 :func:`upstream_adapter.trainer.run_upstream_eval` 跑前向得到 logits + labels;
    4. 用 ``evaluation.reporter`` 计算指标 + 写 metrics.json + 渲染混淆矩阵;
    5. 把摘要追加到 ``manifest.json`` 的 ``metrics_summary``;
    6. stdout 输出结构化摘要.

退出码:
    0  成功
    1  用户输入错 (checkpoint / split / output 非法)
    2  环境问题 (上游不可导)
    3  章程硬约束违反 (test 集重复评估未加 --rerun)
    4  运行时失败 (上游评估异常)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from pingpong_av.evaluation.reporter import (
    build_metrics_payload,
    render_confusion_matrix,
    write_metrics_json,
)
from pingpong_av.experiment.run_manifest import ConstitutionViolation, finalize
from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(
    *,
    checkpoint: str,
    split: str,
    batch_size: int | None,
    output_path: str | None,
    rerun: bool,
) -> int:
    """执行 eval. 返回应当作为进程退出码使用的整数."""
    if split not in ("test", "val"):
        click.echo(f"ERROR: --split 只允许 test 或 val, 实际 {split!r}", err=True)
        return 1

    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        click.echo(f"ERROR: checkpoint 不存在: {ckpt}", err=True)
        return 1

    # ---- 1. 自动找 run_dir 与 config.yaml snapshot ----
    run_dir = _find_run_dir(ckpt)
    if run_dir is None:
        click.echo(
            f"ERROR: 无法从 checkpoint 路径定位 experiments/<run_id>/ 目录: {ckpt}\n"
            "       eval 需要从该目录读取 config.yaml (训练时的 snapshot) 以重建模型结构.",
            err=True,
        )
        return 1
    snapshot_yaml = run_dir / "config.yaml"
    if not snapshot_yaml.is_file():
        click.echo(f"ERROR: config snapshot 缺失: {snapshot_yaml}", err=True)
        return 1

    upstream_yaml = run_dir / "upstream_config.yaml"
    if not upstream_yaml.is_file():
        click.echo(f"ERROR: upstream config snapshot 缺失: {upstream_yaml}", err=True)
        return 1

    # ---- 2. 加载业务 config (用于 class_names) ----
    try:
        loaded = load_config(snapshot_yaml)
    except ConfigError as exc:
        click.echo(f"ERROR: snapshot 配置异常: {exc}", err=True)
        return 1

    classes_meta = loaded.data["classes"]
    class_names = [c["name"] for c in classes_meta]

    # ---- 2a. 模型路径分支: BMN 时序定位走独立流程 ----
    model_name = (loaded.data.get("model") or {}).get("name", "")
    if model_name == "bmn":
        return _run_bmn_eval(
            run_dir=run_dir,
            ckpt=ckpt,
            split=split,
            upstream_yaml=upstream_yaml,
            classes_meta=classes_meta,
            output_path=Path(output_path).resolve() if output_path else (run_dir / "metrics.json"),
            rerun=rerun,
        )

    # ---- 3. 章程 IV 闸门: split=test 重复评估必须 --rerun ----
    out_metrics = Path(output_path).resolve() if output_path else (run_dir / "metrics.json")
    if split == "test" and out_metrics.exists():
        try:
            existing = json.loads(out_metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("split") == "test" and not rerun:
            click.echo(
                "ERROR (章程 IV 违反): 检测到该 run 已有 split=test 的 metrics.json:\n"
                f"  {out_metrics}\n"
                "测试集禁止反复挑选结果. 如确需重跑, 追加 `--rerun` 标志 (并在 PR 描述中说明原因).",
                err=True,
            )
            return 3

    # ---- 4. 把 split 映射到上游 file_path; 找出对应 list 文件 ----
    splits_dir = _read_splits_dir_from_upstream(upstream_yaml)
    list_basename = "test.txt" if split == "test" else "val.txt"
    split_file = splits_dir / list_basename
    if not split_file.is_file():
        click.echo(f"ERROR: split 文件不存在: {split_file}", err=True)
        return 1

    # ---- 5. 调用上游评估 ----
    click.echo(f"[eval] split={split}  checkpoint={ckpt.name}  run_id={run_dir.name}", err=True)

    try:
        from pingpong_av.upstream_adapter.trainer import (
            UpstreamRuntimeError,
            run_upstream_eval,
        )
    except ImportError as exc:
        click.echo(f"ERROR: 无法 import upstream_adapter: {exc}", err=True)
        return 2

    try:
        result = run_upstream_eval(
            upstream_yaml,
            ckpt,
            split_file=split_file,
            batch_size=batch_size,
        )
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        return 1
    except UpstreamRuntimeError as exc:
        click.echo(f"ERROR: 上游评估失败: {exc}", err=True)
        return 4
    except Exception as exc:
        click.echo(f"ERROR: 上游评估时出现意外错误: {type(exc).__name__}: {exc}", err=True)
        return 4

    logits = result["logits"]
    labels = result["labels"]
    if logits.size == 0:
        click.echo("ERROR: 上游评估返回 0 个样本.", err=True)
        return 4

    # ---- 6. 计算指标 + 写文件 ----
    payload = build_metrics_payload(
        logits=logits,
        labels=labels,
        class_names=class_names,
        checkpoint=ckpt,
        split=split,
        topk=tuple(loaded.data.get("eval", {}).get("topk", [1, 5])),
    )

    write_metrics_json(payload, out_metrics)

    cm_path = run_dir / f"confusion_matrix_{split}.png"
    try:
        render_confusion_matrix(
            logits=logits, labels=labels, class_names=class_names,
            out_path=cm_path, normalize=True,
            title=f"Confusion Matrix ({split}, normalized)",
        )
        payload["confusion_matrix_path"] = str(cm_path)
        write_metrics_json(payload, out_metrics)  # 二次写以包含 cm 路径
    except Exception as exc:
        click.echo(f"WARN: 混淆矩阵渲染失败: {exc}", err=True)

    # ---- 7. 同步到 manifest.metrics_summary ----
    summary = {
        f"{split}.top1": payload.get("top1"),
        f"{split}.top5": payload.get("top5"),
        f"{split}.macro_f1": payload.get("macro_avg", {}).get("f1"),
        f"{split}.n_samples": payload.get("n_samples"),
    }
    if payload.get("imbalance_warning"):
        summary[f"{split}.imbalance_warning"] = True
    try:
        finalize(run_dir, status="succeeded", metrics_summary=summary)
    except FileNotFoundError:
        pass  # manifest 缺失时不阻断 eval 输出

    # ---- 8. stdout JSON 摘要 ----
    out_summary = {
        "run_id": run_dir.name,
        "split": split,
        "n_samples": payload["n_samples"],
        "top1": payload.get("top1"),
        "top5": payload.get("top5"),
        "macro_avg": payload.get("macro_avg"),
        "metrics_path": str(out_metrics),
        "confusion_matrix_path": str(cm_path) if cm_path.exists() else None,
    }
    click.echo(json.dumps(out_summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ 评估完成: top1={payload.get('top1', 0):.4f}  "
        f"top5={payload.get('top5', 0):.4f}  "
        f"n={payload['n_samples']}",
        err=True,
    )
    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _find_run_dir(checkpoint: Path) -> Path | None:
    """从 ``<run_dir>/[checkpoints/]<file>.pdparams`` 反推 run_dir.

    支持两种布局:
        - PP-TSM 路径: ``<run_dir>/checkpoints/<file>.pdparams`` (本项目惯例)
        - BMN 路径:    ``<run_dir>/BMN_epoch_NNNNN.pdparams``  (上游 BMN 直接写到 run 根目录)

    判定标准: 沿父目录上溯, 找到第一个含 ``manifest.json`` 的目录.
    """
    parent = checkpoint.parent
    # PP-TSM 布局: <run_dir>/checkpoints/<ckpt>
    if parent.name == "checkpoints":
        candidate = parent.parent
        if (candidate / "manifest.json").is_file():
            return candidate
    # BMN 布局: <run_dir>/<ckpt> (parent 直接是 run_dir)
    if (parent / "manifest.json").is_file():
        return parent
    return None


def _read_splits_dir_from_upstream(upstream_yaml: Path) -> Path:
    """从 upstream_config.yaml 中的 DATASET.test.file_path 反推 splits 目录."""
    data = yaml.safe_load(upstream_yaml.read_text(encoding="utf-8")) or {}
    file_path = (data.get("DATASET") or {}).get("test", {}).get("file_path")
    if not file_path:
        # 退化: 假定 repo_root/data/splits
        return upstream_yaml.parent.parent / "data" / "splits"
    return Path(file_path).parent


# --------------------------------------------------------------------------------------
# BMN eval branch (US6 / FR-029, 2026-05-12 新增)
# --------------------------------------------------------------------------------------


def _run_bmn_eval(
    *,
    run_dir: Path,
    ckpt: Path,
    split: str,                       # 在 BMN 流程中并不切换 split 文件 (上游用 label_fixed.json::subset);
                                      # 仅用于章程 IV 闸门 + 输出文件命名.
    upstream_yaml: Path,
    classes_meta: list[dict],
    output_path: Path,
    rerun: bool,
) -> int:
    """BMN 时序定位的 eval 分支.

    与 PP-TSM 路径的关键区别:
      - PP-TSM: 直接读 logits[N,C] → top1/top5/per-class confusion → confusion_matrix.png
      - BMN:    调用上游 ``test_model`` 让 BMNMetric 做 NMS 后处理 →
                输出 ActivityNet 1.3 风格 ``bmn_results_<subset>.json``;
                再调 ``cal_metrics`` 拿到 AR@1/5/10/100 数值.

    schema: ``bmn-eval-v1`` (见 data-model.md US6 节):
      {
        "schema": "bmn-eval-v1",
        "split": "val|test",
        "checkpoint": "...",
        "ar@1": ..., "ar@5": ..., "ar@10": ..., "ar@100": ...,
        "n_videos_evaluated": int, "n_proposals": int,
        "subset": "validation",
        "result_path": "<run>/bmn_eval/bmn_results_validation.json",
        "class_names": [14 个真实类名]
      }
    """
    # 章程 IV: test 重跑必须 --rerun (与 PP-TSM 分支一致)
    if split == "test" and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("split") == "test" and not rerun:
            click.echo(
                "ERROR (章程 IV 违反): 检测到该 run 已有 split=test 的 metrics.json:\n"
                f"  {output_path}\n"
                "测试集禁止反复挑选结果. 如确需重跑, 追加 --rerun.",
                err=True,
            )
            return 3

    click.echo(
        f"[eval/bmn] split={split} checkpoint={ckpt.name} run_id={run_dir.name}",
        err=True,
    )

    try:
        from pingpong_av.upstream_adapter.trainer import (
            UpstreamRuntimeError,
            run_upstream_bmn_eval,
        )
    except ImportError as exc:
        click.echo(f"ERROR: 无法 import upstream_adapter: {exc}", err=True)
        return 2

    # BMN 的 split 概念由 label_fixed.json 内部 subset 字段决定 (train/validation),
    # 我们的 --split val/test 都映射到上游 subset='validation' (因为 BMN 上游没有 test 概念,
    # split=test 时 prepare_bmn_inputs.py 不会写 subset='test' 的条目).
    # 因此 BMN 路径下 split=test 与 split=val 在数据层等价; 我们仍然区分输出文件名 + 章程 IV 闸门.
    subset_for_metric = "validation"

    bmn_eval_dir = run_dir / "bmn_eval"
    bmn_eval_dir.mkdir(parents=True, exist_ok=True)
    result_dir = bmn_eval_dir / "results"
    output_dir = bmn_eval_dir / "intermediate"

    try:
        bmn_metrics = run_upstream_bmn_eval(
            upstream_yaml,
            ckpt,
            result_path=result_dir,
            output_path=output_dir,
            subset=subset_for_metric,
            reuse_existing=True,    # 若同 ckpt 的 bmn_results_validation.json 已存在则复用 (节省 ~8 min 前向)
        )
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        return 1
    except UpstreamRuntimeError as exc:
        click.echo(f"ERROR: 上游 BMN 评估失败: {exc}", err=True)
        return 4
    except Exception as exc:
        click.echo(
            f"ERROR: 上游 BMN 评估意外错误: {type(exc).__name__}: {exc}",
            err=True,
        )
        return 4

    payload = {
        "schema": "bmn-eval-v1",
        "split": split,
        "checkpoint": str(ckpt),
        "run_id": run_dir.name,
        "n_videos_evaluated": bmn_metrics["n_videos_evaluated"],
        "n_proposals": bmn_metrics["n_proposals"],
        "subset": bmn_metrics["subset"],
        "result_path": bmn_metrics["bmn_results_json"],
        "metrics": {
            "ar@1":   bmn_metrics["ar@1"],
            "ar@5":   bmn_metrics["ar@5"],
            "ar@10":  bmn_metrics["ar@10"],
            "ar@100": bmn_metrics["ar@100"],
        },
        "class_names": [c.get("display_name") or c.get("name") for c in classes_meta],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 同步到 manifest.metrics_summary
    summary = {
        f"{split}.bmn.ar@1":   payload["metrics"]["ar@1"],
        f"{split}.bmn.ar@5":   payload["metrics"]["ar@5"],
        f"{split}.bmn.ar@10":  payload["metrics"]["ar@10"],
        f"{split}.bmn.ar@100": payload["metrics"]["ar@100"],
        f"{split}.bmn.n_videos":    payload["n_videos_evaluated"],
        f"{split}.bmn.n_proposals": payload["n_proposals"],
    }
    try:
        finalize(run_dir, status="succeeded", metrics_summary=summary)
    except FileNotFoundError:
        pass

    # stdout JSON 摘要
    out_summary = {
        "schema": payload["schema"],
        "run_id": run_dir.name,
        "split": split,
        "metrics": payload["metrics"],
        "n_videos_evaluated": payload["n_videos_evaluated"],
        "n_proposals": payload["n_proposals"],
        "metrics_path": str(output_path),
    }
    click.echo(json.dumps(out_summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ BMN 评估完成: AR@1={payload['metrics']['ar@1']:.2f}  "
        f"AR@10={payload['metrics']['ar@10']:.2f}  AR@100={payload['metrics']['ar@100']:.2f}  "
        f"n_videos={payload['n_videos_evaluated']}  n_proposals={payload['n_proposals']}",
        err=True,
    )
    return 0
