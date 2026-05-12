"""``pp train`` 子命令 (FR-008/009/010/012/018, 章程 II/III/VIII).

编排:
    1. 加载业务配置 YAML, 计算 config_hash;
    2. 通过 :func:`pingpong_av.experiment.run_manifest.create_run_dir` 创建实验目录,
       同时**校验 git 工作区是否干净** (脏且无 ``--allow-dirty`` → 退出码 3, 章程 II);
    3. ``snapshot_config`` 把合并后配置写入 ``<run>/config.yaml`` 供 eval 时回溯;
    4. 通过模型 loader (T042/T043) 把配置转为上游格式 YAML;
    5. 调用 :func:`pingpong_av.upstream_adapter.trainer.run_upstream_train`;
    6. 完成时 :func:`finalize` 更新 manifest (status / finished_at / metrics_summary);
    7. stdout 输出结构化结果 JSON.

退出码:
    0  成功
    1  用户输入错 (配置不存在 / resume checkpoint 缺失)
    2  环境问题 (paddle 不可导 — 由 upstream_adapter 抛出)
    3  章程硬约束违反 (工作区脏未加 --allow-dirty 等)
    4  运行时失败 (训练发散 / 上游异常)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pingpong_av.experiment.run_manifest import (
    ConstitutionViolation,
    create_run_dir,
    finalize,
    snapshot_config,
)
from pingpong_av.models.registry import get_model_loader
from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger
from pingpong_av.utils.seeding import set_seed

_log = get_logger(__name__)


def run(
    *,
    config_path: str,
    seed_override: int | None,
    resume_path: str | None,
    allow_dirty: bool,
    output_root: str,
) -> int:
    """执行 train. 返回应当作为进程退出码使用的整数."""

    # ---- 1. 加载业务配置 ----
    cfg_path = Path(config_path).resolve()
    if not cfg_path.is_file():
        click.echo(f"ERROR: 配置文件不存在: {cfg_path}", err=True)
        return 1
    try:
        loaded = load_config(cfg_path)
    except ConfigError as exc:
        click.echo(f"ERROR: 配置校验失败: {exc}", err=True)
        return 1

    config = loaded.data
    seed = seed_override if seed_override is not None else int(config.get("seed", 2026))

    repo_root = find_repo_root()
    output_root_p = Path(output_root)
    if not output_root_p.is_absolute():
        output_root_p = (repo_root / output_root_p).resolve()

    # ---- 1a. resume 路径校验 ----
    resume_abs: Path | None = None
    if resume_path is not None:
        resume_abs = Path(resume_path).resolve()
        if not resume_abs.is_file():
            click.echo(f"ERROR: --resume 指向的 checkpoint 不存在: {resume_abs}", err=True)
            return 1

    # ---- 2. 创建实验目录 + manifest (章程 II 闸门在此触发) ----
    notes: dict = {}
    if resume_abs is not None:
        notes["resumed_from"] = str(resume_abs)
    if allow_dirty:
        notes["allow_dirty"] = True

    try:
        run_dir, manifest = create_run_dir(
            kind="train",
            config_hash=loaded.config_hash,
            seed=seed,
            slug=cfg_path.stem,
            output_root=output_root_p,
            repo_root=repo_root,
            dataset_split_version=str(config.get("split_version", "unknown")),
            allow_dirty=allow_dirty,
            notes=notes,
        )
    except ConstitutionViolation as exc:
        click.echo("ERROR (章程 II 违反): " + str(exc), err=True)
        return 3

    _log.info("experiment dir created", extra={"run_id": manifest.run_id, "path": str(run_dir)})
    click.echo(f"[train] run_id = {manifest.run_id}", err=True)

    # ---- 3. snapshot_config ----
    snapshot_config(dict(config), run_dir, filename="config.yaml")

    # ---- 4. 转为上游格式 ----
    try:
        loader = get_model_loader(config.get("model", {}).get("name", ""))
    except KeyError as exc:
        finalize(run_dir, status="failed", extra_notes={"error": str(exc)})
        click.echo(f"ERROR: {exc}", err=True)
        return 1

    try:
        upstream_yaml, upstream_dict = loader(
            config,
            splits_dir=repo_root / "data" / "splits",
            output_dir=run_dir,
            repo_root=repo_root,
        )
    except ConfigError as exc:
        finalize(run_dir, status="failed", extra_notes={"error": str(exc)})
        click.echo(f"ERROR: 模型配置合并失败: {exc}", err=True)
        return 1

    # 同步把 upstream 格式 dict 也存入 run_dir, 便于排查"实际丢给上游"的内容
    snapshot_config(upstream_dict, run_dir, filename="upstream_config.yaml")

    # ---- 5. 调用上游训练 ----
    set_seed(seed)
    try:
        from pingpong_av.upstream_adapter.trainer import run_upstream_train
    except ImportError as exc:
        finalize(run_dir, status="failed", extra_notes={"error": f"import upstream_adapter failed: {exc}"})
        click.echo(f"ERROR: 无法 import upstream_adapter: {exc}", err=True)
        return 2

    try:
        run_upstream_train(
            upstream_yaml,
            output_dir=run_dir,
            seed=seed,
            resume=resume_abs,
            weights=None,  # 冷启动权重由 model.backbone.pretrained 在 upstream config 中指定
            validate=True,
            amp=bool((config.get("train") or {}).get("amp", False)),
        )
    except KeyboardInterrupt:
        finalize(run_dir, status="interrupted")
        click.echo("INTERRUPTED: 训练被用户中断 (Ctrl-C). manifest.status = interrupted.", err=True)
        return 4
    except Exception as exc:  # 上游训练失败 → 退出码 4
        err_repr = f"{type(exc).__name__}: {exc}"
        finalize(run_dir, status="failed", extra_notes={"error": err_repr})
        click.echo(f"ERROR: 上游训练失败: {err_repr}", err=True)
        return 4

    # ---- 6. finalize ----
    # best_val_top1 / best_checkpoint 由上游训练循环自己写到 run_dir;
    # 这里尝试发现 best.pdparams, 失败则只标记成功状态
    best_ckpt = _find_best_checkpoint(run_dir)
    metrics_summary = {}
    if best_ckpt is not None:
        metrics_summary["best_checkpoint"] = str(best_ckpt)
    finalize(run_dir, status="succeeded", metrics_summary=metrics_summary)

    # ---- 7. stdout JSON 摘要 ----
    payload = {
        "run_id": manifest.run_id,
        "status": "succeeded",
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "experiment_dir": str(run_dir),
        "config_hash": loaded.config_hash,
        "seed": seed,
    }
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    click.echo(f"✓ 训练完成: {run_dir}", err=True)

    # 临时上游 yaml 清理 (保留 run_dir 内的 snapshot 即可)
    try:
        Path(upstream_yaml).unlink(missing_ok=True)
        Path(upstream_yaml).parent.rmdir()
    except OSError:
        pass

    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    """尝试发现 best.pdparams; 失败则返回最新的 epoch_*.pdparams; 都没有则 None.

    上游 PaddleVideo 的命名习惯不完全一致, 这里尽量兼容.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    # 优先 best.pdparams
    for cand in (ckpt_dir / "best.pdparams", ckpt_dir / "best_model.pdparams"):
        if cand.is_file():
            return cand

    # 次选最新的 epoch_*.pdparams (按 mtime 排序)
    epochs = sorted(
        ckpt_dir.glob("*.pdparams"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return epochs[0] if epochs else None
