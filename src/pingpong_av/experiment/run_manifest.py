"""实验运行清单 (章程 II: 可复现实验).

每个 ``pp train`` / ``pp eval`` 调用都对应一个 ``experiments/<run_id>/`` 目录, 本模块提供:

* :class:`RunManifest` — 持久化为 ``manifest.json`` 的数据结构.
* :func:`create_run_dir` — 启动期生成 run_id + 创建目录 + 写初始 manifest.
* :func:`finalize` — 成功 / 失败结束时更新 manifest (status, finished_at, metrics).
* :func:`snapshot_config` — 把本次使用的合并配置原样写入 ``<run>/config.yaml``.

**硬约束** (章程 II / VIII):

* commit SHA + config_hash + seed + dataset_split_version 四元组缺一不可.
* 工作区脏 (有未提交修改) 时默认拒绝启动, 需 ``allow_dirty=True`` 才能继续.
* python_version 必须以 ``3.11`` 开头, 否则抛 :class:`ConstitutionViolation`.

本模块不触碰业务指标的 schema (那是 ``pingpong_av.evaluation.reporter``); 只存储
简单的键值对指标用于 manifest 级别的快速浏览.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "ConstitutionViolation",
    "RunKind",
    "RunManifest",
    "create_run_dir",
    "finalize",
    "snapshot_config",
]

RunKind = Literal["train", "eval"]
_STATUS = Literal["running", "succeeded", "failed", "interrupted"]


class ConstitutionViolation(RuntimeError):
    """违反章程硬约束 (章程 II/IV/VIII) 时抛出, 由 CLI 映射为退出码 3."""


# --------------------------------------------------------------------------------------
# Manifest 数据结构
# --------------------------------------------------------------------------------------


@dataclass
class RunManifest:
    """实验清单 (与 data-model.md 中 manifest schema 对齐)."""

    run_id: str
    kind: RunKind
    commit: str
    dirty: bool
    config_hash: str
    seed: int
    python_version: str
    cuda_version: str | None
    gpu_model: str | None
    started_at: str                    # ISO 8601 UTC
    finished_at: str | None = None
    status: _STATUS = "running"
    dataset_split_version: str | None = None
    # 可选的瞬时指标 (训练结束时的 best_val_top1, 评估完成后的 top1/top5 等)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    # 备注字段, 例如 "resumed_from": "<checkpoint_path>"
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Git / 环境探测
# --------------------------------------------------------------------------------------


def _git_commit(repo_root: Path) -> str:
    """仓库 HEAD 的完整 SHA. 若非 git 仓库则返回 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _git_is_dirty(repo_root: Path) -> bool:
    """工作区是否有未提交修改 (包括未跟踪文件不算; 关注 tracked 文件的修改)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return False  # 无 git 时不拒绝, 但 commit 会是 'unknown'


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _cuda_version() -> str | None:
    """优先通过 paddle.version.cuda() 读取; 否则尝试 nvidia-smi; 失败返回 None."""
    try:
        import paddle

        v = paddle.version.cuda() if hasattr(paddle.version, "cuda") else None
        if v and v != "False":
            return str(v)
    except Exception:
        pass

    import shutil
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except subprocess.SubprocessError:
            pass
    return None


def _gpu_model() -> str | None:
    try:
        import paddle

        if paddle.device.cuda.device_count() > 0:
            return paddle.device.cuda.get_device_name(0)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------------------
# run_id 生成
# --------------------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """把字符串转成安全的 slug: 小写字母数字下划线."""
    import re
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text).strip("_").lower()
    return s[:48] or "run"


def _new_run_id(kind: RunKind, commit: str, slug: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha7 = (commit[:7] if commit and commit != "unknown" else "nocommit")
    return f"{now}-{sha7}-{kind}-{_slugify(slug)}"


# --------------------------------------------------------------------------------------
# 对外 API
# --------------------------------------------------------------------------------------


def create_run_dir(
    *,
    kind: RunKind,
    config_hash: str,
    seed: int,
    slug: str,
    output_root: str | Path,
    repo_root: Path,
    dataset_split_version: str | None,
    allow_dirty: bool = False,
    notes: dict[str, Any] | None = None,
) -> tuple[Path, RunManifest]:
    """创建实验目录并写入初始 ``manifest.json``.

    参数:
        kind: ``"train"`` 或 ``"eval"``.
        config_hash: 由 :func:`pingpong_av.utils.config.compute_config_hash` 产生.
        seed: 随机种子 (章程 II).
        slug: 人类可读的短 slug, 通常取自 config 文件名 (e.g. ``"pp_tsm_pingpong"``).
        output_root: ``experiments/`` 根 (默认).
        repo_root: 项目仓库根, 用于 git 探测.
        dataset_split_version: 数据划分版本 (章程 IV). None 表示本次 run 与数据集划分无关
                               (极少见, 通常是 eval 直接复用旧 split).
        allow_dirty: 工作区脏时是否允许启动 (默认 False).

    返回:
        ``(run_dir: Path, manifest: RunManifest)``.

    抛出:
        :class:`ConstitutionViolation`: 工作区脏且 ``allow_dirty=False``;
                                        或 python 版本不是 3.11.x;
                                        或 commit='unknown' 且 allow_dirty=False.
    """
    # 章程 VIII: Python 版本门 (冗余但廉价)
    py = _python_version()
    if not py.startswith("3.11"):
        raise ConstitutionViolation(
            f"Python 版本 {py} 不是 3.11.x (章程 VIII). "
            "请在项目 .venv 中运行此命令."
        )

    commit = _git_commit(repo_root)
    dirty = _git_is_dirty(repo_root)

    if dirty and not allow_dirty:
        raise ConstitutionViolation(
            "工作区有未提交的修改, 为保证可复现性 (章程 II) 禁止启动训练/评估.\n"
            "请先提交或 stash 改动; 如确需继续, 追加 `--allow-dirty` (本次结果不得作为正式指标)."
        )
    if commit == "unknown" and not allow_dirty:
        raise ConstitutionViolation(
            "无法读取 git commit, 无法写入 manifest 的 commit 字段 (章程 II). "
            "请确认当前目录是 git 仓库; 如确需跳过, 追加 `--allow-dirty`."
        )

    run_id = _new_run_id(kind, commit, slug)
    output_root_path = Path(output_root).resolve()
    run_dir = output_root_path / run_id
    if run_dir.exists():
        raise ConstitutionViolation(
            f"run_id 冲突 (目录已存在): {run_dir}. 等待 1 秒后重试, 或自行清理."
        )
    (run_dir / "log").mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)

    manifest = RunManifest(
        run_id=run_id,
        kind=kind,
        commit=commit,
        dirty=dirty,
        config_hash=config_hash,
        seed=seed,
        python_version=py,
        cuda_version=_cuda_version(),
        gpu_model=_gpu_model(),
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        dataset_split_version=dataset_split_version,
        notes=dict(notes or {}),
    )
    _write_manifest(run_dir, manifest)
    return run_dir, manifest


def finalize(
    run_dir: str | Path,
    *,
    status: _STATUS,
    metrics_summary: dict[str, Any] | None = None,
    extra_notes: dict[str, Any] | None = None,
) -> RunManifest:
    """结束一次 run, 更新 ``manifest.json`` 的 status / finished_at / metrics_summary.

    不改变已有字段 (commit / config_hash / seed 等不会在此处被重写).
    """
    run_dir = Path(run_dir).resolve()
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在, 无法 finalize: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    data["finished_at"] = datetime.now(timezone.utc).isoformat()
    if metrics_summary:
        existing = data.get("metrics_summary", {}) or {}
        existing.update(metrics_summary)
        data["metrics_summary"] = existing
    if extra_notes:
        existing_notes = data.get("notes", {}) or {}
        existing_notes.update(extra_notes)
        data["notes"] = existing_notes

    # 重建 RunManifest 用于返回 (忽略未识别的字段以向前兼容)
    known = {f for f in RunManifest.__dataclass_fields__}
    clean = {k: v for k, v in data.items() if k in known}
    manifest = RunManifest(**clean)
    _write_manifest(run_dir, manifest)
    return manifest


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    path = run_dir / "manifest.json"
    # 写入时用 sort_keys + indent, 便于人工 review 与 diff
    text = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# T020: 配置 snapshot
# --------------------------------------------------------------------------------------


def snapshot_config(
    config_data: dict[str, Any],
    run_dir: str | Path,
    *,
    filename: str = "config.yaml",
) -> Path:
    """把合并后的配置 dict 原样写入 ``<run_dir>/<filename>`` (章程 II / III).

    该文件在后续 ``pp eval`` 中会被读取, 用于**基于 checkpoint 自动找回训练配置**
    (contracts/cli.md: ``--checkpoint`` 自动探测同目录 config.yaml).

    已存在同名文件时会被**覆盖** (同一 run 内配置不应变化; 若变化说明上层逻辑错误).
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    out = run_dir / filename
    # 使用 safe_dump + allow_unicode, 避免中文 class 名称被转义
    text = yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    out.write_text(text, encoding="utf-8")
    return out
