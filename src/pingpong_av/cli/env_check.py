"""``pp env-check`` 子命令实现 (章程 VIII).

职责:
- 调用 ``pingpong_av.utils.env`` 的检查函数集合;
- 把结构化结果以 JSON 打印到 **stdout** (machine-readable);
- 把每个失败检查的修复指引以人类可读形式打印到 **stderr**;
- 按 contracts/cli.md 约定映射退出码: 通过 → 0, 任一致命检查失败 → 2.

不做的事:
- 不直接执行任何修复 (那是用户的责任 / scripts/bootstrap.sh 的职责).
- 不调用 paddle / paddlevideo 之外的运行时检查 (例如不验证 GPU 显存).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from pingpong_av.utils import env as env_mod
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(*, strict: bool) -> int:
    """执行环境自检并打印结果. 返回应当作为进程退出码使用的整数.

    参数:
        strict: 若为 True, 在基础检查 (Python 版本/解释器路径/GPU) 之上, 还会尝试
                ``import paddle`` 与 ``import paddlevideo``, 并把它们也纳入"是否致命"判定.
                quickstart 第 1 步 (``pp env-check --strict``) 走此分支.

    返回:
        0 — 全部致命检查通过 (允许进入后续命令).
        2 — 任一致命检查失败 (符合 contracts/cli.md 中"环境问题"的退出码).
    """
    results = env_mod.collect_strict() if strict else env_mod.collect_basic()

    # 1. stdout: 机器可读 JSON. 顶层是 {check_name: {ok, detail, hint?}, ...}.
    payload: dict[str, dict[str, Any]] = {r.name: r.as_dict() for r in results}
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    # 2. stderr: 人类可读的失败报告 + 修复指引
    failed = [r for r in results if not r.ok]
    fatal_failed = [r for r in failed if r.name in _FATAL_CHECK_NAMES]

    if not failed:
        click.echo("✓ 所有环境检查通过.", err=True)
        return 0

    click.echo("", err=True)
    click.echo(f"环境检查共 {len(results)} 项, 失败 {len(failed)} 项 (致命 {len(fatal_failed)} 项):", err=True)
    for r in failed:
        kind = "FATAL" if r.name in _FATAL_CHECK_NAMES else "WARN "
        click.echo(f"  [{kind}] {r.name}: {_format_detail_short(r.detail)}", err=True)
        if r.hint:
            for line in r.hint.splitlines():
                click.echo(f"          {line}", err=True)

    if env_mod.all_passed(results):
        # 没有致命失败, 仅有 GPU 缺失之类的警告 — 视为通过
        click.echo("", err=True)
        click.echo("⚠ 上述失败均为非致命 (例如 GPU 缺失). 视为通过, 但训练可能受影响.", err=True)
        return 0

    click.echo("", err=True)
    click.echo("✗ env-check 未通过. 请按上面的指引修复, 然后重试.", err=True)
    return 2


_FATAL_CHECK_NAMES = {
    "python_version",
    "interpreter_is_project_venv",
    "paddle_importable",
    "paddlevideo_importable",
}


def _format_detail_short(detail: dict[str, Any]) -> str:
    """detail dict 的单行摘要, 用于 stderr 报告."""
    if "error" in detail:
        return str(detail["error"])
    if "actual" in detail and "expected" in detail:
        return f"actual={detail['actual']!r}  expected={detail['expected']!r}"
    if "version" in detail:
        return f"version={detail['version']}"
    return ", ".join(f"{k}={v}" for k, v in detail.items() if v is not None)
