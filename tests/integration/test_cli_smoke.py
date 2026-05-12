"""CLI 启动性集成测试 (FR-019).

每个 ``pp <subcommand> --help`` 必须 exit 0 且打印帮助文本.
此测试不依赖 paddle / paddlevideo, 仅验证 click 注册与 stub 解析.

通过 ``CliRunner`` 在进程内调用避免子进程开销; 同时也验证
``pyproject.toml`` 的 entry point 在 .venv 安装后能正确解析到 :func:`main`.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from pingpong_av.cli import cli


# 6 个子命令名 (与 contracts/cli.md 对齐)
SUBCOMMANDS = [
    "env-check",
    "data-prepare",
    "train",
    "eval",
    "infer-clip",
    "infer-video",
]


@pytest.fixture()
def runner() -> CliRunner:
    # mix_stderr=False 让 result.stdout 只包含 JSON, stderr 单独存放;
    # 这是 contracts/cli.md 的"stdout=结构化输出, stderr=人类可读"约定的对应测试侧设置.
    try:
        return CliRunner(mix_stderr=False)  # click < 8.2
    except TypeError:
        return CliRunner()  # click >= 8.2 默认就是分离的


def test_top_level_help(runner: CliRunner) -> None:
    """`pp --help` exit 0 并列出所有子命令."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for sub in SUBCOMMANDS:
        assert sub in result.output, f"`pp --help` 输出中缺少子命令: {sub}\n{result.output}"


def test_top_level_version(runner: CliRunner) -> None:
    """`pp --version` exit 0 并打印版本号."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # 版本号格式不强制, 但必须包含 'pp' 或数字
    assert any(ch.isdigit() for ch in result.output), result.output


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_each_subcommand_help(runner: CliRunner, sub: str) -> None:
    """每个 `pp <sub> --help` 都必须 exit 0."""
    result = runner.invoke(cli, [sub, "--help"])
    assert result.exit_code == 0, (
        f"`pp {sub} --help` 退出码非 0: {result.exit_code}\n--- output ---\n{result.output}"
    )
    # 帮助输出至少应包含子命令名 (click 默认行为)
    assert sub.replace("-", "") in result.output.replace("-", "").lower() or sub in result.output


@pytest.mark.parametrize("sub", [s for s in SUBCOMMANDS if s != "env-check"])
def test_unimplemented_subcommands_exit_2(runner: CliRunner, sub: str) -> None:
    """除 env-check 已通过 utils.env 接通, 其他子命令在缺必填参数时应退出非 0.

    具体退出码: 缺参数时 click 返回 2 (用法错误); stub 实际调用时也是 2.
    """
    # 不传必填参数, 期望非 0 退出 (click 的 usage error 或 stub 的 'not implemented')
    result = runner.invoke(cli, [sub])
    assert result.exit_code != 0, (
        f"`pp {sub}` 不传参数应非 0 退出, 实际 exit={result.exit_code}\n{result.output}"
    )


def test_env_check_runs_and_returns(runner: CliRunner) -> None:
    """`pp env-check` (非 strict) 必须能跑完 — 即便系统 Python 不是 3.11 也应该输出 JSON."""
    import json

    result = runner.invoke(cli, ["env-check"])
    # 不强制 exit 0 (在系统 Python 非 3.11 的环境里就是会退出 2);
    # 但 stdout 必须是合法 JSON, 表明检查函数都走完了.
    assert result.exit_code in (0, 2), result.output
    payload = json.loads(result.stdout)
    # 至少包含三项基础检查
    assert "python_version" in payload
    assert "interpreter_is_project_venv" in payload
    assert "gpu" in payload
