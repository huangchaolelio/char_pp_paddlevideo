"""``pp env-check --strict`` 集成测试 (T071, 章程 VIII).

通过 ``subprocess`` 真正启动 ``.venv/bin/pp env-check --strict``, 断言:
  - 退出码 0 (一致性闸门)
  - stdout 是合法 JSON, 含约定的 5 项 (basic 3 + strict 2)
  - python_version.detail.actual 以 "3.11." 开头
  - paddle_importable / paddlevideo_importable 都 ok=true (章程 VIII 的核心要求)

若 ``.venv`` 不存在 → ``pytest.skip`` (例如在 CI 矩阵的非完整环境中).
若上游 paddle 不可导 (.venv 中漏装) → 测试失败 (是真实问题, 应被这层测试捕获).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PP_BIN = _REPO_ROOT / ".venv" / "bin" / "pp"


@pytest.fixture(scope="module")
def pp_bin() -> Path:
    if not _PP_BIN.is_file():
        pytest.skip(
            f"{_PP_BIN} 不存在; 跳过 env-check 集成测试. "
            "请先运行 `bash scripts/bootstrap.sh` 准备 .venv (章程 VIII).",
            allow_module_level=False,
        )
    return _PP_BIN


# --------------------------------------------------------------------------------------
# 基础启动性 (即便 paddle 缺失也应通过)
# --------------------------------------------------------------------------------------


def test_env_check_help_runs(pp_bin: Path) -> None:
    """`pp env-check --help` 必须 exit 0 (T027 的 unit 已覆盖, 此处子进程级回归)."""
    result = subprocess.run(
        [str(pp_bin), "env-check", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "env-check" in result.stdout.lower() or "--strict" in result.stdout


def test_env_check_basic_returns_valid_json(pp_bin: Path) -> None:
    """`pp env-check` (无 --strict) 必须输出合法 JSON 到 stdout."""
    result = subprocess.run(
        [str(pp_bin), "env-check"],
        capture_output=True, text=True, timeout=30,
    )
    # exit 可以是 0 (绿) 或 2 (致命检查失败), 但 stdout 必须是 JSON
    assert result.returncode in (0, 2), result.stderr
    payload = json.loads(result.stdout)
    for key in ("python_version", "interpreter_is_project_venv", "gpu"):
        assert key in payload, f"缺少 {key}: {payload}"


# --------------------------------------------------------------------------------------
# T071 主验收: --strict 在 .venv 完备时必须 exit 0
# --------------------------------------------------------------------------------------


def test_env_check_strict_returns_zero(pp_bin: Path) -> None:
    """章程 VIII 主验收: `.venv` 完整时 strict 必须全绿 exit 0."""
    result = subprocess.run(
        [str(pp_bin), "env-check", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"env-check --strict 退出码非 0 (实际 {result.returncode}). "
            "请检查 .venv 中是否有 paddle / paddlevideo / Python 3.11.x.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    payload = json.loads(result.stdout)
    # 5 项检查全在
    for key in ("python_version", "interpreter_is_project_venv", "gpu",
                "paddle_importable", "paddlevideo_importable"):
        assert key in payload, f"strict 模式缺少 {key}: {payload}"
    # 每项都 ok
    for key, item in payload.items():
        assert item.get("ok") is True, f"{key} 未通过: {item}"


def test_env_check_strict_python_version_is_3_11(pp_bin: Path) -> None:
    """章程 VIII 核心: Python 必须是 3.11.x."""
    result = subprocess.run(
        [str(pp_bin), "env-check", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip("env-check --strict 未通过 — 见前一个测试; 不重复报错")
    payload = json.loads(result.stdout)
    actual = payload["python_version"]["detail"]["actual"]
    assert actual.startswith("3.11."), (
        f"章程 VIII 违反: Python 版本应以 3.11. 开头, 实际为 {actual!r}"
    )


def test_env_check_strict_paddle_is_importable(pp_bin: Path) -> None:
    """章程 VIII: .venv 中的 paddle 必须真实可导."""
    result = subprocess.run(
        [str(pp_bin), "env-check", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip("env-check --strict 未通过 — 见前一个测试")
    payload = json.loads(result.stdout)
    paddle_item = payload["paddle_importable"]
    assert paddle_item["ok"] is True, paddle_item
    # detail 里应包含版本号
    assert "version" in paddle_item.get("detail", {}), paddle_item


def test_env_check_strict_paddlevideo_is_importable(pp_bin: Path) -> None:
    """章程 VI: paddlevideo 由 importer.py 通过 sys.path 注入, env-check 复用此机制."""
    result = subprocess.run(
        [str(pp_bin), "env-check", "--strict"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.skip("env-check --strict 未通过 — 见前一个测试")
    payload = json.loads(result.stdout)
    ppv = payload["paddlevideo_importable"]
    assert ppv["ok"] is True, ppv
    # module_path 应在 third_party/PaddleVideo/paddlevideo/__init__.py
    module_path = ppv.get("detail", {}).get("module_path", "")
    assert "third_party/PaddleVideo" in module_path, (
        f"paddlevideo 不是从 submodule 加载: {module_path}"
    )
