"""环境自检函数集合 (章程 VIII 的执行单元).

每个函数返回**结构化结果**而**不直接打印**, 由 `pp env-check` 子命令 (T028) 决定如何展示.
这样设计便于:
- 集成测试以 dict 形式断言;
- 多个检查并行展示, 避免散乱的 print 调用.

不在本模块的范围:
- CLI 层的 JSON 输出格式 (那是 `pingpong_av.cli.env_check`).
- 上游 PaddleVideo 的 import 兜底逻辑 (那是 `upstream_adapter.importer`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    """单条自检的结果."""

    name: str
    ok: bool
    detail: dict[str, Any]
    hint: str | None = None  # 失败时给用户的修复指引

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["hint"] is None:
            d.pop("hint")
        return d


# --------------------------------------------------------------------------------------
# Python 版本与解释器路径
# --------------------------------------------------------------------------------------

EXPECTED_PY_MAJOR = 3
EXPECTED_PY_MINOR = 11


def check_python_version() -> CheckResult:
    """要求 Python 3.11.x (章程 VIII)."""
    major, minor = sys.version_info[:2]
    actual = f"{major}.{minor}.{sys.version_info.micro}"
    ok = (major, minor) == (EXPECTED_PY_MAJOR, EXPECTED_PY_MINOR)
    hint = (
        None
        if ok
        else (
            f"本项目锁定 Python {EXPECTED_PY_MAJOR}.{EXPECTED_PY_MINOR}.x (章程 VIII). "
            f"当前为 {actual}. 请用 python3.11 重新创建 .venv: "
            f"`rm -rf .venv && bash scripts/bootstrap.sh`"
        )
    )
    return CheckResult(
        name="python_version",
        ok=ok,
        detail={"actual": actual, "expected": f"{EXPECTED_PY_MAJOR}.{EXPECTED_PY_MINOR}.x"},
        hint=hint,
    )


def _project_venv_python(repo_root: Path) -> Path:
    """计算项目 .venv 的 python 解释器路径."""
    if os.name == "nt":  # pragma: no cover — 我们目标是 Linux
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def find_repo_root(start: Path | None = None) -> Path:
    """从给定路径或当前模块向上查找仓库根 (含 pyproject.toml + .specify/).

    若找不到, 回退到当前工作目录.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / ".specify").is_dir():
            return candidate
    return Path.cwd().resolve()


def check_interpreter_is_project_venv(repo_root: Path | None = None) -> CheckResult:
    """要求当前解释器位于项目 .venv/ 内 (章程 VIII)."""
    root = repo_root or find_repo_root()
    expected = _project_venv_python(root)
    actual = Path(sys.executable).resolve()
    # 允许通过 symlink 指向同一物理文件 (例如 conda + .venv 混用场景)
    try:
        ok = actual == expected.resolve()
    except OSError:
        ok = actual == expected
    hint = (
        None
        if ok
        else (
            "当前解释器不在项目 .venv 内, 违反章程 VIII. "
            f"请改用 `{expected}` 或先执行 `source {root}/.venv/bin/activate`. "
            f"如 .venv 不存在, 请运行 `bash scripts/bootstrap.sh`."
        )
    )
    return CheckResult(
        name="interpreter_is_project_venv",
        ok=ok,
        detail={
            "actual": str(actual),
            "expected": str(expected),
            "repo_root": str(root),
        },
        hint=hint,
    )


# --------------------------------------------------------------------------------------
# 上游可导性
# --------------------------------------------------------------------------------------


def check_paddle_importable() -> CheckResult:
    """是否可 `import paddle`."""
    try:
        import paddle  # noqa: WPS433  (运行时 import 是本函数的本意)
    except Exception as exc:  # ImportError / 任意 paddle 导入期错误
        return CheckResult(
            name="paddle_importable",
            ok=False,
            detail={"error": f"{type(exc).__name__}: {exc}"},
            hint=(
                "无法导入 paddle. 请确认已运行 `bash scripts/bootstrap.sh` 完成依赖安装, "
                "或检查 requirements/upstream-py311.txt 中的 paddlepaddle-gpu 是否已安装."
            ),
        )
    version = getattr(paddle, "__version__", "unknown")
    return CheckResult(
        name="paddle_importable",
        ok=True,
        detail={"version": version},
    )


def check_paddlevideo_importable() -> CheckResult:
    """是否可 ``import paddlevideo``.

    本项目**不**通过 pip 安装上游 PaddleVideo (上游 setup.py 在 Python 3.11 下
    无法直接生效, 见 scripts/bootstrap.sh 关于 R3 / T033 的说明). 取而代之, 由
    ``pingpong_av.upstream_adapter.importer.ensure_paddlevideo_on_path()`` 在
    运行时把 ``third_party/PaddleVideo`` 加入 ``sys.path``. 本检查复用同一逻辑,
    与业务代码实际加载路径保持一致.
    """
    # 延迟 import, 避免 utils 包加载期硬依赖 upstream_adapter
    try:
        from pingpong_av.upstream_adapter.importer import (
            UpstreamImportError,
            ensure_paddlevideo_on_path,
        )
    except ImportError as exc:  # pragma: no cover  本项目自身被破坏
        return CheckResult(
            name="paddlevideo_importable",
            ok=False,
            detail={"error": f"内部错误: 无法导入 upstream_adapter: {exc}"},
            hint="请确认 src/pingpong_av/upstream_adapter/importer.py 完整存在.",
        )

    try:
        paddlevideo = ensure_paddlevideo_on_path()
    except UpstreamImportError as exc:
        first_line = (str(exc).splitlines() or ["unknown"])[0]
        return CheckResult(
            name="paddlevideo_importable",
            ok=False,
            detail={"error": first_line},
            hint=(
                "无法导入 paddlevideo. 请确认: "
                "(1) submodule 已 init (`git submodule update --init --recursive`); "
                "(2) bootstrap 已成功 (`bash scripts/bootstrap.sh`); "
                "(3) 如有 3.11 兼容错误, 在 third_party/patches/ 追加补丁后重跑 bootstrap."
            ),
        )
    except Exception as exc:  # 上游 import 期抛出的其他错误 (3.11 兼容问题)
        return CheckResult(
            name="paddlevideo_importable",
            ok=False,
            detail={"error": f"{type(exc).__name__}: {exc}"},
            hint=(
                "import paddlevideo 抛出非 ImportError 错误 (可能是 3.11 兼容). "
                "在 third_party/patches/ 下追加补丁修复, 然后重新运行 bootstrap."
            ),
        )

    version = getattr(paddlevideo, "__version__", "unknown")
    return CheckResult(
        name="paddlevideo_importable",
        ok=True,
        detail={"version": version, "module_path": getattr(paddlevideo, "__file__", "unknown")},
    )


# --------------------------------------------------------------------------------------
# GPU / CUDA
# --------------------------------------------------------------------------------------


def check_gpu() -> CheckResult:
    """探测 GPU 可用性. 失败不视为致命 (CPU 回退是 spec.md 边界情况允许的)."""
    detail: dict[str, Any] = {"gpu_available": False, "gpu_count": 0, "gpu_model": None,
                              "cuda_version": None}
    # 优先通过 paddle 检测 (与训练实际使用的栈一致)
    try:
        import paddle
        detail["gpu_available"] = bool(paddle.device.cuda.device_count())
        detail["gpu_count"] = int(paddle.device.cuda.device_count())
        if detail["gpu_available"]:
            detail["gpu_model"] = paddle.device.cuda.get_device_name(0)
    except Exception:  # paddle 不可导, 兜底用 nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0:
                    names = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
                    if names:
                        detail["gpu_available"] = True
                        detail["gpu_count"] = len(names)
                        detail["gpu_model"] = names[0]
            except (subprocess.SubprocessError, OSError):
                pass

    # CUDA 版本: 通过环境变量或 nvidia-smi 获取
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                detail["driver_version"] = proc.stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, OSError):
            pass

    return CheckResult(
        name="gpu",
        ok=True,  # 不强制要求 GPU; 上层根据 detail.gpu_available 自行决定是否警告
        detail=detail,
    )


# --------------------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------------------


def collect_basic() -> list[CheckResult]:
    """非 strict 模式: 只检查解释器与版本, 不 import paddle/paddlevideo."""
    return [check_python_version(), check_interpreter_is_project_venv(), check_gpu()]


def collect_strict() -> list[CheckResult]:
    """strict 模式: 在基本检查之上, 尝试 import paddle 与 paddlevideo."""
    return [
        check_python_version(),
        check_interpreter_is_project_venv(),
        check_paddle_importable(),
        check_paddlevideo_importable(),
        check_gpu(),
    ]


def all_passed(results: list[CheckResult]) -> bool:
    """聚合判定: 章程相关检查全部 ok 才算通过. GPU 缺失只是警告, 不致命."""
    fatal_names = {
        "python_version",
        "interpreter_is_project_venv",
        "paddle_importable",
        "paddlevideo_importable",
    }
    return all(r.ok for r in results if r.name in fatal_names)
