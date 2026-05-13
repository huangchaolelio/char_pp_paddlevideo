"""pytest configuration: skip `@pytest.mark.slow` tests by default.

用 `--runslow` 启用慢测试 (含 GPU 端到端).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="跑标记为 @pytest.mark.slow 的慢测试 (含 GPU e2e).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return  # 不跳过任何
    skip_slow = pytest.mark.skip(reason="需要 --runslow 启用 (含 GPU + 真实模型加载)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
