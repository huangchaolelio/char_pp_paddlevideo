"""统一结构化日志.

设计原则:
- 默认输出到 **stderr** (人类可读), stdout 留给 CLI 的结构化结果 JSON (contracts/cli.md 通则).
- 可选同时写入 JSON Lines 文件, 用于实验目录 (`experiments/<run>/log/*.log`) 的机读分析.
- 不向运行时强制引入第三方日志框架; 保留标准库 `logging`, 让上游 PaddleVideo 的日志默认行为不被劫持.

不在本模块的范围:
- 业务级别的指标记录 (那是 `pingpong_av.evaluation.reporter` 的职责).
- 实验生命周期事件 (那是 `pingpong_av.experiment.run_manifest` 的职责).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


class _JsonLineFormatter(logging.Formatter):
    """每条日志一行 JSON, 适合 grep / jq 处理.

    字段: ts, level, name, message, 以及 extra={} 中的所有键值.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # 透传 extra={...} 注入的键 (LogRecord 上额外的属性)
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _has_handler_of_kind(logger: logging.Logger, kind: type) -> bool:
    return any(isinstance(h, kind) for h in logger.handlers)


def get_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    json_file: str | Path | None = None,
) -> logging.Logger:
    """返回配置好的 Logger.

    参数:
        name: logger 名 (惯用 `__name__`).
        level: 日志级别 (默认 INFO).
        json_file: 若提供, 同时把日志以 JSON Lines 写入该文件 (追加模式).

    幂等:
        多次以相同 name 调用不会重复添加 handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 不让父 logger (如 paddle 的 root) 重复打印

    # stderr 文本 handler — 始终添加, 但仅添加一次
    if not _has_handler_of_kind(logger, logging.StreamHandler):
        text_h = logging.StreamHandler(stream=sys.stderr)
        text_h.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        text_h.setLevel(level)
        logger.addHandler(text_h)

    # JSON 文件 handler — 可选, 同样幂等 (按目标文件路径去重)
    if json_file is not None:
        target = Path(json_file).resolve()
        existing = {
            getattr(h, "_pingpong_target", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if target not in existing:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_h = logging.FileHandler(target, mode="a", encoding="utf-8")
            file_h.setFormatter(_JsonLineFormatter())
            file_h.setLevel(level)
            file_h._pingpong_target = target  # type: ignore[attr-defined]
            logger.addHandler(file_h)

    return logger


def log_kv(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """发送一条携带结构化字段的日志.

    例: ``log_kv(log, logging.INFO, "split done", split="train", n=1234)``
    在文本 handler 中输出 ``... | split done`` (字段无侵入), 在 JSON handler 中
    每行 JSON 自动包含 ``split`` 与 ``n`` 字段, 便于离线分析.
    """
    logger.log(level, message, extra=fields)


def to_jsonl(payload: Mapping[str, Any]) -> str:
    """将一段结构化数据序列化为单行 JSON 字符串 (CLI stdout 使用)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
