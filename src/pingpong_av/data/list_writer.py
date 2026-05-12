"""把 :class:`Splits` 序列化为 PaddleVideo 兼容的 list 文件 + meta jsonl.

输出文件 (data-model.md `DatasetSplit` 一节):

    data/splits/<split>.txt          —— PaddleVideo 训练入口直接读取的 list (path<TAB>label_id)
    data/splits/<split>.meta.jsonl   —— 每行一个 VideoClip 的完整 JSON, 供本项目后续回溯

list 格式由上游 `paddlevideo.loader.dataset` 决定; 此处保持最广通用的两列形式.

**章程 IV 关键事实**: 这两类文件**必须入库** (.gitignore 已放行 data/splits/).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pingpong_av.data.splitter import Splits, VideoClip
from pingpong_av.utils.logging import get_logger

__all__ = ["write_paddlevideo_lists"]

_log = get_logger(__name__)


def write_paddlevideo_lists(
    splits: Splits,
    out_dir: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> dict[str, Path]:
    """落盘三份 split 的 list 文件与 meta jsonl 文件.

    参数:
        splits: 划分结果.
        out_dir: 输出目录, 通常是 ``data/splits/``. 若不存在则创建.
        relative_to: 若提供, 把 ``clip.path`` 转为相对 ``relative_to`` 的路径写入 list;
                     传 None 则原样写入 (假定 clip.path 已是相对/绝对路径中合适的形式).

    返回:
        ``{"train.txt": Path, "val.txt": Path, "test.txt": Path,
            "train.meta.jsonl": ..., "val.meta.jsonl": ..., "test.meta.jsonl": ...}``

    异常:
        ValueError: split 中含 label_id < 0 (unknown 不应进入训练 list); 或 path 为空.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_root: Path | None = None
    if relative_to is not None:
        rel_root = Path(relative_to).resolve()

    written: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        clips: list[VideoClip] = list(getattr(splits, split_name))

        # ---- 校验 ----
        for c in clips:
            if c.label_id is None or c.label_id < 0:
                raise ValueError(
                    f"split={split_name} 中存在 label_id<0 的样本 ({c.clip_id}); "
                    "训练 list 不应包含 unknown 类别 (那是推理产物, 见 data-model.md)."
                )
            if not c.path:
                raise ValueError(f"split={split_name} 中存在空 path 的样本 ({c.clip_id})")

        # ---- list 文件 (PaddleVideo 兼容: path<TAB>label_id) ----
        list_path = out_dir / f"{split_name}.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for c in clips:
                p = _as_str_path(c.path, rel_root)
                f.write(f"{p}\t{c.label_id}\n")
        written[f"{split_name}.txt"] = list_path

        # ---- meta jsonl (每行完整 VideoClip dict) ----
        meta_path = out_dir / f"{split_name}.meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for c in clips:
                f.write(json.dumps(asdict(c), ensure_ascii=False, sort_keys=True) + "\n")
        written[f"{split_name}.meta.jsonl"] = meta_path

    _log.info(
        "list files written",
        extra={
            "out_dir": str(out_dir),
            "files": {k: str(v) for k, v in written.items()},
            "counts": splits.counts(),
        },
    )
    return written


def read_meta_jsonl(path: str | Path) -> list[VideoClip]:
    """读取 meta.jsonl 还原 VideoClip 列表 (用于校验或重新生成 list)."""
    p = Path(path)
    out: list[VideoClip] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(VideoClip(**d))
    return out


def _as_str_path(path: str | Path, rel_root: Path | None) -> str:
    p = Path(path)
    if rel_root is None:
        return str(p)
    if p.is_absolute():
        try:
            return str(p.relative_to(rel_root))
        except ValueError:
            return str(p)  # 不在 rel_root 下, 保持原样
    return str(p)
