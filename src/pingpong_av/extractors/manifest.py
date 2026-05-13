"""manifest.csv 写出器 (T205).

对应:
    - research.md R12 (manifest.csv 字段表, 13 列)
    - FR-034 (幂等性) / FR-048 (透传审计元信息) / FR-049 (临时目录管理)
    - data-model.md ImageFeaturePkl.pkl_sha256 + pp_tsm_weight_sha256 + extraction_commit

设计:
    - 线程安全 (US2 ``pp build-feature-pkls --workers N`` 可能并行抽)
    - append-only (断点续抽时不会重写已有行)
    - CSV 标准库, 不引入新依赖
    - 列顺序固定 (便于下游工具消费)
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from pingpong_av.utils.logging import get_logger

__all__ = ["ManifestRow", "ManifestWriter", "MANIFEST_COLUMNS"]

_log = get_logger(__name__)


#: manifest.csv 列顺序 (对应 research.md R12 表, 保持稳定以便下游脚本硬编码列位)
MANIFEST_COLUMNS: tuple[str, ...] = (
    "video_path",
    "clip_id",
    "n_frames",
    "fps_original",
    "fps_used",
    "duration_sec",
    "pkl_path",
    "pkl_sha256",
    "pp_tsm_weight_sha256",
    "pp_tsm_inference_sha256",
    "pp_tsm_config_hash",
    "extraction_commit",
    "extracted_at",
    "error",
)


@dataclass
class ManifestRow:
    """一行 manifest 记录. 所有字段都有默认值, 空值写空字符串."""

    video_path:               str = ""
    clip_id:                  str = ""
    n_frames:                 int = 0
    fps_original:             float = 0.0
    fps_used:                 int = 0
    duration_sec:             float = 0.0
    pkl_path:                 str = ""
    pkl_sha256:               str = ""
    pp_tsm_weight_sha256:     str = ""
    pp_tsm_inference_sha256:  str = ""
    pp_tsm_config_hash:       str = ""
    extraction_commit:        str = ""
    extracted_at:             str = ""         # ISO8601 UTC
    error:                    str = ""         # 空 = 成功

    def to_csv_dict(self) -> dict[str, Any]:
        """转成 CSV DictWriter 接受的字典 (列序由 MANIFEST_COLUMNS 保证)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


class ManifestWriter:
    """线程安全的 manifest.csv append 写出器.

    使用:
        >>> writer = ManifestWriter("manifest.csv")
        >>> writer.append(ManifestRow(clip_id="abc...", ...))
        >>> writer.close()

    或者作为 context manager:
        >>> with ManifestWriter("manifest.csv") as writer:
        ...     writer.append(ManifestRow(...))

    幂等性:
        - 如果 ``manifest.csv`` 已存在, 默认**追加** (不重写 header).
        - 调用 ``writer.get_existing_clip_ids()`` 可拿到已写入的 clip_id 集合,
          让调用方在抽特征前跳过已处理的视频 (FR-034).
    """

    def __init__(self, csv_path: Path | str) -> None:
        self._path = Path(csv_path)
        self._lock = threading.Lock()
        self._file = None          # type: ignore[assignment]
        self._writer: csv.DictWriter | None = None
        self._open_file_and_writer()

    # ---- lifecycle ----

    def _open_file_and_writer(self) -> None:
        """打开 CSV 文件 (追加模式); 若文件不存在则创建并写 header."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        # newline='' 按 CSV 规范避免 Windows 下额外换行
        self._file = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(MANIFEST_COLUMNS))
        if needs_header:
            self._writer.writeheader()
            self._file.flush()
        _log.info(
            "manifest opened",
            extra={"path": str(self._path), "appended": not needs_header},
        )

    def append(self, row: ManifestRow) -> None:
        """线程安全地追加一行.

        文件末尾 flush 保证 Ctrl-C 时已写的行不丢; 性能代价是每行一次系统调用, 对
        几百/几千视频规模可接受.
        """
        if self._writer is None:
            raise RuntimeError("ManifestWriter already closed")
        with self._lock:
            self._writer.writerow(row.to_csv_dict())
            # flush 保证崩溃时已写的行不丢
            self._file.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        """关闭文件."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._writer = None

    # ---- 幂等性辅助 ----

    def get_existing_clip_ids(self) -> set[str]:
        """从 CSV 读已写入的 clip_id 集合, 用于跳过已处理视频 (FR-034).

        使用前调用: 在 append 之前扫描; 读的时候不持 lock (允许并行 append).
        """
        if not self._path.exists():
            return set()
        # 单独打开只读, 不影响 append 句柄
        out: set[str] = set()
        with self._path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("clip_id", "").strip()
                if cid:
                    out.add(cid)
        return out

    # ---- context manager ----

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
