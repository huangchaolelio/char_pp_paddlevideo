"""原始视频 → 2048-d 特征 pkl 抽特征子系统 (002 feature).

子模块:
    :mod:`clip_id`        — sha256(file_bytes)[:32] 稳定 hash 工具
    :mod:`ffmpeg_frames`  — ffmpeg 抽帧 + fps 探测 (T204)
    :mod:`manifest`       — manifest.csv 线程安全写出器 (T205)
    :mod:`pp_tsm_inference` — PP-TSM 静态图 Predictor 包装 (T207)

与 :mod:`pingpong_av.models.pp_tsm` (训练用, 动态图) **严格分离**:
本模块只做 **inference-time** 抽特征, 调用 ``paddle.inference.Predictor`` 走静态图路径,
完全不 import 训练相关 dataloader / optimizer / loss.

设计参考:
    - 上游 ``applications/FootballAction/extractor/extract_feat.py``
    - 上游 ``applications/TableTennis/extractor/configs/configs.yaml``
    - research.md R10/R11/R12/R13

章程对齐:
    - III (配置驱动): 参数全部从 ``configs/models/pp_tsm_extractor.yaml`` 读
    - IV  (数据完整性): clip_id = sha256(file_bytes)[:32], 跨机器一致
    - VI  (上游最小侵入): 只通过上游公共 API (``paddle.inference``, ``paddlevideo.modeling.builder``), 不改源码
"""

from __future__ import annotations

__all__ = [
    "compute_clip_id",
    "extract_frames_to_dir",
    "probe_video_metadata",
    "FramesResult",
    "FFmpegError",
    "ManifestRow",
    "ManifestWriter",
    "MANIFEST_COLUMNS",
    "PPTSMExtractor",
    "PPTSMExtractorError",
    "ExtractorConfig",
]

from pingpong_av.extractors.clip_id import compute_clip_id
from pingpong_av.extractors.ffmpeg_frames import (
    FFmpegError,
    FramesResult,
    extract_frames_to_dir,
    probe_video_metadata,
)
from pingpong_av.extractors.manifest import (
    MANIFEST_COLUMNS,
    ManifestRow,
    ManifestWriter,
)
from pingpong_av.extractors.pp_tsm_inference import (
    ExtractorConfig,
    PPTSMExtractor,
    PPTSMExtractorError,
)
