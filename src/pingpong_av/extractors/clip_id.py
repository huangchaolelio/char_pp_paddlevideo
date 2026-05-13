"""clip_id 工具: 为任意视频文件计算稳定的 32-hex 内容 hash.

对应:
    - spec clarify Q1 (Session 2026-05-13): clip_id = sha256(file_bytes)[:32]
    - FR-034 幂等性 / FR-043 GT url 重写 / SC-013 跨机器一致性
    - data-model.md RawVideo.clip_id 字段

**为什么用文件内容 hash 而不是文件名**:
    - 跨机器一致 (SC-013): 同一视频在任何机器上 hash 相同
    - 抗改名: 用户自由改名不触发重抽特征, 节省 GPU 时间
    - 抗冲突: sha256 空间 2^256, 实质不可能碰撞
    - 审计链: manifest.csv 的 clip_id 是视频字节的数字签名

**为什么取 32-hex 前缀而不是完整 64-hex**:
    与 ``Features_competition_train.tar.gz`` 内部 .pkl 命名布局一致
    (上游命名是 32-hex). 32-hex = 128 bit ≈ 3.4e38 种可能, 足够唯一.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["compute_clip_id", "CLIP_ID_LENGTH"]


#: clip_id 字符长度 (32 个 16 进制字符 = 16 字节 = 128 bit)
CLIP_ID_LENGTH = 32

#: 流式读取的块大小 (1 MB). 大 1GB+ 视频不会 OOM.
_HASH_CHUNK_BYTES = 1 << 20


def compute_clip_id(video_path: Path | str, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """对视频文件做流式 sha256 hash, 返回前 32 个 16 进制字符.

    Args:
        video_path: 视频文件路径 (必须存在且可读).
        chunk_bytes: 流式读取块大小 (默认 1 MB); 用于防止大文件 OOM.
                     测试时可传小值验证流式逻辑.

    Returns:
        32-char 小写十六进制字符串, 例如 ``"0018d6cbdf1f43f1a8a6d801b847f326"``.

    Raises:
        FileNotFoundError: ``video_path`` 不存在.
        IsADirectoryError: ``video_path`` 指向目录.
        PermissionError:   文件存在但不可读.

    Example:
        >>> from pathlib import Path
        >>> cid = compute_clip_id(Path("tests/fixtures/mini_pingpong_5s.mp4"))
        >>> len(cid)
        32
        >>> all(c in "0123456789abcdef" for c in cid)
        True
    """
    path = Path(video_path)
    # Path.open 在不存在/无权限时会抛合适的异常, 不需要预检
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()[:CLIP_ID_LENGTH]
