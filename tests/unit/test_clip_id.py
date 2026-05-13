"""T214: clip_id 单元测试 (002 feature).

验证:
    - 相同内容不同路径返回同 hash (FR-034 抗改名)
    - 不同内容返回不同 hash
    - 流式 hash 对大文件不 OOM (用小 chunk 验证流式逻辑等价性)
    - 输出格式: 32 个 16 进制小写字符

参考: tests/unit/test_splitter.py 风格.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pingpong_av.extractors.clip_id import CLIP_ID_LENGTH, compute_clip_id

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_pingpong_5s.mp4"


def test_fixture_exists():
    """前置: 测试 fixture 必须存在 (由 T202 入库)."""
    assert FIXTURE.is_file(), f"fixture 缺失: {FIXTURE}"


def test_clip_id_format():
    """clip_id 必须是 32 个小写 16 进制字符."""
    cid = compute_clip_id(FIXTURE)
    assert len(cid) == CLIP_ID_LENGTH == 32, f"长度: {len(cid)}"
    assert all(c in "0123456789abcdef" for c in cid), f"非 16 进制: {cid}"


def test_clip_id_deterministic():
    """同一文件多次调用必须返回相同 hash."""
    cid1 = compute_clip_id(FIXTURE)
    cid2 = compute_clip_id(FIXTURE)
    assert cid1 == cid2


def test_clip_id_streaming_equivalent(tmp_path):
    """小 chunk 与默认 chunk 必须返回相同结果 (验证流式正确性)."""
    cid_default = compute_clip_id(FIXTURE)
    cid_chunk128 = compute_clip_id(FIXTURE, chunk_bytes=128)
    cid_chunk1 = compute_clip_id(FIXTURE, chunk_bytes=1)
    assert cid_default == cid_chunk128 == cid_chunk1


def test_clip_id_rename_invariant(tmp_path):
    """改名后 hash 必须不变 (FR-034 抗改名核心)."""
    cid_original = compute_clip_id(FIXTURE)
    renamed = tmp_path / "totally_different_name.foo"
    shutil.copy(FIXTURE, renamed)
    cid_renamed = compute_clip_id(renamed)
    assert cid_original == cid_renamed, "改名后 hash 不应改变"


def test_clip_id_content_sensitive(tmp_path):
    """内容差 1 字节 hash 必须不同 (抗冲突)."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    # 完全相同
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert compute_clip_id(a) == compute_clip_id(b)
    # 差 1 byte
    a.write_bytes(b"hello")
    b.write_bytes(b"hellp")  # 'o' → 'p'
    assert compute_clip_id(a) != compute_clip_id(b)


def test_clip_id_missing_file(tmp_path):
    """文件不存在应抛 FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compute_clip_id(tmp_path / "nonexistent.mp4")


def test_clip_id_fixture_golden():
    """Fixture clip_id 是确定性合成 (testsrc2), 与 T203 实测值一致.

    这个 golden value 跨机器应该一致 (假设 ffmpeg + libx264 同版本编码).
    若失败说明 fixture 被改了或 ffmpeg 行为变了, 需要重新生成 fixture.
    """
    cid = compute_clip_id(FIXTURE)
    # 实测值 from Phase 2 T203 sanity test
    expected = "bf10fdc237533e8d943bcff1a5434597"
    assert cid == expected, (
        f"fixture clip_id 漂移: {cid} != {expected}. "
        f"若 ffmpeg 版本变更导致, 请按 tests/fixtures/README.md 重新生成 fixture, "
        f"并更新此 golden 值."
    )
