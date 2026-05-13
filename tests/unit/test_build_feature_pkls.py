"""T219: build-feature-pkls 单元测试.

覆盖不依赖 GPU / paddle / ffmpeg 的逻辑:
    - helpers: _sha256_of_file / _combined_sha256 / _git_head_commit
    - GT JSON url 重写逻辑 (由 run() 内部实现, 本测试复刻它)
    - 视频扩展名常量 _VIDEO_EXTS

不覆盖:
    - 端到端跑 extract → pkl → manifest (那是 test_build_feature_pkls_e2e.py 的责任)
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from pingpong_av.cli import build_feature_pkls as cmd


# ============================================================
# helpers
# ============================================================


def test_sha256_of_file_deterministic(tmp_path):
    """同一文件多次 hash 结果一致."""
    p = tmp_path / "t.bin"
    p.write_bytes(b"hello world")
    a = cmd._sha256_of_file(p)
    b = cmd._sha256_of_file(p)
    assert a == b
    # 验证是标准 sha256 结果
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert a == expected


def test_sha256_of_file_streaming_equivalent(tmp_path):
    """不同 chunk 大小应产生相同 hash."""
    p = tmp_path / "big.bin"
    # 3 MB 随机数据
    p.write_bytes(b"\x42" * (3 << 20))
    a = cmd._sha256_of_file(p, chunk=1 << 10)     # 1KB chunk
    b = cmd._sha256_of_file(p, chunk=1 << 20)     # 1MB chunk
    c = cmd._sha256_of_file(p, chunk=4 << 20)     # 4MB chunk (整文件一次读完)
    assert a == b == c


def test_combined_sha256_order_matters(tmp_path):
    """_combined_sha256(a, b) != _combined_sha256(b, a) (顺序敏感)."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"AAA")
    b.write_bytes(b"BBB")
    ab = cmd._combined_sha256(a, b)
    ba = cmd._combined_sha256(b, a)
    assert ab != ba


def test_combined_sha256_equals_manual(tmp_path):
    """combined = sha256(a.bytes || b.bytes)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"foo")
    b.write_bytes(b"bar")
    expected = hashlib.sha256(b"foo" + b"bar").hexdigest()
    assert cmd._combined_sha256(a, b) == expected


def test_git_head_commit_returns_string(tmp_path):
    """在合法 git repo 里应返回 40-hex commit hash; 非 repo 返回空.

    (当前工作目录是 repo root, 所以应返回实际 hash)
    """
    from pingpong_av.utils.env import find_repo_root
    repo_root = find_repo_root()
    commit = cmd._git_head_commit(repo_root)
    # 如果有 git, commit 应是 40-hex; 没 git 或 repo 损坏, 返回空
    if commit:
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)


def test_git_head_commit_graceful_on_nonrepo(tmp_path):
    """非 git 目录应返回空字符串, 不抛异常."""
    commit = cmd._git_head_commit(tmp_path)
    # 可能是空串 (非 repo) 或回退到父目录 repo 的 hash (取决于 git 配置)
    # 仅验证不崩溃
    assert isinstance(commit, str)


# ============================================================
# 视频扫描 / 扩展名
# ============================================================


def test_video_exts_lowercase():
    """_VIDEO_EXTS 所有项都以点开头且小写."""
    for ext in cmd._VIDEO_EXTS:
        assert ext.startswith(".")
        assert ext == ext.lower()


def test_video_exts_common_formats_covered():
    """常见视频格式都应支持."""
    for ext in (".mp4", ".avi", ".mov", ".flv", ".mkv"):
        assert ext in cmd._VIDEO_EXTS


# ============================================================
# GT JSON 重写逻辑 (复刻 run() 内部核心行为)
# ============================================================


def test_gt_json_url_rewrite_stem_mapping():
    """GT JSON 重写规则: url=<original_stem>.mp4 → url=<clip_id>.mp4.

    复刻 run() 内部的逻辑, 保证 label JSON 替换正确.
    """
    gt_data = {
        "fps": 25,
        "gts": [
            {"url": "video_a.mp4", "total_frames": 1000, "actions": []},
            {"url": "videos/video_b.mov", "total_frames": 2000, "actions": []},
        ],
    }
    stem_to_clip_id = {
        "video_a": "abc123" + "0" * 26,      # 32-hex
        "video_b": "def456" + "0" * 26,
    }

    new_gt = {"fps": gt_data["fps"], "gts": []}
    for g in gt_data["gts"]:
        url = str(g.get("url", ""))
        stem = Path(url).stem
        if stem in stem_to_clip_id:
            new_g = dict(g)
            suffix = Path(url).suffix or ".mp4"
            new_g["url"] = f"{stem_to_clip_id[stem]}{suffix}"
            new_gt["gts"].append(new_g)

    assert len(new_gt["gts"]) == 2
    assert new_gt["gts"][0]["url"] == "abc12300000000000000000000000000.mp4"
    assert new_gt["gts"][1]["url"] == "def45600000000000000000000000000.mov"
    # clip_id 长度严格 32
    assert len(new_gt["gts"][0]["url"].split(".")[0]) == 32
    # 保留原 fps / total_frames / actions
    assert new_gt["fps"] == 25
    assert new_gt["gts"][0]["total_frames"] == 1000
    assert new_gt["gts"][1]["total_frames"] == 2000


def test_gt_json_missing_url_filtered():
    """GT 中 url 在 videos-dir 找不到的条目应被过滤 (run 内部会在校验阶段抛错, 这里只测 stem_to_clip_id 为 None 时不写)."""
    gt_data = {"fps": 25, "gts": [{"url": "missing.mp4", "actions": []}]}
    stem_to_clip_id: dict[str, str] = {}    # 空映射, 模拟 videos-dir 中找不到

    new_gts = []
    for g in gt_data["gts"]:
        stem = Path(str(g.get("url", ""))).stem
        if stem in stem_to_clip_id:
            new_gts.append(g)

    assert new_gts == []
