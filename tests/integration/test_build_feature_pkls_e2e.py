"""T220: pp build-feature-pkls 端到端集成测试 (002 feature, US2).

策略 (与 test_infer_rawvideo_e2e.py 一致):
    - @pytest.mark.slow + @pytest.mark.gpu, 默认 CI 跳过 (用 --runslow)
    - 用 fixture 的 3 份 copy 模拟 "批量" 场景
    - 覆盖:
      (a) 基础场景: 无 --gt-json, 只产 pkl + manifest.csv
      (b) 幂等性: 第二次跑, 已存在的 pkl 被跳过
      (c) 带 --gt-json: url 字段正确重写
      (d) --gt-json 中 url 缺失 → 退出码 1 + 列出缺失项

需要的前置:
    - tests/fixtures/mini_pingpong_5s.mp4
    - ppTSM_k400_dense.pdparams + ppTSM.pdmodel/.pdiparams (与 Phase 2 一致)
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini_pingpong_5s.mp4"
PPTSM_PDMODEL = REPO_ROOT / "data" / "raw" / "pretrained" / "ppTSM.pdmodel"
PP_CLI = REPO_ROOT / ".venv" / "bin" / "pp"


@pytest.fixture
def videos_dir_with_3_copies(tmp_path):
    """用 fixture 复制 3 份 (不同文件名, 相同内容) 模拟批量场景.

    注意: 3 份文件**内容完全相同**, clip_id 也完全相同, 所以幂等测试能保证
    只处理 1 个, 剩下 2 个被跳过.

    如果要测"3 份不同视频", 得用不同的 ffmpeg 合成命令, 下一版再加.
    """
    if not FIXTURE.is_file():
        pytest.skip("fixture mp4 缺失")
    src_dir = tmp_path / "videos"
    src_dir.mkdir()
    for i in range(3):
        shutil.copy(FIXTURE, src_dir / f"video_{i}.mp4")
    return src_dir


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_build_feature_pkls_basic(videos_dir_with_3_copies, tmp_path):
    """基础场景: 无 GT JSON, 3 份 fixture → 1 处理 + 2 幂等跳过 (同内容同 clip_id).

    注意: 由于 3 份 mp4 内容完全相同, clip_id 都一样, 第一个视频处理后
    后面 2 个会因 pkl 已存在被跳过. 这是**预期行为** — 证明 FR-034 抗重复.
    """
    if not PPTSM_PDMODEL.is_file() or not PP_CLI.is_file():
        pytest.skip("PP-TSM inference 模型或 pp cli 缺失")

    out_dir = tmp_path / "build_out"
    cmd = [
        str(PP_CLI), "build-feature-pkls",
        "--videos-dir", str(videos_dir_with_3_copies),
        "--output-dir", str(out_dir),
        "--name", "mytest",
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print("STDOUT:", result.stdout[-1500:])
    print("STDERR:", result.stderr[-1500:])
    assert result.returncode == 0, f"失败 rc={result.returncode}"

    # 产物校验
    features_dir = out_dir / "Features_mytest"
    assert features_dir.is_dir()

    # 所有 3 份视频内容一样 → 只有 1 个 .pkl
    pkls = list(features_dir.glob("*.pkl"))
    assert len(pkls) == 1, f"预期 1 个 pkl (内容相同), 实际 {len(pkls)}"
    assert len(pkls[0].stem) == 32   # clip_id 长度

    # manifest.csv: 3 行 (1 处理 + 2 跳过)
    manifest = out_dir / "manifest.csv"
    assert manifest.is_file()
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3, f"预期 3 manifest 行, 实际 {len(rows)}"
    n_skipped = sum(1 for r in rows if "skipped" in r.get("error", ""))
    assert n_skipped == 2, f"预期 2 跳过, 实际 {n_skipped}"

    # stdout JSON 摘要
    lines = result.stdout.strip().split("\n")
    summary = json.loads(lines[-1])
    assert summary["schema"] == "build-feature-pkls-v1"
    assert summary["n_videos_total"] == 3
    assert summary["n_videos_processed"] + summary["n_videos_skipped"] == 3
    assert summary["label_json_path"] is None      # 没传 --gt-json


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_build_feature_pkls_idempotent(videos_dir_with_3_copies, tmp_path):
    """幂等性 (FR-034): 第二次跑全部 3 份都被跳过."""
    if not PPTSM_PDMODEL.is_file() or not PP_CLI.is_file():
        pytest.skip("前置文件缺失")

    out_dir = tmp_path / "build_out"
    cmd_base = [
        str(PP_CLI), "build-feature-pkls",
        "--videos-dir", str(videos_dir_with_3_copies),
        "--output-dir", str(out_dir),
        "--name", "mytest",
        "--allow-dirty",
    ]

    # 第一次跑
    r1 = subprocess.run(cmd_base, capture_output=True, text=True, timeout=300)
    assert r1.returncode == 0

    # 第二次跑
    r2 = subprocess.run(cmd_base, capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0
    lines = r2.stdout.strip().split("\n")
    summary2 = json.loads(lines[-1])
    assert summary2["n_videos_skipped"] == 3, (
        f"第二次应 3 全跳过, 实际 processed={summary2['n_videos_processed']} "
        f"skipped={summary2['n_videos_skipped']}"
    )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_build_feature_pkls_with_gt_json(videos_dir_with_3_copies, tmp_path):
    """带 --gt-json: url 字段必须被重写为 <clip_id>.mp4."""
    if not PPTSM_PDMODEL.is_file() or not PP_CLI.is_file():
        pytest.skip("前置文件缺失")

    # 构造一个最小 GT JSON 指向 video_0.mp4
    gt_input = tmp_path / "my_label.json"
    gt_data = {
        "fps": 25,
        "gts": [
            {
                "url": "video_0.mp4",
                "total_frames": 125,
                "actions": [
                    {"label_names": ["摆短"], "label_ids": [0],
                     "start_id": 1.0, "end_id": 2.0}
                ],
            }
        ],
    }
    gt_input.write_text(json.dumps(gt_data, ensure_ascii=False), encoding="utf-8")

    out_dir = tmp_path / "build_out"
    cmd = [
        str(PP_CLI), "build-feature-pkls",
        "--videos-dir", str(videos_dir_with_3_copies),
        "--output-dir", str(out_dir),
        "--gt-json", str(gt_input),
        "--name", "mytest",
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"失败: {result.stderr[-500:]}"

    # label_cls14_mytest.json 存在
    label_path = out_dir / "label_cls14_mytest.json"
    assert label_path.is_file()

    # url 已重写为 clip_id
    new_gt = json.loads(label_path.read_text(encoding="utf-8"))
    assert len(new_gt["gts"]) == 1
    new_url = new_gt["gts"][0]["url"]
    assert new_url.endswith(".mp4")
    cid_part = new_url[:-4]
    assert len(cid_part) == 32
    assert cid_part != "video_0"       # 已经不是原 stem

    # 原 actions 保留
    assert new_gt["gts"][0]["actions"][0]["label_ids"] == [0]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_build_feature_pkls_gt_missing_url_rejected(videos_dir_with_3_copies, tmp_path):
    """边界情况 (FR-043): GT JSON url 在 videos-dir 找不到, 退出码 1 + 列出缺失."""
    if not PPTSM_PDMODEL.is_file() or not PP_CLI.is_file():
        pytest.skip("前置文件缺失")

    # GT 指向一个不存在的视频名
    gt_input = tmp_path / "bad_label.json"
    gt_data = {
        "fps": 25,
        "gts": [{"url": "this_video_does_not_exist.mp4", "actions": []}],
    }
    gt_input.write_text(json.dumps(gt_data, ensure_ascii=False), encoding="utf-8")

    out_dir = tmp_path / "build_out"
    cmd = [
        str(PP_CLI), "build-feature-pkls",
        "--videos-dir", str(videos_dir_with_3_copies),
        "--output-dir", str(out_dir),
        "--gt-json", str(gt_input),
        "--name", "bad",
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, f"应退出码 1, 实际 {result.returncode}"
    assert "this_video_does_not_exist.mp4" in result.stderr
    assert "找不到" in result.stderr or "缺失" in result.stderr
