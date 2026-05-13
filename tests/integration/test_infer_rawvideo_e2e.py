"""T216: pp infer-rawvideo 端到端集成测试 (002 feature, US1 MVP).

策略:
    - 标记 @pytest.mark.slow + @pytest.mark.gpu — 默认 CI 跳过, 用 `pytest --runslow` 启用
    - 需要前置: 已下载 ppTSM_k400_dense.pdparams + 已导出 inference 双文件
    - **不**需要真实 BMN ckpt 的有效预测 — 验证的是 pipeline 走通 + schema 正确
      (随机权重 BMN 也能产出合法 timeline.json, 只是预测无意义)

如何运行:
    .venv/bin/pytest tests/integration/test_infer_rawvideo_e2e.py --runslow -v

需要的环境前置 (Phase 2 已完成的话都齐了):
    - tests/fixtures/mini_pingpong_5s.mp4
    - data/raw/pretrained/ppTSM_k400_dense.pdparams (~148 MB, 用户手动下载)
    - data/raw/pretrained/ppTSM.{pdmodel,pdiparams} (export 脚本派生)
    - 一份 BMN ckpt (使用 v0.2.x 训练产物; 或自动跳过测试)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini_pingpong_5s.mp4"
PPTSM_PDPARAMS = REPO_ROOT / "data" / "raw" / "pretrained" / "ppTSM_k400_dense.pdparams"
PPTSM_PDMODEL = REPO_ROOT / "data" / "raw" / "pretrained" / "ppTSM.pdmodel"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PP_CLI = REPO_ROOT / ".venv" / "bin" / "pp"


def _find_bmn_ckpt() -> Path | None:
    """找一个本仓库的 BMN ckpt (v0.2.x 训练产物). 没有就返回 None."""
    pattern_dirs = list((REPO_ROOT / "experiments").glob("*-train-bmn_pingpong"))
    for run_dir in sorted(pattern_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        ckpts = sorted(run_dir.glob("BMN_epoch_*.pdparams"))
        if ckpts:
            return ckpts[-1]   # 最大 epoch
    return None


@pytest.fixture(scope="module")
def bmn_ckpt():
    """提供一个可用的 BMN ckpt; 缺失则 skip."""
    ckpt = _find_bmn_ckpt()
    if ckpt is None:
        pytest.skip("无 BMN ckpt (experiments/*-bmn_pingpong/BMN_epoch_*.pdparams); "
                    "请先跑 v0.2.x 训练.")
    return ckpt


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_extract_feat_e2e_smoke(tmp_path):
    """SC-011 部分验证: 跑 pp extract-feat 在 fixture 上, 检查 pkl 产物 + meta.json."""
    if not FIXTURE.is_file():
        pytest.skip("fixture mp4 缺失")
    if not PPTSM_PDPARAMS.is_file():
        pytest.skip("ppTSM_k400_dense.pdparams 缺失 (下载 ~148MB)")
    if not PPTSM_PDMODEL.is_file():
        pytest.skip("ppTSM.pdmodel 缺失 (运行 scripts/export_pptsm_inference.py)")
    if not PP_CLI.is_file():
        pytest.skip(".venv/bin/pp 不存在 (运行 bash scripts/bootstrap.sh)")

    out_pkl = tmp_path / "feat.pkl"
    cmd = [
        str(PP_CLI), "extract-feat",
        "--input", str(FIXTURE),
        "--output", str(out_pkl),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"pp extract-feat 失败 rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    assert out_pkl.is_file(), "未产出 .pkl"
    meta_p = out_pkl.with_suffix(".meta.json")
    assert meta_p.is_file(), "未产出 .meta.json"

    # 验证 meta schema
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["schema"] == "extract-feat-meta-v1"
    assert meta["n_frames"] == 125    # fixture 是 5s × 25fps
    assert meta["feat_dim"] == 2048    # FR-037 硬约束
    assert meta["n_samples"] in (15, 16)  # 125 / 8 ≈ 16
    assert len(meta["clip_id"]) == 32

    # 验证 stdout 是合法 JSON
    summary = json.loads(result.stdout.strip().split("\n")[-1])
    assert summary["schema"] == "extract-feat-v1"
    assert summary["clip_id"] == meta["clip_id"]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_infer_rawvideo_e2e_smoke(tmp_path, bmn_ckpt):
    """US1 MVP e2e: pp infer-rawvideo 在 fixture + 真实 BMN ckpt 上.

    断言:
        - 退出码 0
        - timeline.json 存在且 schema = rawvideo-timeline-v1
        - feature.pkl 存在 (因 --keep-features default True)
        - bmn_eval/results/bmn_results_validation.json 存在
        - visualized.mp4 存在 (除非可视化模块失败, 已被 cli 容错)
    """
    if not FIXTURE.is_file():
        pytest.skip("fixture mp4 缺失")
    if not PPTSM_PDPARAMS.is_file():
        pytest.skip("ppTSM_k400_dense.pdparams 缺失")
    if not PPTSM_PDMODEL.is_file():
        pytest.skip("ppTSM.pdmodel 缺失 (运行 scripts/export_pptsm_inference.py)")
    if not PP_CLI.is_file():
        pytest.skip(".venv/bin/pp 不存在")

    out_dir = tmp_path / "infer_out"
    cmd = [
        str(PP_CLI), "infer-rawvideo",
        "--input", str(FIXTURE),
        "--bmn-checkpoint", str(bmn_ckpt),
        "--output-dir", str(out_dir),
        "--no-visualize",   # 跳过可视化省时间 (cv2 IO)
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print("STDOUT:", result.stdout[-2000:])
    print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"pp infer-rawvideo 失败 rc={result.returncode}"

    # 验证主产物
    timeline_p = out_dir / "timeline.json"
    assert timeline_p.is_file(), "未产出 timeline.json"

    timeline = json.loads(timeline_p.read_text(encoding="utf-8"))
    assert timeline["schema"] == "rawvideo-timeline-v1"
    assert "input_video_clip_id" in timeline
    assert "extraction" in timeline
    assert "bmn_inference" in timeline
    assert "results" in timeline
    assert isinstance(timeline["results"], list)

    # extraction 子对象
    ext = timeline["extraction"]
    assert ext["fps_used"] == 25
    assert ext["n_frames"] == 125
    assert ext["feat_dim"] == 2048

    # bmn_inference 子对象
    bmn = timeline["bmn_inference"]
    assert bmn["checkpoint"]
    assert len(bmn["checkpoint_sha256"]) == 64   # full sha256
    assert bmn["ar_at"] is None                  # 推理时无 GT

    # feature.pkl 存在 (--keep-features default ON)
    assert (out_dir / "feature.pkl").is_file()

    # bmn 中间产物存在
    assert (out_dir / "bmn_eval" / "results" / "bmn_results_validation.json").is_file()


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.integration
def test_infer_rawvideo_threshold_filter(tmp_path, bmn_ckpt):
    """验证 --threshold 过滤逻辑: 设 0.99 → 应过滤掉绝大多数 proposal."""
    if not all(p.is_file() for p in [FIXTURE, PPTSM_PDPARAMS, PPTSM_PDMODEL, PP_CLI]):
        pytest.skip("前置文件缺失")

    out_dir = tmp_path / "infer_out_high_thresh"
    cmd = [
        str(PP_CLI), "infer-rawvideo",
        "--input", str(FIXTURE),
        "--bmn-checkpoint", str(bmn_ckpt),
        "--output-dir", str(out_dir),
        "--threshold", "0.99",
        "--no-visualize",
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0

    timeline = json.loads((out_dir / "timeline.json").read_text(encoding="utf-8"))
    # 严格阈值下 results 数量应远少于 BMN 默认输出 (一般为 0 或个位数)
    # 不强制为 0 (随机权重可能偶尔 ≥ 0.99), 只要求合法即可
    for r in timeline["results"]:
        assert r["score"] >= 0.99, f"score {r['score']} 应 ≥ threshold"
