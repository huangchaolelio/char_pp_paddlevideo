"""``pp build-feature-pkls`` 子命令 (T217, FR-034 / FR-043 / FR-044).

批量 mp4 → Features_<name>/<clip_id>.pkl + (可选) 重写 GT JSON.

设计 (contracts/cli.md):
    1. 扫 ``--videos-dir`` 递归找所有 mp4/avi/mov/flv/mkv
    2. 可选校验 ``--gt-json``: 每个 url 必须在 videos-dir 中存在, 否则退出码 1
    3. 对每个视频:
       a. 算 clip_id = sha256(file_bytes)[:32]
       b. 检查 ``<out>/Features_<name>/<clip_id>.pkl`` 是否已存在 → 幂等跳过 (FR-034)
       c. 否则调 extract_feat 等价流程 (ffmpeg + PP-TSM) → 写 pkl
       d. append manifest.csv 一行 (13 列, research.md R12)
    4. 如 ``--gt-json`` 提供: 最后写 ``<out>/label_cls14_<name>.json``, 把 url 字段
       替换为 ``<clip_id>.mp4`` (FR-043)

退出码 (FR-047):
    0  全部成功 (含跳过 + 个别失败)
    1  用户输入错 (videos-dir 不存在 / gt-json url 缺失视频)
    2  环境问题 (ffmpeg / 权重缺失)
    3  保留
    4  运行时失败 (磁盘满等)

JSON stdout (build-feature-pkls-v1 schema): 见 contracts/cli.md.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from pingpong_av.extractors import (
    ExtractorConfig,
    FFmpegError,
    ManifestRow,
    ManifestWriter,
    PPTSMExtractor,
    PPTSMExtractorError,
    compute_clip_id,
    extract_frames_to_dir,
)
from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)

#: 支持的视频扩展名 (ffmpeg 可解的常见容器)
_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".flv", ".mkv", ".webm", ".m4v")


def run(
    *,
    videos_dir: str,
    output_dir: str,
    gt_json: str | None,
    name: str | None,
    workers: int,
    config_path: str,
    allow_dirty: bool,
    force: bool,
) -> int:
    """`pp build-feature-pkls` 入口. 返回退出码."""
    t_start = time.time()

    # ---- 1. 校验输入 ----
    videos_p = Path(videos_dir).resolve()
    if not videos_p.is_dir():
        click.echo(f"ERROR: videos-dir 不存在或非目录: {videos_p}", err=True)
        return 1

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 推断 dataset 子集名
    subset_name = name or videos_p.name
    features_dir = out_root / f"Features_{subset_name}"
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"

    # ---- 2. 加载业务 config ----
    cfg_p = Path(config_path)
    if not cfg_p.is_file():
        click.echo(f"ERROR: 配置不存在: {cfg_p}", err=True)
        return 1
    try:
        loaded = load_config(cfg_p)
    except ConfigError as exc:
        click.echo(f"ERROR: 配置校验失败: {exc}", err=True)
        return 1

    cfg = loaded.data
    config_hash = loaded.config_hash
    ext_cfg = ExtractorConfig.from_yaml_dict(cfg.get("extraction") or {})

    # ---- 3. 扫视频 ----
    all_videos = sorted(
        p for p in videos_p.rglob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )
    if not all_videos:
        click.echo(f"ERROR: videos-dir 内无视频文件 (扩展名 {_VIDEO_EXTS}): {videos_p}", err=True)
        return 1

    _log.info(
        "video scan",
        extra={"videos_dir": str(videos_p), "n_videos": len(all_videos),
               "subset_name": subset_name},
    )

    # ---- 4. 可选: 解析 GT JSON, 预校验 url → 视频映射 ----
    gt_map: dict[str, Path] | None = None   # video_name_stem → local Path
    gt_data: dict[str, Any] | None = None
    if gt_json:
        gt_p = Path(gt_json).resolve()
        if not gt_p.is_file():
            click.echo(f"ERROR: --gt-json 不存在: {gt_p}", err=True)
            return 1
        try:
            gt_data = json.loads(gt_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            click.echo(f"ERROR: --gt-json 不是合法 JSON: {exc}", err=True)
            return 1

        # 构造 videos-dir 内"stem → Path" 索引 (按 mp4 文件名去后缀)
        videos_by_stem: dict[str, Path] = {}
        for v in all_videos:
            videos_by_stem[v.stem] = v

        # 校验每个 url 能找到对应视频
        gt_map = {}
        missing: list[str] = []
        for g in gt_data.get("gts", []):
            url = str(g.get("url", ""))
            stem = Path(url).stem
            if stem in videos_by_stem:
                gt_map[stem] = videos_by_stem[stem]
            else:
                missing.append(url)

        if missing:
            click.echo(
                f"ERROR: --gt-json 中有 {len(missing)} 个 url 在 videos-dir 里找不到.\n"
                f"  前 10 个缺失: {missing[:10]}",
                err=True,
            )
            return 1

        _log.info(
            "gt json parsed",
            extra={"gt_json": str(gt_p), "n_gt_entries": len(gt_data.get("gts", [])),
                   "n_mapped": len(gt_map)},
        )

    # ---- 5. 检查 ffmpeg + 权重 + inference model ----
    if shutil.which("ffmpeg") is None:
        click.echo(
            "ERROR (环境问题): ffmpeg 命令不可用. 请安装 ffmpeg.",
            err=True,
        )
        return 2

    repo_root = find_repo_root()
    pretrained = cfg.get("pretrained") or {}
    train_weight_path = Path(pretrained.get("train_weight_path", ""))
    if not train_weight_path.is_absolute():
        train_weight_path = repo_root / train_weight_path
    if not train_weight_path.is_file():
        url = pretrained.get("train_weight_url", "")
        click.echo(
            f"ERROR (用户输入错): PP-TSM 训练权重缺失: {train_weight_path}\n"
            f"  curl -fL -o {train_weight_path} \\\n"
            f"      {url}",
            err=True,
        )
        return 1

    inf_dir = Path(pretrained.get("inference_dir", "data/raw/pretrained"))
    if not inf_dir.is_absolute():
        inf_dir = repo_root / inf_dir
    pdmodel = inf_dir / pretrained.get("inference_pdmodel", "ppTSM.pdmodel")
    pdiparams = inf_dir / pretrained.get("inference_pdiparams", "ppTSM.pdiparams")
    if not pdmodel.is_file() or not pdiparams.is_file():
        click.echo(
            f"INFO: inference 文件缺失, 自动调 export_pptsm_inference.py...",
            err=True,
        )
        try:
            subprocess.run(
                [sys.executable, str(repo_root / "scripts" / "export_pptsm_inference.py"),
                 "--config", str(cfg_p)],
                capture_output=True, text=True, check=True, timeout=600,
            )
        except subprocess.CalledProcessError as exc:
            click.echo(
                f"ERROR (环境问题): export 失败: {exc.stderr[-500:]}",
                err=True,
            )
            return 2

    # ---- 6. 初始化 extractor (一次加载, 多视频复用) ----
    try:
        extractor = PPTSMExtractor(
            pdmodel=pdmodel,
            pdiparams=pdiparams,
            config=ext_cfg,
            gpu_mem_mb=int((cfg.get("runtime") or {}).get("gpu_mem_mb", 8000)),
        )
    except PPTSMExtractorError as exc:
        click.echo(f"ERROR (环境问题): PPTSMExtractor 构造失败: {exc}", err=True)
        return 2
    except Exception as exc:
        click.echo(f"ERROR (环境问题): {type(exc).__name__}: {exc}", err=True)
        return 2

    # PP-TSM sha256 (cache 一次, 同一 run 所有视频共用)
    pp_tsm_weight_sha = _sha256_of_file(train_weight_path)
    pp_tsm_inf_sha = _combined_sha256(pdmodel, pdiparams)
    extraction_commit = _git_head_commit(repo_root)

    # ---- 7. 处理每个视频 ----
    tmp_cfg = cfg.get("tmp") or {}
    tmp_root_str = tmp_cfg.get("frames_root", "data/raw/.tmp/")
    tmp_root = Path(tmp_root_str)
    if not tmp_root.is_absolute():
        tmp_root = repo_root / tmp_root
    tmp_root.mkdir(parents=True, exist_ok=True)

    writer = ManifestWriter(manifest_path)
    existing_cids = writer.get_existing_clip_ids() if not force else set()
    clip_id_to_video: dict[str, Path] = {}   # 用于 GT JSON 重写 (stem → clip_id)
    stem_to_clip_id: dict[str, str] = {}

    n_total = len(all_videos)
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for idx, video_path in enumerate(all_videos):
        click.echo(
            f"[{idx + 1}/{n_total}] {video_path.relative_to(videos_p)} ...",
            err=True,
        )

        # 算 clip_id (永远要算, 用于 GT JSON 重写)
        try:
            cid = compute_clip_id(video_path)
        except Exception as exc:
            _log.warning("clip_id failed", extra={"video": str(video_path), "error": str(exc)})
            writer.append(ManifestRow(
                video_path=str(video_path),
                error=f"clip_id failed: {exc}",
                extracted_at=datetime.now(timezone.utc).isoformat(),
            ))
            n_failed += 1
            continue

        stem_to_clip_id[video_path.stem] = cid

        pkl_path = features_dir / f"{cid}.pkl"

        # 幂等: 已存在就跳过
        if not force and (cid in existing_cids or pkl_path.is_file()):
            _log.info("skip (already exists)", extra={"clip_id": cid})
            writer.append(ManifestRow(
                video_path=str(video_path),
                clip_id=cid,
                pkl_path=str(pkl_path),
                extracted_at=datetime.now(timezone.utc).isoformat(),
                error="skipped (already exists)",
            ))
            n_skipped += 1
            continue

        # 抽帧 → PP-TSM → pkl
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        frames_dir = tmp_root / f"extract_{run_id}"

        try:
            frames_result = extract_frames_to_dir(
                video_path, frames_dir, fps=ext_cfg.fps,
            )
        except FFmpegError as exc:
            shutil.rmtree(frames_dir, ignore_errors=True)
            writer.append(ManifestRow(
                video_path=str(video_path),
                clip_id=cid,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                error=f"ffmpeg: {exc}",
            ))
            n_failed += 1
            continue

        try:
            features = extractor.extract_from_frames_dir(frames_dir)
        except Exception as exc:
            shutil.rmtree(frames_dir, ignore_errors=True)
            writer.append(ManifestRow(
                video_path=str(video_path),
                clip_id=cid,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                error=f"pp_tsm: {type(exc).__name__}: {exc}",
            ))
            n_failed += 1
            continue

        # 写 pkl
        with pkl_path.open("wb") as f:
            pickle.dump({"image_feature": features}, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_sha = _sha256_of_file(pkl_path)

        # append manifest 完整行
        writer.append(ManifestRow(
            video_path=str(video_path),
            clip_id=cid,
            n_frames=frames_result.n_frames,
            fps_original=round(frames_result.fps_original, 3),
            fps_used=frames_result.fps_used,
            duration_sec=round(frames_result.duration_sec, 3),
            pkl_path=str(pkl_path),
            pkl_sha256=pkl_sha,
            pp_tsm_weight_sha256=pp_tsm_weight_sha,
            pp_tsm_inference_sha256=pp_tsm_inf_sha,
            pp_tsm_config_hash=config_hash,
            extraction_commit=extraction_commit,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            error="",
        ))

        # 清理临时帧目录
        shutil.rmtree(frames_dir, ignore_errors=True)
        n_processed += 1

    writer.close()

    # ---- 8. 如有 GT JSON, 重写 url 字段 → clip_id ----
    label_json_path = None
    if gt_data is not None and gt_map is not None:
        new_gt = {
            "fps":      gt_data.get("fps", 25),
            "gts":      [],
        }
        for g in gt_data.get("gts", []):
            url = str(g.get("url", ""))
            stem = Path(url).stem
            if stem in stem_to_clip_id:
                new_g = dict(g)
                # 保留原 url 扩展名, 但 stem 替换为 clip_id
                original_suffix = Path(url).suffix or ".mp4"
                new_g["url"] = f"{stem_to_clip_id[stem]}{original_suffix}"
                new_gt["gts"].append(new_g)

        label_json_path = out_root / f"label_cls14_{subset_name}.json"
        label_json_path.write_text(
            json.dumps(new_gt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(
            f"INFO: GT JSON 重写到 {label_json_path} ({len(new_gt['gts'])} entries)",
            err=True,
        )

    # ---- 9. stdout 摘要 ----
    elapsed = time.time() - t_start
    summary = {
        "schema":             "build-feature-pkls-v1",
        "videos_dir":         str(videos_p),
        "output_dir":         str(features_dir),
        "name":               subset_name,
        "n_videos_total":     n_total,
        "n_videos_processed": n_processed,
        "n_videos_skipped":   n_skipped,
        "n_videos_failed":    n_failed,
        "manifest_path":      str(manifest_path),
        "label_json_path":    str(label_json_path) if label_json_path else None,
        "elapsed_sec":        round(elapsed, 3),
    }
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ 批量抽特征完成: {n_processed} 处理 / {n_skipped} 跳过 / {n_failed} 失败 / {n_total} 总",
        err=True,
    )
    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _combined_sha256(a: Path, b: Path) -> str:
    """两个文件拼接的 sha256 (与 export_pptsm_inference.py 一致)."""
    h = hashlib.sha256()
    for p in [a, b]:
        with p.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk: break
                h.update(chunk)
    return h.hexdigest()


def _git_head_commit(repo_root: Path) -> str:
    """取 git HEAD commit hash; 失败返回空字符串."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""
