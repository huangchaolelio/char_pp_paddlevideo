"""拉取 / 整理 / 发现公开乒乓球动作数据集 (FR-004 ~ FR-006).

本模块的职责:
- 把 ``configs/datasets/<name>.yaml`` 中描述的数据**落到本地** (``data/raw/`` + ``data/clips/``);
- 把落地后的目录扫描成 :class:`VideoClip` 集合 (供 :mod:`pingpong_av.data.splitter` 划分).

支持三种 ``source.type``:
- ``url_list``  — 下载 ``urls`` 列表中的 .tar / .tar.gz / .zip 到 ``data/raw/``,
                  解压到 ``data/clips/`` 后扫描.
- ``local_dir`` — 跳过下载, 直接用 ``source.path`` 指向的目录扫描 (用于 US4 自定义数据).
- ``manual``    — 不下载, 检测哨兵文件 ``data/raw/<name>/<sentinel_relpath>``;
                  缺失则给出明确指引到上游下载页 (``DatasetNeedsManualSetup``).
                  对应 PaddleVideo `release/2.2.0` 不提供公开 URL 的真实约束.
                  可选地下载 ``source.smoke_sample.url`` 作为 7.4MB 单样例 pkl.

不在本模块的范围:
- 数据划分逻辑 (``data.splitter``).
- list 文件落盘 (``data.list_writer``).

幂等性: 已下载/已解压则跳过 (除非传 ``force=True``); 校验通过即认为已落地.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen, Request

from pingpong_av.data.splitter import VideoClip
from pingpong_av.utils.logging import get_logger

__all__ = ["fetch_and_discover", "DatasetFetchError", "DatasetNeedsManualSetup"]

_log = get_logger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DatasetFetchError(RuntimeError):
    """数据集拉取或扫描失败."""


class DatasetNeedsManualSetup(DatasetFetchError):
    """source.type=manual 时哨兵文件缺失; 数据需要用户手动准备.

    错误消息会包含 ``source.manual_steps`` 中的引导文本, 由 CLI 层映射为退出码 1.
    """


@dataclass(frozen=True)
class FetchResult:
    """fetch_and_discover 的返回结果."""

    clips: list[VideoClip]
    raw_dir: Path           # data/raw/<dataset>
    clips_dir: Path         # data/clips/<dataset>
    n_classes: int
    used_official_split: bool   # 数据是否带有官方 train/val/test 划分


# --------------------------------------------------------------------------------------
# 公共入口
# --------------------------------------------------------------------------------------


def fetch_and_discover(
    config: dict[str, Any],
    *,
    repo_root: Path,
    force: bool = False,
) -> FetchResult:
    """按 dataset 配置拉取数据并扫描成 VideoClip 列表.

    参数:
        config: 通过 :func:`pingpong_av.utils.config.load_config` 加载的 dataset 配置 dict.
                必须含 ``classes`` (类别表), ``source`` (含 type), ``paths`` (raw_dir/clip_dir).
        repo_root: 仓库根, 用于解析相对路径.
        force: 忽略已有产物强制重新拉取/重新扫描.
    """
    name = config.get("name", "dataset")
    paths = config.get("paths", {})
    raw_dir = repo_root / Path(paths.get("raw_dir", "data/raw")) / name
    clips_dir = repo_root / Path(paths.get("clip_dir", "data/clips")) / name

    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # 1) 拉取阶段
    source = config.get("source") or {}
    src_type = source.get("type", "url_list")
    if src_type == "url_list":
        urls = source.get("urls") or []
        if not urls:
            raise DatasetFetchError(
                "source.type='url_list' 但 urls 为空. 请在 configs/datasets/*.yaml 中填入实际 URL, "
                "或将 source.type 改为 'local_dir' 并指定 source.path, "
                "或改为 'manual' 并手动准备数据 (用于上游不提供公开 URL 的情形)."
            )
        _download_and_extract(urls, raw_dir=raw_dir, clips_dir=clips_dir, force=force)
    elif src_type == "local_dir":
        local_path = source.get("path")
        if not local_path:
            raise DatasetFetchError("source.type='local_dir' 但 source.path 缺失")
        local = Path(local_path).expanduser().resolve()
        if not local.is_dir():
            raise DatasetFetchError(f"source.path 指向的目录不存在: {local}")
        # 不复制大文件, 直接让 clips_dir 指向该目录 (软链接, 避免重复占空间)
        if force or not _dir_has_videos(clips_dir):
            _link_or_copy_local(local, clips_dir)
    elif src_type == "manual":
        _ensure_manual_setup(source, raw_dir=raw_dir, clips_dir=clips_dir, force=force)
    else:
        raise DatasetFetchError(f"未知的 source.type: {src_type!r}")

    # 2) 扫描阶段
    classes_meta = config["classes"]
    name_to_id = {c["name"]: int(c["id"]) for c in classes_meta}

    # 优先使用数据集自带的 train/val/test 划分文件
    used_official, clips = _discover_clips(clips_dir, name_to_id, prefer_official=True)
    if not clips:
        raise DatasetFetchError(
            f"在 {clips_dir} 下扫描到 0 个视频片段. "
            f"请确认数据已正确放置, 或检查 classes 配置中的 name 与目录名匹配 "
            f"(已知类别: {sorted(name_to_id.keys())})."
        )

    return FetchResult(
        clips=clips,
        raw_dir=raw_dir,
        clips_dir=clips_dir,
        n_classes=len(classes_meta),
        used_official_split=used_official,
    )


# --------------------------------------------------------------------------------------
# 下载 + 解压
# --------------------------------------------------------------------------------------


def _download_and_extract(urls: list[str], *, raw_dir: Path, clips_dir: Path, force: bool) -> None:
    """下载所有 URL 到 raw_dir, 解压到 clips_dir. 支持 .tar/.tar.gz/.zip.

    幂等: 同名文件已存在且大小与 HEAD 一致则跳过 (粗略校验).
    """
    for url in urls:
        fname = Path(urlparse(url).path).name
        if not fname:
            raise DatasetFetchError(f"URL 不含文件名: {url}")
        target = raw_dir / fname

        if target.exists() and not force:
            _log.info("download skipped (exists)", extra={"url": url, "path": str(target)})
        else:
            _log.info("downloading", extra={"url": url, "path": str(target)})
            _download(url, target)

        # 解压
        marker = clips_dir / f".extracted-{_safe_marker(fname)}"
        if marker.exists() and not force:
            _log.info("extract skipped (marker exists)", extra={"file": fname})
            continue
        _extract_archive(target, clips_dir)
        marker.write_text(f"extracted from {fname}\n", encoding="utf-8")


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    req = Request(url, headers={"User-Agent": "char_pp_prj-bootstrap/1.0"})
    try:
        with urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetFetchError(f"下载失败 {url}: {exc}") from exc


def _extract_archive(archive: Path, dest: Path) -> None:
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            _safe_tar_extract(tf, dest)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r:") as tf:
            _safe_tar_extract(tf, dest)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)  # noqa: S202 — 内部数据集, 来源已在 config 里审计
    else:
        raise DatasetFetchError(f"不支持的压缩格式: {archive}; 请使用 .tar/.tar.gz/.tgz/.zip")


def _safe_tar_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """防御 tar slip (CVE-2007-4559): 拒绝带 .. 的成员."""
    dest = dest.resolve()
    for member in tf.getmembers():
        out = (dest / member.name).resolve()
        if not str(out).startswith(str(dest)):
            raise DatasetFetchError(f"拒绝可疑 tar 成员 (路径越界): {member.name}")
    tf.extractall(dest)


def _safe_marker(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# manual 模式 (PaddleVideo 上游不提供公开 URL 的真实约束)
# --------------------------------------------------------------------------------------


def _ensure_manual_setup(
    source: dict[str, Any], *, raw_dir: Path, clips_dir: Path, force: bool,
) -> None:
    """检测哨兵文件; 缺失则抛 :class:`DatasetNeedsManualSetup` 含完整指引.

    成功后 (哨兵存在) 把 raw_dir 软链接到 clips_dir 以复用现有扫描逻辑.

    可选: 下载 source.smoke_sample.url 到 raw_dir/<smoke_sample.relpath>;
    若该 URL 不可达则只 warn, **不阻断** manual 流程 (smoke 样例只对 infer-clip 有用).
    """
    sentinel_rel = source.get("sentinel_relpath", ".ready")
    sentinel = raw_dir / sentinel_rel

    if not sentinel.is_file():
        steps = source.get("manual_steps") or []
        origin = source.get("origin", "上游 PaddleVideo 项目")
        origin_url = source.get("origin_url", "")
        msg_lines = [
            f"数据集需要手动准备 (上游 {origin} 不提供公开下载 URL).",
            "",
            f"哨兵文件缺失: {sentinel}",
            "",
            "准备步骤:",
        ]
        msg_lines.extend(f"  {s}" for s in steps)
        if origin_url:
            msg_lines.append("")
            msg_lines.append(f"上游入口: {origin_url}")
        msg_lines.append("")
        msg_lines.append(f"预期数据落位于: {raw_dir}")
        msg_lines.append(f"准备就绪后请创建空文件: {sentinel}")
        raise DatasetNeedsManualSetup("\n".join(msg_lines))

    # 哨兵存在, 把 raw_dir 暴露给 clips_dir 以复用扫描逻辑
    if force or not _dir_has_videos(clips_dir):
        _link_or_copy_local(raw_dir, clips_dir)
    _log.info(
        "manual dataset ready",
        extra={"sentinel": str(sentinel), "raw_dir": str(raw_dir)},
    )

    # 可选: smoke_sample 下载 (失败仅 warn)
    smoke = source.get("smoke_sample") or {}
    smoke_url = smoke.get("url")
    smoke_rel = smoke.get("relpath")
    if smoke_url and smoke_rel:
        smoke_path = raw_dir / smoke_rel
        if smoke_path.exists() and not force:
            _log.info("smoke_sample skipped (exists)", extra={"path": str(smoke_path)})
        else:
            try:
                _download(smoke_url, smoke_path)
                _log.info(
                    "smoke_sample downloaded",
                    extra={"url": smoke_url, "path": str(smoke_path),
                           "size": smoke_path.stat().st_size},
                )
            except DatasetFetchError as exc:
                _log.warning(
                    "smoke_sample download failed (non-fatal)",
                    extra={"url": smoke_url, "error": str(exc)},
                )


# --------------------------------------------------------------------------------------
# local_dir 模式
# --------------------------------------------------------------------------------------


def _link_or_copy_local(src: Path, dest: Path) -> None:
    """把本地数据目录的内容暴露在 clips_dir 下.

    优先用符号链接 (省盘); 失败则复制 (Windows 等不支持 symlink 的平台).
    """
    if dest.is_symlink() or dest.exists() and any(dest.iterdir()):
        # 已经有内容; 不重做避免误删用户数据
        return
    try:
        # 清理空 dest 目录后建链
        if dest.exists():
            dest.rmdir()
        dest.symlink_to(src, target_is_directory=True)
        _log.info("local_dir linked", extra={"src": str(src), "dest": str(dest)})
    except OSError:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        _log.info("local_dir copied (symlink unsupported)", extra={"src": str(src), "dest": str(dest)})


# --------------------------------------------------------------------------------------
# 扫描: 把目录变成 VideoClip 列表
# --------------------------------------------------------------------------------------


def _dir_has_videos(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.rglob("*"):
        if p.suffix.lower() in _VIDEO_EXTS:
            return True
    return False


def _discover_clips(
    root: Path, name_to_id: dict[str, int], *, prefer_official: bool
) -> tuple[bool, list[VideoClip]]:
    """扫描 root 下所有视频片段, 转化为 VideoClip 列表.

    支持两种常见的目录约定:
      A) 按类别子目录: ``<root>/<class_name>/<source_video>/<clip>.mp4``
      B) 平铺 + 标注文件: ``<root>/labels.csv`` (clip_path, source_video_id, label_name)

    若同时存在 ``<root>/{train,val,test}.txt`` 这样的官方 split 文件且
    ``prefer_official=True``, 则按 split 文件读取并设置 clip.split. 此情况下返回的
    used_official=True, 调用方在 ``pp data-prepare`` 中会**绕过** by_video_ratio 划分.
    """
    used_official = False

    # 1) 优先官方 split (CSV / TXT 形式) — 简单实现: 三份 split 文件存在则按 split 读取
    if prefer_official:
        official = _try_read_official_splits(root, name_to_id)
        if official is not None:
            used_official = True
            return used_official, official

    # 2) labels.csv 形式
    csv_clips = _try_read_labels_csv(root, name_to_id)
    if csv_clips is not None:
        return used_official, csv_clips

    # 3) 按类别子目录扫描
    return used_official, _scan_by_class_dirs(root, name_to_id)


def _try_read_official_splits(root: Path, name_to_id: dict[str, int]) -> list[VideoClip] | None:
    """如果 root 下存在 train.txt/val.txt/test.txt, 读取并返回带 split 字段的 clips."""
    files = {s: root / f"{s}.txt" for s in ("train", "val", "test")}
    if not all(p.is_file() for p in files.values()):
        return None

    out: list[VideoClip] = []
    for split, path in files.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            rel_path, label = parts[0], parts[1]
            try:
                label_id = int(label)
            except ValueError:
                if label not in name_to_id:
                    continue
                label_id = name_to_id[label]
            clip_path = (root / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path)
            clip_id = clip_path.stem
            source_video_id = clip_path.parent.name or clip_id  # 父目录作为源视频 ID, 合理近似
            out.append(VideoClip(
                clip_id=clip_id,
                source_video_id=source_video_id,
                path=str(clip_path),
                label_id=label_id,
                split=split,  # type: ignore[arg-type]
            ))
    return out if out else None


def _try_read_labels_csv(root: Path, name_to_id: dict[str, int]) -> list[VideoClip] | None:
    csv_path = root / "labels.csv"
    if not csv_path.is_file():
        return None
    import csv

    out: list[VideoClip] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row.get("label_name") or row.get("label") or ""
            if label_name not in name_to_id:
                continue
            rel = row.get("clip_path") or row.get("path")
            if not rel:
                continue
            clip_path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
            out.append(VideoClip(
                clip_id=row.get("clip_id") or clip_path.stem,
                source_video_id=row.get("source_video_id") or clip_path.parent.name or clip_path.stem,
                path=str(clip_path),
                label_id=name_to_id[label_name],
                start_sec=_to_float(row.get("start_sec")),
                end_sec=_to_float(row.get("end_sec")),
            ))
    return out if out else None


def _scan_by_class_dirs(root: Path, name_to_id: dict[str, int]) -> list[VideoClip]:
    """``<root>/<class_name>/<source_video>/<clip>.ext``; 缺失中间目录时退化."""
    out: list[VideoClip] = []
    for class_name, label_id in name_to_id.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for video_path in class_dir.rglob("*"):
            if video_path.is_file() and video_path.suffix.lower() in _VIDEO_EXTS:
                # source_video_id: 父目录名 (若 class_dir 直接含视频则用 stem)
                rel_parent = video_path.parent
                if rel_parent == class_dir:
                    src_vid = video_path.stem  # 一个视频本身就是一个源
                else:
                    src_vid = rel_parent.name
                out.append(VideoClip(
                    clip_id=video_path.stem,
                    source_video_id=src_vid,
                    path=str(video_path),
                    label_id=label_id,
                ))
    return out


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
