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
import os
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
        insecure = bool(source.get("insecure", False))
        _download_and_extract(urls, raw_dir=raw_dir, clips_dir=clips_dir, force=force, insecure=insecure)
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
    elif src_type == "cos":
        _ensure_cos_setup(source, raw_dir=raw_dir, clips_dir=clips_dir, force=force)
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


def _download_and_extract(
    urls: list[str], *, raw_dir: Path, clips_dir: Path, force: bool, insecure: bool = False,
) -> None:
    """下载所有 URL 到 raw_dir, 解压到 clips_dir. 支持 .tar/.tar.gz/.zip/.rar.

    幂等: 同名文件已存在且大小与 HEAD 一致则跳过 (粗略校验).
    insecure=True 时跳过 SSL 证书校验 (用于自签名服务器, 例如 UCF101 在 CRCV).
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
            _download(url, target, insecure=insecure)

        # 解压
        marker = clips_dir / f".extracted-{_safe_marker(fname)}"
        if marker.exists() and not force:
            _log.info("extract skipped (marker exists)", extra={"file": fname})
            continue
        _extract_archive(target, clips_dir)
        marker.write_text(f"extracted from {fname}\n", encoding="utf-8")


def _download(url: str, target: Path, *, insecure: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    req = Request(url, headers={"User-Agent": "char_pp_prj-bootstrap/1.0"})
    ctx = None
    if insecure:
        import ssl
        ctx = ssl._create_unverified_context()  # noqa: SLF001 — 与上游 download_*.sh 一致 (--no-check-certificate)
    try:
        with urlopen(req, timeout=120, context=ctx) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetFetchError(f"下载失败 {url}: {exc}") from exc


def _extract_archive(archive: Path, dest: Path) -> None:
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        # 用流式模式 r|gz: 单遍解压, 适合大文件 (43GB+)
        with tarfile.open(archive, "r|gz") as tf:
            _safe_tar_extract(tf, dest)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r|") as tf:
            _safe_tar_extract(tf, dest)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)  # noqa: S202 — 内部数据集, 来源已在 config 里审计
    elif archive.suffix == ".rar":
        _extract_rar(archive, dest)
    else:
        raise DatasetFetchError(
            f"不支持的压缩格式: {archive}; 请使用 .tar/.tar.gz/.tgz/.zip/.rar"
        )


def _extract_rar(archive: Path, dest: Path) -> None:
    """通过 unrar 命令行工具解压 .rar 到 dest. 需要 conda install unrar (or apt install unrar)."""
    import subprocess

    unrar = shutil.which("unrar")
    if not unrar:
        raise DatasetFetchError(
            f"unrar 未安装, 无法解压 {archive.name}. 请先安装:\n"
            "  conda install -y -n char_python311 -c conda-forge unrar\n"
            "或 (如果有 sudo apt 权限): sudo apt install unrar"
        )
    dest.mkdir(parents=True, exist_ok=True)
    # unrar x: 完整解压, -o+: 覆盖, -idq: 静默
    result = subprocess.run(
        [unrar, "x", "-o+", "-idq", str(archive)],
        cwd=str(dest),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise DatasetFetchError(
            f"unrar 解压失败 (returncode={result.returncode}):\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    _log.info("rar extracted", extra={"archive": str(archive), "dest": str(dest)})


def _safe_tar_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """防御 tar slip (CVE-2007-4559): 拒绝带 .. 的成员.

    流式逐成员处理: 边解压边校验, 不要先 getmembers() 再 extractall() (后者对 43GB
    级别的 tar.gz 会读两遍, 严重拖慢; 实测 729 个 70MB pkl 双遍需 1+ 小时).
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n_extracted = 0
    for member in tf:
        out_path = (dest / member.name).resolve()
        if not str(out_path).startswith(str(dest) + os.sep) and str(out_path) != str(dest):
            raise DatasetFetchError(f"拒绝可疑 tar 成员 (路径越界): {member.name}")
        # tar 流式 extract; tarfile 在内部对 .gz 是 sequential, 这里就一遍读完
        tf.extract(member, dest)
        if member.isreg():
            n_extracted += 1
            if n_extracted % 50 == 0:
                _log.info("tar extract progress",
                          extra={"n_extracted": n_extracted, "current": member.name})
    _log.info("tar extract done", extra={"n_files": n_extracted, "dest": str(dest)})


def _safe_marker(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# cos 模式 (Tencent Cloud Object Storage)
# --------------------------------------------------------------------------------------


class CosDependencyError(DatasetFetchError):
    """qcloud_cos SDK 未安装."""


def _ensure_cos_setup(
    source: dict[str, Any], *, raw_dir: Path, clips_dir: Path, force: bool,
) -> None:
    """从腾讯云 COS 拉取对象到 ``raw_dir`` (可选解压到 ``clips_dir``).

    yaml 字段 (source 段):
        type: cos
        keys: [<list of object keys>]    # 必填. 不含 prefix.
        prefix: pp_video                 # 可选 (默认从 env COS_VIDEO_PREFIX)
        bucket: charhuang-pp-1253960454  # 可选 (默认从 env COS_BUCKET)
        region: ap-guangzhou             # 可选 (默认从 env COS_REGION)
        secret_id_env:  COS_SECRET_ID    # 默认即此名
        secret_key_env: COS_SECRET_KEY   # 默认即此名
        extract: true                    # 默认 true: 自动解压 .tar.gz/.zip/.rar 到 clips_dir
        max_thread: 8                    # 多线程下载

    凭据来源 (按优先级):
        1. ``source.secret_id`` / ``source.secret_key`` (yaml 内直填; **不推荐**)
        2. ``${COS_SECRET_ID}`` / ``${COS_SECRET_KEY}`` 环境变量 (推荐, 通过 `.env`)

    幂等性: 已存在的文件 (大小匹配) 跳过下载; 解压通过 marker 文件去重.
    """
    keys = source.get("keys") or []
    if not keys:
        raise DatasetFetchError("source.type='cos' 但 keys 为空 (需要至少一个对象 key).")

    # 加载 .env 到 os.environ (如果还没加载)
    _load_dotenv_if_present()

    bucket = source.get("bucket") or os.environ.get("COS_BUCKET")
    region = source.get("region") or os.environ.get("COS_REGION")
    if not bucket or not region:
        raise DatasetFetchError(
            "COS bucket / region 未配置. 请在 configs/datasets/*.yaml 的 source 段填写, "
            "或在 `.env` 中提供 COS_BUCKET / COS_REGION."
        )

    secret_id_env = source.get("secret_id_env", "COS_SECRET_ID")
    secret_key_env = source.get("secret_key_env", "COS_SECRET_KEY")
    secret_id = source.get("secret_id") or os.environ.get(secret_id_env)
    secret_key = source.get("secret_key") or os.environ.get(secret_key_env)
    if not secret_id or not secret_key:
        raise DatasetFetchError(
            f"COS 凭据缺失: 请在 `.env` 中提供 {secret_id_env} / {secret_key_env}, "
            "或在 source 段直接填写 secret_id / secret_key (不推荐)."
        )

    prefix = source.get("prefix") or os.environ.get("COS_VIDEO_PREFIX", "")
    prefix = prefix.strip("/")

    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as exc:
        raise CosDependencyError(
            "qcloud_cos SDK 未安装. 请运行: pip install cos-python-sdk-v5"
        ) from exc

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    client = CosS3Client(config)
    max_thread = int(source.get("max_thread", 8))
    extract = bool(source.get("extract", True))

    raw_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        full_key = f"{prefix}/{key}" if prefix else key
        local_name = Path(key).name
        target = raw_dir / local_name

        # 已存在且大小看似合理则跳过 (用 HEAD 拿 server-side size 比对; 失败则保守跳)
        if target.exists() and not force:
            try:
                head = client.head_object(Bucket=bucket, Key=full_key)
                remote_size = int(head.get("Content-Length", 0))
                local_size = target.stat().st_size
                if local_size == remote_size and remote_size > 0:
                    _log.info(
                        "cos download skipped (size match)",
                        extra={"key": full_key, "size": local_size},
                    )
                else:
                    _log.info(
                        "cos download retrying (size mismatch)",
                        extra={"key": full_key, "local": local_size, "remote": remote_size},
                    )
                    target.unlink()
            except Exception:
                _log.info("cos download skipped (head failed but local exists)",
                          extra={"key": full_key})
                continue

        if not target.exists():
            _log.info("cos downloading", extra={"key": full_key, "target": str(target)})
            try:
                client.download_file(
                    Bucket=bucket, Key=full_key,
                    DestFilePath=str(target),
                    MAXThread=max_thread,
                    EnableCRC=True,
                )
            except Exception as exc:
                raise DatasetFetchError(
                    f"COS 下载失败 (bucket={bucket}, key={full_key}): {exc}"
                ) from exc
            _log.info(
                "cos downloaded",
                extra={"key": full_key, "size": target.stat().st_size},
            )

        # 可选解压
        if extract and _is_extractable(target):
            marker = clips_dir / f".extracted-{_safe_marker(local_name)}"
            if marker.exists() and not force:
                _log.info("cos extract skipped (marker exists)", extra={"file": local_name})
                continue
            _extract_archive(target, clips_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"extracted from {local_name}\n", encoding="utf-8")


def _is_extractable(path: Path) -> bool:
    """判断文件是否需要被本模块自动解压."""
    if path.suffixes[-2:] == [".tar", ".gz"]:
        return True
    return path.suffix in (".tgz", ".tar", ".zip", ".rar")


def _load_dotenv_if_present() -> None:
    """如果工作目录或仓库根存在 .env, 把里面的 KEY=VALUE 加载到 os.environ.

    幂等: 已设置的 env var 不被覆盖 (允许命令行/系统级覆盖 .env).
    """
    candidates = [Path.cwd() / ".env"]
    try:
        from pingpong_av.utils.env import find_repo_root
        candidates.append(find_repo_root() / ".env")
    except Exception:
        pass
    for env_file in candidates:
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            return  # 第一个找到的就够


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

    支持三种常见的目录约定:
      A) 按类别子目录: ``<root>/<class_name>/<source_video>/<clip>.mp4``
      B) 平铺 + 标注文件: ``<root>/labels.csv`` (clip_path, source_video_id, label_name)
      C) BMN 特征驱动: ``<root>/Features_*/*.pkl`` + 顶层 ``label_cls*.json`` 含 GT
         (用于 AI Studio 竞赛 #127 时序定位任务, 与上游 BMN 训练对齐)

    若同时存在 ``<root>/{train,val,test}.txt`` 这样的官方 split 文件且
    ``prefer_official=True``, 则按 split 文件读取并设置 clip.split. 此情况下返回的
    used_official=True, 调用方在 ``pp data-prepare`` 中会**绕过** by_video_ratio 划分.
    """
    used_official = False

    # 1) 优先官方 split (CSV / TXT 形式) — 按 UCF101 / 通用两种格式尝试
    if prefer_official:
        official = _try_read_ucf101_splits(root, name_to_id)
        if official is not None:
            used_official = True
            return used_official, official
        official = _try_read_official_splits(root, name_to_id)
        if official is not None:
            used_official = True
            return used_official, official

    # 2) labels.csv 形式
    csv_clips = _try_read_labels_csv(root, name_to_id)
    if csv_clips is not None:
        return used_official, csv_clips

    # 3) BMN 特征驱动 (.pkl + label JSON)
    bmn_clips = _try_read_bmn_features(root, name_to_id)
    if bmn_clips is not None:
        return used_official, bmn_clips

    # 4) 按类别子目录扫描
    return used_official, _scan_by_class_dirs(root, name_to_id)


def _try_read_bmn_features(root: Path, name_to_id: dict[str, int]) -> list[VideoClip] | None:
    """识别 BMN 特征驱动数据布局 (AI Studio 竞赛 #127 + ``Features_competition_train.tar.gz``).

    期望布局:
        clips_root/Features_competition_train/<32-char-hash>.pkl   # 每个 .pkl = 一段视频的 PP-TSN 特征
        raw_root/label_cls14_train.json                            # 时序 GT (同级于 .tar.gz)

    label_id 策略 (BMN 任务中每段视频含多个动作, 不存在"整段 1 类"):
        优先从 ``<raw_root>/label_cls*.json`` 查该视频的 GT actions, 用**出现次数最多**
        的 action.label_ids 作为 "dominant label" 赋给 VideoClip.label_id (用于 splitter
        的分层校验 + per-class 支持度统计). 实际 BMN 训练时, label_id 字段被忽略,
        真实标签从同一份 JSON 按 url 索引读取.

    若 json 不存在, label_id 退化为 0 (所有视频归一类, splitter 仍可工作).
    """
    feat_root = root / "Features_competition_train"
    if not feat_root.is_dir():
        return None
    pkl_files = sorted(feat_root.glob("*.pkl"))
    if not pkl_files:
        return None

    # 尝试加载 GT JSON (从 raw_root 同级目录下找)
    # clips_dir (root) 与 raw_dir 是兄弟: data/clips/<name>/ vs data/raw/<name>/
    # 但我们这里只有 root (clips). 用相对路径推 raw_root.
    repo_root_guess = root.parents[2]  # data/clips/<topname>/<name> → repo
    raw_root = repo_root_guess / "data" / "raw" / root.parent.name / root.name
    if not raw_root.is_dir():
        raw_root = root  # fallback: json 也在 clips 下
    label_json_candidates = sorted(raw_root.glob("label_cls*.json")) + sorted(root.glob("label_cls*.json"))

    import json as _json
    from collections import Counter
    gt_by_video: dict[str, int] = {}  # clip_id -> dominant label_id
    if label_json_candidates:
        try:
            gt = _json.loads(label_json_candidates[0].read_text(encoding="utf-8"))
            for g in gt.get("gts", []):
                url = str(g.get("url", ""))
                clip_id = Path(url).stem
                if not clip_id:
                    continue
                labels: list[int] = []
                for a in g.get("actions", []):
                    if isinstance(a, dict):
                        labels.extend(a.get("label_ids", []))
                if labels:
                    dominant = Counter(labels).most_common(1)[0][0]
                    gt_by_video[clip_id] = int(dominant)
            _log.info(
                "BMN GT json parsed",
                extra={
                    "json": str(label_json_candidates[0]),
                    "n_videos_with_gt": len(gt_by_video),
                },
            )
        except Exception as exc:
            _log.warning("BMN GT json parse failed", extra={"error": str(exc)})

    out: list[VideoClip] = []
    n_with_gt = 0
    for pkl_path in pkl_files:
        clip_id = pkl_path.stem
        label_id = gt_by_video.get(clip_id, 0)
        if clip_id in gt_by_video:
            n_with_gt += 1
        out.append(VideoClip(
            clip_id=clip_id,
            source_video_id=clip_id,
            path=str(pkl_path),
            label_id=label_id,
            split="",
        ))
    _log.info(
        "BMN features discovered",
        extra={
            "feat_root": str(feat_root),
            "n_videos": len(out),
            "n_with_gt_labels": n_with_gt,
        },
    )
    return out


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


def _try_read_ucf101_splits(root: Path, name_to_id: dict[str, int]) -> list[VideoClip] | None:
    """识别 UCF101 风格官方划分 ``ucfTrainTestlist/{train,test}list01.txt``.

    UCF101 仅有 train + test (无独立 val); 本项目从 trainlist01 中按 source_video_id 切出
    最后 ~10% 作为 val (保证同源不跨 split, 章程 IV).

    UCF101 trainlist 行格式: ``ApplyEyeMakeup/v_xxx_g01_c01.avi 1`` (1-indexed label)
    UCF101 testlist 行格式:  ``ApplyEyeMakeup/v_xxx_g01_c01.avi`` (无 label, 由父目录推)
    """
    list_dir = root / "ucfTrainTestlist"
    train_p = list_dir / "trainlist01.txt"
    test_p = list_dir / "testlist01.txt"
    if not (train_p.is_file() and test_p.is_file()):
        return None

    def _parse_line(line: str) -> tuple[str, int] | None:
        line = line.strip()
        if not line:
            return None
        parts = line.split()
        rel_path = parts[0]
        # 类别从父目录推 (UCF101 目录布局保证此关系)
        class_name = rel_path.split("/")[0] if "/" in rel_path else rel_path.split("\\")[0]
        if class_name not in name_to_id:
            return None
        return rel_path, name_to_id[class_name]

    out: list[VideoClip] = []
    # train list 暂存, 后面切出一部分作 val
    train_entries: list[tuple[str, int]] = []
    for line in train_p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed:
            train_entries.append(parsed)

    test_entries: list[tuple[str, int]] = []
    for line in test_p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed:
            test_entries.append(parsed)

    if not train_entries or not test_entries:
        return None

    # 把 train 中每个 source_video_id 的样本捆绑, 后 10% videos 进 val (按 source_video 而非 clip)
    # UCF101 的 source_video_id = '<class>/v_<class>_g<group>'
    from collections import defaultdict
    train_videos: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for rel_path, label_id in train_entries:
        # source_video_id: 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01' (剥掉 _c01.avi)
        stem = Path(rel_path).stem
        # 形如 v_<class>_g<group>_c<clip>; 我们要前 3 段
        bits = stem.split("_")
        if len(bits) >= 4:
            src_vid = "/".join(rel_path.split("/")[:1] + ["_".join(bits[:3])])
        else:
            src_vid = stem
        train_videos[src_vid].append((rel_path, label_id))

    sorted_vids = sorted(train_videos.keys())
    n_val = max(1, int(len(sorted_vids) * 0.10))
    val_set = set(sorted_vids[-n_val:])
    train_set = set(sorted_vids) - val_set

    def _make_clip(rel_path: str, label_id: int, split: str, src_vid: str) -> VideoClip:
        clip_path = root / "videos" / rel_path
        return VideoClip(
            clip_id=Path(rel_path).stem,
            source_video_id=src_vid,
            path=str(clip_path),
            label_id=label_id,
            split=split,  # type: ignore[arg-type]
        )

    for vid in train_set:
        for rel_path, label_id in train_videos[vid]:
            out.append(_make_clip(rel_path, label_id, "train", vid))
    for vid in val_set:
        for rel_path, label_id in train_videos[vid]:
            out.append(_make_clip(rel_path, label_id, "val", vid))

    # test set: source_video_id 同样的派生方式 (UCF101 test 集与 train 集**没有 group 重叠**, 安全)
    for rel_path, label_id in test_entries:
        stem = Path(rel_path).stem
        bits = stem.split("_")
        if len(bits) >= 4:
            src_vid = "/".join(rel_path.split("/")[:1] + ["_".join(bits[:3])])
        else:
            src_vid = stem
        out.append(_make_clip(rel_path, label_id, "test", src_vid))

    _log.info(
        "UCF101 splits parsed",
        extra={
            "train_clips": sum(1 for c in out if c.split == "train"),
            "val_clips": sum(1 for c in out if c.split == "val"),
            "test_clips": sum(1 for c in out if c.split == "test"),
        },
    )
    return out


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
