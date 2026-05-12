"""``pp data-prepare`` 子命令实现 (FR-004 ~ FR-007).

编排:
    1. 加载 dataset 配置 YAML (走 :func:`pingpong_av.utils.config.load_config`,
       但 dataset config 里没有 ``model.name``, 因此用 ``validate=False`` + 手动校验);
    2. 调用 :func:`pingpong_av.data.public_datasets.fetch_and_discover` 把数据落地并扫描;
    3. 若数据集自带官方 split, 直接使用; 否则按 config 里的 ``ratios`` + ``seed`` 划分;
    4. **章程 IV 闸门**: :func:`verify_no_leakage` — 失败 → 退出码 3, 不写 list 文件;
    5. :func:`write_paddlevideo_lists` 落盘 ``data/splits/{train,val,test}.txt`` + meta jsonl;
    6. stdout 输出 contracts/cli.md 约定的 JSON 摘要.

退出码 (与 contracts/cli.md 对齐):
    0  成功
    1  用户输入错 (配置不存在 / 参数非法 / 数据源 URL 不可达)
    3  章程硬约束违反 (划分泄漏)
    4  运行时失败 (扫描到 0 个片段等)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from pingpong_av.data.list_writer import write_paddlevideo_lists
from pingpong_av.data.public_datasets import (
    DatasetFetchError,
    DatasetNeedsManualSetup,
    FetchResult,
    fetch_and_discover,
)
from pingpong_av.data.splitter import (
    Splits,
    VideoClip,
    split_by_video_id,
    verify_no_leakage,
)
from pingpong_av.experiment.run_manifest import ConstitutionViolation
from pingpong_av.utils.config import ConfigError, _validate as _validate_full_required  # type: ignore  # noqa: WPS450
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(*, config_path: str, force: bool) -> int:
    """执行 data-prepare. 返回应当作为进程退出码使用的整数."""

    # ---- 1. 加载 dataset 配置 ----
    cfg_path = Path(config_path).resolve()
    if not cfg_path.is_file():
        click.echo(f"ERROR: 配置文件不存在: {cfg_path}", err=True)
        return 1
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        click.echo(f"ERROR: YAML 解析失败 ({cfg_path}): {exc}", err=True)
        return 1
    if not isinstance(cfg_data, dict):
        click.echo(f"ERROR: 顶层配置必须是 mapping, 实际为 {type(cfg_data).__name__}", err=True)
        return 1

    # dataset 配置不需要 model.name; 但 classes / split_version 必填. 复用部分校验:
    try:
        _validate_dataset_config(cfg_data)
    except ConfigError as exc:
        click.echo(f"ERROR: 配置校验失败: {exc}", err=True)
        return 1

    repo_root = find_repo_root()

    # ---- 2. 拉取 + 扫描 ----
    try:
        result: FetchResult = fetch_and_discover(cfg_data, repo_root=repo_root, force=force)
    except DatasetNeedsManualSetup as exc:
        # source.type=manual 时哨兵文件缺失; 多行指引保留原样输出, 不被压成单行.
        click.echo("ERROR: 数据集需要手动准备:", err=True)
        for line in str(exc).splitlines():
            click.echo("  " + line if line else "", err=True)
        return 1
    except DatasetFetchError as exc:
        click.echo(f"ERROR: 数据集准备失败: {exc}", err=True)
        return 1
    except Exception as exc:  # 兜底, 不让 traceback 直接糊到 stderr
        click.echo(f"ERROR: 数据集准备时出现意外错误: {type(exc).__name__}: {exc}", err=True)
        return 4

    if not result.clips:
        click.echo("ERROR: 扫描到 0 个视频片段, 无法继续.", err=True)
        return 4

    # ---- 3. 划分: 官方 / by_video_ratio ----
    strategy = (cfg_data.get("split_strategy") or "official").lower()

    if result.used_official_split:
        # 数据集自带官方 split; clip.split 已被 public_datasets 设置
        train = [c for c in result.clips if c.split == "train"]
        val = [c for c in result.clips if c.split == "val"]
        test = [c for c in result.clips if c.split == "test"]
        splits = Splits(train=train, val=val, test=test)
        _log.info("using official split files from dataset", extra=splits.counts())
    elif strategy in ("by_video_ratio", "official"):
        # official 策略但数据集没提供 split 文件; 退化到按视频 id 划分.
        ratios = cfg_data.get("ratios") or {"train": 0.7, "val": 0.15, "test": 0.15}
        seed = int(cfg_data.get("seed", 2026))
        try:
            splits = split_by_video_id(result.clips, ratios=ratios, seed=seed)
        except (ValueError, KeyError) as exc:
            click.echo(f"ERROR: 划分参数非法: {exc}", err=True)
            return 1
    else:
        click.echo(f"ERROR: 未知的 split_strategy: {strategy!r}", err=True)
        return 1

    # ---- 4. 章程 IV 闸门 ----
    try:
        verify_no_leakage(splits)
    except ConstitutionViolation as exc:
        click.echo("ERROR (章程 IV 违反): " + str(exc), err=True)
        return 3

    # ---- 4.5. 章程 IV 软提醒: 新增类别但 split_version 未 bump (T069) ----
    _warn_class_table_changed_without_version_bump(
        cfg_data, repo_root=repo_root,
    )

    # ---- 5. 落盘 list 文件 + meta jsonl ----
    splits_dir = repo_root / Path((cfg_data.get("paths") or {}).get("splits_dir", "data/splits"))
    written = write_paddlevideo_lists(splits, splits_dir, relative_to=result.clips_dir)

    # ---- 6. stdout JSON 摘要 (contracts/cli.md) ----
    payload = {
        "dataset": cfg_data.get("name", cfg_path.stem),
        "split_version": cfg_data.get("split_version", "unknown"),
        "counts": splits.counts(),
        "num_classes": result.n_classes,
        "list_files": [_relpath(p, repo_root) for k, p in written.items() if k.endswith(".txt")],
        "meta_files": [_relpath(p, repo_root) for k, p in written.items() if k.endswith(".jsonl")],
        "used_official_split": result.used_official_split,
    }
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    click.echo(f"✓ 数据准备完成: {sum(splits.counts().values())} 个片段, {result.n_classes} 个类别.", err=True)

    # 类别样本量警告 (validation.min_samples_per_class)
    min_samples = (cfg_data.get("validation") or {}).get("min_samples_per_class", 0)
    if min_samples and min_samples > 0:
        _warn_low_class_counts(splits, min_samples)

    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _relpath(p: Path, base: Path) -> str:
    """尽量返回相对 base 的路径; 若 p 不在 base 子树下则原样输出绝对路径."""
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def _validate_dataset_config(d: dict) -> None:
    """dataset 配置必填字段校验 (与 utils.config 的全量校验不同, 此处不要求 model.name)."""
    classes = d.get("classes")
    if not classes or not isinstance(classes, list):
        raise ConfigError("dataset 配置必须含非空的 classes 列表 (章程 III)")
    seen_ids: set[int] = set()
    for i, c in enumerate(classes):
        if not isinstance(c, dict) or "id" not in c or "name" not in c:
            raise ConfigError(f"classes[{i}] 必须含 id 与 name, 实际为 {c!r}")
        cid = c["id"]
        if not isinstance(cid, int) or cid < 0 or cid in seen_ids:
            raise ConfigError(f"classes[{i}].id 非法或重复: {cid!r}")
        seen_ids.add(cid)
    if seen_ids != set(range(len(classes))):
        raise ConfigError(f"classes id 必须从 0 连续到 {len(classes)-1}, 实际 {sorted(seen_ids)}")
    if "split_version" not in d:
        raise ConfigError("dataset 配置必须含 split_version (章程 IV: 重新划分必须 bump)")


def _warn_low_class_counts(splits: Splits, min_samples: int) -> None:
    from collections import Counter

    train_counts = Counter(c.label_id for c in splits.train)
    low = sorted(
        (label, count) for label, count in train_counts.items() if count < min_samples
    )
    if low:
        click.echo(
            f"⚠ 警告: 以下 {len(low)} 个类别在 train 中样本数 < {min_samples}: "
            + ", ".join(f"id={lid}({n})" for lid, n in low),
            err=True,
        )


def _warn_class_table_changed_without_version_bump(
    cfg_data: dict, *, repo_root: Path,
) -> None:
    """T069 (章程 IV): 检测 classes 集合变化但 split_version 未 bump.

    判定方式:
        - 优先读取 ``<repo_root>/data/splits/.last_class_table.json`` (本函数维护的元信息);
        - 比较新旧 ``classes`` 的 (id, name) 集合;
        - 同时 split_version 也未变 → 打 warning (软提醒, 不阻断).
        - 划分完成后写入 last_class_table.json 供下次比较.
    """
    splits_dir = repo_root / Path(
        (cfg_data.get("paths") or {}).get("splits_dir", "data/splits")
    )
    record_path = splits_dir / ".last_class_table.json"

    new_table = sorted(
        (int(c["id"]), str(c["name"])) for c in cfg_data["classes"]
    )
    new_version = str(cfg_data.get("split_version", ""))

    if record_path.is_file():
        try:
            prev = json.loads(record_path.read_text(encoding="utf-8"))
            prev_table = sorted(tuple(x) for x in prev.get("classes", []))
            prev_version = str(prev.get("split_version", ""))
            if prev_table != new_table and new_version == prev_version:
                added = [n for (i, n) in new_table if (i, n) not in prev_table]
                removed = [n for (i, n) in prev_table if (i, n) not in new_table]
                hints = []
                if added:
                    hints.append(f"新增 {added}")
                if removed:
                    hints.append(f"删除 {removed}")
                click.echo(
                    "⚠ 警告 (章程 IV 软提醒): 类别表已变化但 split_version 未 bump.\n"
                    f"  当前 split_version={new_version!r}, "
                    + "; ".join(hints) + "\n"
                    "  按章程 IV, 类别变化等同于"
                    "新实验, 请把 split_version 改成新值 (例: v1.1) 后重跑.",
                    err=True,
                )
        except (OSError, json.JSONDecodeError):
            pass

    # 写入 (覆盖) 当次记录
    try:
        splits_dir.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {"split_version": new_version, "classes": new_table},
                ensure_ascii=False, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
