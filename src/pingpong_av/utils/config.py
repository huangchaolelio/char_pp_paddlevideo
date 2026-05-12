"""YAML 配置加载器 (章程 III: 配置驱动, 拒绝硬编码).

特性:
- 支持 ``!include <相对路径>`` 引入其他 YAML, 实现 dataset/model 配置的 modular 复用
  (research.md R4: 训练配置通过 !include 注入数据集类别表, 避免双向耦合).
- 加载后计算 ``config_hash`` (SHA256 前 16 位), 用于 manifest 四元组 (章程 II).
- 必填字段校验: ``classes`` (类别表), ``split_version`` (划分版本), ``model.name`` (模型名).
  缺失任意必填项抛 ``ConfigError``, 由 CLI 层映射到退出码 1 (用户输入错误).

不在本模块的范围:
- 与 PaddleVideo 训练入口适配的字段映射 (那是 ``models/pp_tsm.py`` 的职责).
- 配置 snapshot 写入实验目录 (那是 ``experiment.run_manifest`` 的职责).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ConfigError", "LoadedConfig", "load_config"]


class ConfigError(ValueError):
    """配置加载或校验失败."""


@dataclass(frozen=True)
class LoadedConfig:
    """加载结果: 合并后的配置 + 稳定哈希 + 解析路径.

    config_hash 计算自 ``data`` 的规范化 JSON 序列化 (sort_keys=True), 因此对
    YAML 注释 / 字段顺序变化不敏感, 只要语义内容相同, 哈希就稳定.
    """

    data: dict[str, Any]
    config_hash: str
    source_path: Path
    included_paths: list[Path]


# --------------------------------------------------------------------------------------
# !include 标签实现
# --------------------------------------------------------------------------------------


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader 子类, 支持 ``!include <相对路径>`` 标签.

    实例上携带 ``_base_dir`` 与 ``_visited`` 用于解析相对路径与防止循环 include.
    """

    _base_dir: Path
    _visited: set[Path]
    _included: list[Path]


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    if not isinstance(node, yaml.ScalarNode):
        raise ConfigError(
            f"!include 期望一个标量字符串路径, 而非 {type(node).__name__}; 位置: {node.start_mark}"
        )
    rel = loader.construct_scalar(node)
    if not isinstance(rel, str) or not rel.strip():
        raise ConfigError(f"!include 的路径不能为空; 位置: {node.start_mark}")

    target = (loader._base_dir / rel).resolve()
    if not target.is_file():
        raise ConfigError(f"!include 指向的文件不存在: {target} (位置: {node.start_mark})")
    if target in loader._visited:
        raise ConfigError(
            f"!include 出现循环引用: {target} 已在加载链 {sorted(loader._visited)} 中"
        )

    return _load_yaml_file(target, parent_visited=loader._visited, parent_included=loader._included)


_IncludeLoader.add_constructor("!include", _construct_include)


def _load_yaml_file(
    path: Path, *, parent_visited: set[Path] | None = None, parent_included: list[Path] | None = None
) -> Any:
    """读取一个 YAML 文件, 解析其中可能存在的 !include 节点."""
    visited = set(parent_visited or ())
    visited.add(path)

    included = parent_included if parent_included is not None else []
    if path not in included:
        included.append(path)

    text = path.read_text(encoding="utf-8")

    # 用一次性子类避免污染全局 _IncludeLoader 的状态
    class _Bound(_IncludeLoader):
        pass

    _Bound._base_dir = path.parent.resolve()
    _Bound._visited = visited
    _Bound._included = included

    try:
        return yaml.load(text, Loader=_Bound) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败 ({path}): {exc}") from exc


# --------------------------------------------------------------------------------------
# 必填项校验
# --------------------------------------------------------------------------------------

# 必填字段路径 (用 "." 表示嵌套). 每条都对应一条章程或 FR.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "classes",         # 章程 III: 类别表必须由配置提供, 不在源码硬编码
    "split_version",   # 章程 IV: 划分一旦发布, 必须有版本; 重新划分视为新实验
    "model.name",      # 章程 III: 模型名必须显式
)


def _has_path(d: dict[str, Any], dotted: str) -> bool:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _validate(merged: dict[str, Any]) -> None:
    missing = [p for p in _REQUIRED_FIELDS if not _has_path(merged, p)]
    if missing:
        raise ConfigError(
            "配置缺少必填字段 (章程 III/IV): "
            + ", ".join(missing)
            + ". 请在 configs/ 下的 YAML 中补全."
        )

    # classes 必须是非空列表, 每项至少含 id 与 name
    classes = merged["classes"]
    if not isinstance(classes, list) or not classes:
        raise ConfigError("classes 必须是非空列表")
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for i, item in enumerate(classes):
        if not isinstance(item, dict) or "id" not in item or "name" not in item:
            raise ConfigError(f"classes[{i}] 必须含 id 与 name 字段, 实际为 {item!r}")
        cid = item["id"]
        cname = item["name"]
        if not isinstance(cid, int) or cid < 0:
            raise ConfigError(f"classes[{i}].id 必须是非负整数, 实际为 {cid!r}")
        if not isinstance(cname, str) or not cname:
            raise ConfigError(f"classes[{i}].name 必须是非空字符串, 实际为 {cname!r}")
        if cid in seen_ids:
            raise ConfigError(f"classes[{i}].id={cid} 重复")
        if cname in seen_names:
            raise ConfigError(f"classes[{i}].name={cname!r} 重复")
        seen_ids.add(cid)
        seen_names.add(cname)

    # id 必须从 0 连续递增到 N-1
    expected = set(range(len(classes)))
    if seen_ids != expected:
        raise ConfigError(
            f"classes 的 id 必须从 0 连续编号到 {len(classes)-1}, 实际为 {sorted(seen_ids)}"
        )


# --------------------------------------------------------------------------------------
# config_hash
# --------------------------------------------------------------------------------------


def compute_config_hash(data: dict[str, Any], *, prefix_len: int = 16) -> str:
    """对配置内容计算稳定的 SHA256 短哈希.

    通过 ``json.dumps(..., sort_keys=True, ensure_ascii=False)`` 规范化, 因此:
    - 同样语义、不同字段顺序的配置产生相同哈希;
    - YAML 注释、缩进风格的差异不会影响哈希.
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:prefix_len]


# --------------------------------------------------------------------------------------
# 公共入口
# --------------------------------------------------------------------------------------


def load_config(path: str | Path, *, validate: bool = True) -> LoadedConfig:
    """加载并合并 YAML 配置, 解析所有 ``!include``, 计算 config_hash.

    参数:
        path: 入口 YAML 文件路径.
        validate: True 时执行 :func:`_validate` 必填项校验; 单元测试或 CLI helper 子流程
                  (例如未合并 dataset 时) 可置 False.

    返回:
        :class:`LoadedConfig`.

    抛出:
        ConfigError: 文件不存在 / YAML 解析失败 / 必填字段缺失.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")

    included: list[Path] = []
    data = _load_yaml_file(p, parent_included=included)

    if not isinstance(data, dict):
        raise ConfigError(
            f"顶层配置必须是 mapping (即 YAML 字典), 实际为 {type(data).__name__}"
        )

    if validate:
        _validate(data)

    return LoadedConfig(
        data=data,
        config_hash=compute_config_hash(data),
        source_path=p,
        included_paths=list(included),
    )
