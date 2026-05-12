"""utils/config.py 单元测试.

覆盖范围:
- !include 相对路径展开
- config_hash 稳定性 (相同语义内容 → 相同 hash)
- 不同内容 → 不同 hash
- 必填字段缺失 → ConfigError (章程 III)
- classes id 不连续 / 重复 → ConfigError (章程 III)
- 循环 include 检测
- 文件不存在的清晰错误信息

不在本测试范围:
- 实际 paddle / paddlevideo 依赖 (本模块零运行时依赖)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pingpong_av.utils.config import ConfigError, compute_config_hash, load_config


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def valid_classes_yaml(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "classes_list.yaml",
        """
- {id: 0, name: serve, display_name: "发球"}
- {id: 1, name: forehand_attack, display_name: "正手攻"}
- {id: 2, name: backhand_attack, display_name: "反手攻"}
""",
    )


@pytest.fixture()
def valid_config(tmp_path: Path, valid_classes_yaml: Path) -> Path:
    return _write(
        tmp_path / "model.yaml",
        f"""
classes: !include {valid_classes_yaml.name}
split_version: v0.1
model:
  name: pp_tsm
  backbone: ResNet50
seed: 2026
""",
    )


# --------------------------------------------------------------------------------------
# 基础加载与 !include
# --------------------------------------------------------------------------------------


def test_load_config_with_include(valid_config: Path, valid_classes_yaml: Path) -> None:
    cfg = load_config(valid_config)
    assert isinstance(cfg.data["classes"], list)
    assert len(cfg.data["classes"]) == 3
    assert cfg.data["classes"][0]["name"] == "serve"
    # included_paths 应包含主文件 + 被 include 的文件
    names = {p.name for p in cfg.included_paths}
    assert valid_config.name in names
    assert valid_classes_yaml.name in names


def test_config_hash_is_stable(valid_config: Path) -> None:
    cfg1 = load_config(valid_config)
    cfg2 = load_config(valid_config)
    assert cfg1.config_hash == cfg2.config_hash
    assert len(cfg1.config_hash) == 16


def test_config_hash_changes_with_content(valid_config: Path, tmp_path: Path) -> None:
    base_hash = load_config(valid_config).config_hash
    # 改 split_version 一个字符
    text = valid_config.read_text(encoding="utf-8").replace("v0.1", "v0.2")
    valid_config.write_text(text, encoding="utf-8")
    new_hash = load_config(valid_config).config_hash
    assert base_hash != new_hash


def test_config_hash_independent_of_yaml_format(tmp_path: Path) -> None:
    """同样语义、不同字段顺序的两份 YAML, hash 应该一致."""
    a = _write(
        tmp_path / "a.yaml",
        """
seed: 2026
split_version: v0.1
classes:
  - {id: 0, name: x}
  - {id: 1, name: y}
model:
  name: pp_tsm
""",
    )
    b = _write(
        tmp_path / "b.yaml",
        """
model: {name: pp_tsm}
classes:
  - id: 0
    name: x
  - id: 1
    name: y
split_version: v0.1
seed: 2026
""",
    )
    assert load_config(a).config_hash == load_config(b).config_hash


# --------------------------------------------------------------------------------------
# 必填校验 (章程 III)
# --------------------------------------------------------------------------------------


def test_missing_split_version_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "no_split.yaml",
        """
classes: [{id: 0, name: a}]
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="split_version"):
        load_config(p)


def test_missing_model_name_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "no_model.yaml",
        """
classes: [{id: 0, name: a}]
split_version: v0.1
""",
    )
    with pytest.raises(ConfigError, match="model.name"):
        load_config(p)


def test_classes_must_be_nonempty(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "empty.yaml",
        """
classes: []
split_version: v0.1
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="非空列表"):
        load_config(p)


def test_classes_ids_must_be_continuous(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "skip.yaml",
        """
classes:
  - {id: 0, name: a}
  - {id: 2, name: c}
split_version: v0.1
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="连续编号"):
        load_config(p)


def test_classes_ids_no_duplicate(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "dup_id.yaml",
        """
classes:
  - {id: 0, name: a}
  - {id: 0, name: b}
split_version: v0.1
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="重复"):
        load_config(p)


def test_classes_names_no_duplicate(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "dup_name.yaml",
        """
classes:
  - {id: 0, name: same}
  - {id: 1, name: same}
split_version: v0.1
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="重复"):
        load_config(p)


# --------------------------------------------------------------------------------------
# 错误路径
# --------------------------------------------------------------------------------------


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_config(tmp_path / "no_such.yaml")


def test_include_target_missing(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "main.yaml",
        """
classes: !include does_not_exist.yaml
split_version: v0.1
model: {name: pp_tsm}
""",
    )
    with pytest.raises(ConfigError, match="不存在"):
        load_config(p)


def test_include_circular(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("foo: !include b.yaml\n", encoding="utf-8")
    b.write_text("bar: !include a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="循环引用"):
        load_config(a)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path / "list.yaml", "- a\n- b\n")
    with pytest.raises(ConfigError, match="顶层"):
        load_config(p)


def test_validate_false_skips_required(tmp_path: Path) -> None:
    """validate=False 用于 helper 子流程; 缺必填不应抛错."""
    p = _write(tmp_path / "partial.yaml", "anything: 1\n")
    cfg = load_config(p, validate=False)
    assert cfg.data == {"anything": 1}


# --------------------------------------------------------------------------------------
# compute_config_hash 直接测试
# --------------------------------------------------------------------------------------


def test_compute_config_hash_prefix_length() -> None:
    h = compute_config_hash({"a": 1}, prefix_len=8)
    assert len(h) == 8
    h2 = compute_config_hash({"a": 1}, prefix_len=32)
    assert h2.startswith(h)
