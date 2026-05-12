"""可复现性回归测试 (T074, 章程 II).

T074 原意是"同 seed 跑两次 `pp train`, 比较 manifest 与 metrics.json 一致性"; 但由于:
  - PaddleVideo 官方乒乓球数据集需用户手动从 AI Studio 注册下载 (R2 修正版)
  - 上游 decord 0.4.x 无 Python 3.11 wheel (T056 待 patch)

端到端真实训练当前不可在 CI 中运行. 因此本测试覆盖**可复现性的根本组件**:

1. ``set_seed(s)`` 在相同 ``s`` 下让 Python random / numpy / hashlib 抽样得到相同结果
2. ``compute_config_hash`` 在相同 dict 下产物字节级相同, 不同 dict 必然变化
3. ``RunManifest`` 序列化在固定输入下产物可比较 (除时间戳与运行时探测字段外)

端到端的真实回归在 ``specs/.../checklists/reproducibility-checklist.md`` 中作为 manual 闸门.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from pingpong_av.experiment.run_manifest import RunManifest
from pingpong_av.utils.config import compute_config_hash
from pingpong_av.utils.seeding import set_seed


# --------------------------------------------------------------------------------------
# 1. set_seed 跨次调用一致性 (章程 II 核心)
# --------------------------------------------------------------------------------------


def test_set_seed_python_random_reproducible() -> None:
    """同 seed 下 Python random 必须产出相同序列."""
    set_seed(2026)
    a = [random.random() for _ in range(20)]
    set_seed(2026)
    b = [random.random() for _ in range(20)]
    assert a == b, "Python random 在同 seed 下产出不一致 — 违反章程 II"


def test_set_seed_numpy_reproducible() -> None:
    """同 seed 下 numpy 必须产出相同序列."""
    set_seed(2026)
    a = np.random.rand(50).tolist()
    set_seed(2026)
    b = np.random.rand(50).tolist()
    assert a == b, "NumPy 在同 seed 下产出不一致 — 违反章程 II"


def test_different_seeds_give_different_sequences() -> None:
    """不同 seed 必须产出不同序列 (反证 set_seed 真起作用)."""
    set_seed(1)
    a = np.random.rand(10).tolist()
    set_seed(2)
    b = np.random.rand(10).tolist()
    assert a != b


def test_set_seed_returns_seed_value() -> None:
    """set_seed 应返回含 seed 字段的结果, 便于 manifest 记录."""
    r = set_seed(42)
    assert r.seed == 42
    r = set_seed(0)
    assert r.seed == 0


# --------------------------------------------------------------------------------------
# 2. compute_config_hash 确定性 (章程 II 四元组核心组件)
# --------------------------------------------------------------------------------------


def test_config_hash_deterministic_for_same_dict() -> None:
    """同样的 dict 必须产出同样的 hash (字节级)."""
    cfg = {"model": {"name": "pp_tsm"}, "seed": 2026, "classes": [{"id": 0, "name": "a"}]}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 16  # 16-char prefix per data-model.md


def test_config_hash_changes_when_value_changes() -> None:
    cfg1 = {"model": {"name": "pp_tsm"}, "seed": 2026}
    cfg2 = {"model": {"name": "pp_tsm"}, "seed": 2027}
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


def test_config_hash_key_order_independent() -> None:
    """{a, b} 和 {b, a} 应该 hash 相同 (sort_keys 在内部)."""
    cfg1 = {"a": 1, "b": 2, "c": [3, 4]}
    cfg2 = {"c": [3, 4], "b": 2, "a": 1}
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)


def test_config_hash_changes_when_nested_changes() -> None:
    cfg1 = {"model": {"backbone": {"depth": 50}}}
    cfg2 = {"model": {"backbone": {"depth": 101}}}
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


def test_config_hash_list_order_matters() -> None:
    """类别表 [a,b] 与 [b,a] 应被视为不同 (顺序决定 label_id)."""
    cfg1 = {"classes": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]}
    cfg2 = {"classes": [{"id": 0, "name": "b"}, {"id": 1, "name": "a"}]}
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


# --------------------------------------------------------------------------------------
# 3. RunManifest 在相同输入下的可比较性 (除时间戳/探测字段外)
# --------------------------------------------------------------------------------------


def test_manifest_json_serialization_deterministic(tmp_path: Path) -> None:
    """同样的 RunManifest 反复序列化应得到相同 JSON 字符串 (sort_keys=True)."""
    m = RunManifest(
        run_id="20260512-deadbee-train-test",
        kind="train",
        config_hash="0123456789abcdef",
        commit="deadbeefcafebabe",
        seed=2026,
        dataset_split_version="v0.1",
        python_version="3.11.15",
        cuda_version="11.8",
        gpu_model="Tesla T4",
        dirty=False,
        started_at="2026-05-12T00:00:00+00:00",
    )
    a = json.dumps(m.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    b = json.dumps(m.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    assert a == b, "RunManifest 序列化在同输入下产出不一致"

    # 跨进程模拟: 写盘后再读
    p = tmp_path / "manifest.json"
    p.write_text(a, encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    # 章程 II 四元组必须完整
    for k in ("commit", "config_hash", "seed", "dataset_split_version"):
        assert k in loaded, f"manifest 缺少章程 II 关键字段 {k}"


def test_manifest_four_tuple_required_for_constitution_ii() -> None:
    """章程 II 四元组: commit + config_hash + seed + dataset_split_version 不可省."""
    # 任意一个为 None 都应在 to_json_text 时仍然保留为 None (后续 finalize 写入)
    m = RunManifest(
        run_id="x", kind="train",
        config_hash="abc", commit="def", seed=1,
        dataset_split_version="v0.1",
        python_version="3.11.0", cuda_version=None, gpu_model=None,
        dirty=False, started_at="2026-05-12T00:00:00+00:00",
    )
    d = m.to_dict()
    # 四元组全在
    assert d["commit"] == "def"
    assert d["config_hash"] == "abc"
    assert d["seed"] == 1
    assert d["dataset_split_version"] == "v0.1"


# --------------------------------------------------------------------------------------
# 4. 完整的 set_seed → config_hash → manifest 链路一致性 (模拟 pp train 启动序列)
# --------------------------------------------------------------------------------------


def test_full_reproducibility_chain_simulation() -> None:
    """模拟 cli/train.py 的启动序列: 同 config + 同 seed 两次, 关键值必须一致."""

    def _start_train(config: dict, seed: int) -> dict:
        """模拟 cli/train.py 的前 3 步."""
        cfg_hash = compute_config_hash(config)
        set_seed(seed)
        sample = np.random.rand(5).tolist()
        return {
            "config_hash": cfg_hash,
            "seed": seed,
            "first_5_random": sample,
        }

    cfg = {"model": {"name": "pp_tsm"}, "train": {"epochs": 10}}
    r1 = _start_train(cfg, seed=2026)
    r2 = _start_train(cfg, seed=2026)

    assert r1 == r2, (
        "同 config + 同 seed 两次启动产物不一致 — 这是章程 II 的根本违反:\n"
        f"  r1: {r1}\n  r2: {r2}"
    )

    # 反证: 不同 seed 必有不同 random sample
    r3 = _start_train(cfg, seed=2027)
    assert r3["config_hash"] == r1["config_hash"]  # config 没变
    assert r3["first_5_random"] != r1["first_5_random"]
