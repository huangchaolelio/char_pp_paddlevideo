"""模型注册表 (扩展槽位).

本项目 MVP 仅注册 ``pp_tsm`` 一种模型. 增加新模型 (例如 PP-TSN / SlowFast)
只需:
    1) 实现 ``models/<new_model>.py``, 暴露与 :func:`pingpong_av.models.pp_tsm.load_pp_tsm_config`
       签名一致的 ``load_<new_model>_config`` 函数;
    2) 在 :data:`_REGISTRY` 中注册新条目.

不在本模块的范围:
- 模型权重的下载与加载 (上游负责).
- 训练循环 (``upstream_adapter.trainer``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from pingpong_av.models import pp_tsm

__all__ = ["get_model_loader", "list_supported_models", "ModelLoader"]

# 形如 load_*_config(user_cfg, ...) -> tuple[Path, dict]
ModelLoader = Callable[..., tuple[Any, dict[str, Any]]]


_REGISTRY: dict[str, ModelLoader] = {
    "pp_tsm": pp_tsm.load_pp_tsm_config,
}


def get_model_loader(name: str) -> ModelLoader:
    """根据 model.name 返回对应的 loader 函数.

    抛出 :class:`KeyError` 并给出已注册模型清单, 便于排查配置写错.
    """
    if name not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY.keys())) or "(none registered)"
        raise KeyError(
            f"未注册的模型: {name!r}; 当前支持: {supported}. "
            "增加新模型请在 src/pingpong_av/models/registry.py 中注册."
        )
    return _REGISTRY[name]


def list_supported_models() -> list[str]:
    return sorted(_REGISTRY.keys())
