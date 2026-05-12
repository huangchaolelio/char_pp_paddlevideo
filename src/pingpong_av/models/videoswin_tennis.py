"""PaddleVideo 官方乒乓球 VideoSwin 模型加载与推理 (release/2.2.0 release2.2 BCEBOS).

上游入口位置:
    third_party/PaddleVideo/applications/TableTennis/ActionRecognition/configs/
        videoswin_tabletennis.yaml
    模型架构: RecognizerTransformer + SwinTransformer3D + I3DHead
    预训练权重: VideoSwin_tennis.pdparams (380MB, 由 source.smoke_sample 之外用户单独下载)
    样例数据: example_tennis.pkl (7.4MB, 已通过 manual 模式下载到 data/raw/pingpong_public/smoke/)

本模块**不修改**上游配置, 而是:
- 读取 ``third_party/PaddleVideo/applications/TableTennis/ActionRecognition/configs/
  videoswin_tabletennis.yaml`` 作为基线
- 调用上游 ``build_model`` 构建模型
- 加载我们事先下载的 VideoSwin_tennis.pdparams checkpoint
- 提供 :func:`load_videoswin_tennis_model` 给 :mod:`cli.infer_pkl` 调用

不在本模块的范围:
- pkl 解码 / 帧采样 / 预处理 (那是 :mod:`cli.infer_pkl` 的职责)
- 训练流程 (上游训练入口与本项目 PP-TSM 主线分离)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

__all__ = [
    "load_videoswin_tennis_model",
    "TABLETENNIS_CLASS_NAMES",
    "TABLETENNIS_LABEL_GROUPS",
    "TabletennisModelError",
]

_log = get_logger(__name__)


# 上游 README 未公布具体类别名, 但 example_tennis.pkl 的 labels 字段揭示了**三组任务**:
#   - 正反手 (forehand/backhand): 二分类, id ∈ {0, 1}
#   - 动作类型 (action type):     8 分类, id ∈ {0..7}
#   - 发球 (serve):               二分类, id ∈ {0, 1}
# 但 yaml 中 num_classes=8, head 是单 I3DHead — 说明上游训练目标是"动作类型".
# 其余两个标签字段保留在 pkl 中供其他多任务实验.
TABLETENNIS_LABEL_GROUPS = ("正反手", "动作类型", "发球")

# 8 个动作类别 (动作类型) — 上游 README 未公布, 这里用占位
# (用户在 AI Studio 拿到 metadata 后应该来更新)
TABLETENNIS_CLASS_NAMES: list[str] = [
    "动作0", "动作1", "动作2", "动作3",
    "动作4", "动作5", "动作6", "动作7",
]


UPSTREAM_CONFIG_REL = Path(
    "third_party/PaddleVideo/applications/TableTennis/ActionRecognition/configs/"
    "videoswin_tabletennis.yaml"
)


class TabletennisModelError(RuntimeError):
    """VideoSwin tennis 模型加载失败."""


def load_videoswin_tennis_model(
    *,
    checkpoint: str | Path,
    repo_root: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """加载 PaddleVideo 官方乒乓球 VideoSwin 模型 + 预训练权重.

    返回 ``(model, upstream_cfg_dict)``:
        - model: 已加载权重的 paddle.nn.Layer (eval 模式由调用方负责)
        - upstream_cfg_dict: 上游 get_config 返回的完整 AttrDict, 供调用方读 PIPELINE / METRIC 等
    """
    repo_root = repo_root or find_repo_root()
    upstream_yaml = repo_root / UPSTREAM_CONFIG_REL
    if not upstream_yaml.is_file():
        raise TabletennisModelError(
            f"上游 VideoSwin TableTennis 配置不存在: {upstream_yaml}; "
            "请确认 third_party/PaddleVideo submodule 已 init."
        )

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise TabletennisModelError(
            f"checkpoint 文件不存在: {checkpoint}\n"
            "请先下载 (380MB):\n"
            "  curl -fL -o data/raw/pingpong_public/checkpoints/VideoSwin_tennis.pdparams \\\n"
            "    https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_tennis.pdparams"
        )

    # 走上游导入器 (sys.path 注入)
    from pingpong_av.upstream_adapter.importer import ensure_paddlevideo_on_path
    ensure_paddlevideo_on_path()

    import paddle
    from paddlevideo.utils import get_config
    from paddlevideo.modeling.builder import build_model

    up_cfg = get_config(str(upstream_yaml), show=False)
    # 不让上游再去下载 K400 预训练权重 (我们要加载的是 VideoSwin_tennis 微调后的权重)
    up_cfg.MODEL.backbone.pretrained = ""

    model = build_model(up_cfg.MODEL)

    # 加载乒乓球 checkpoint
    state = paddle.load(str(checkpoint))
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.set_state_dict(state)
    if missing or unexpected:
        _log.warning(
            "VideoSwin_tennis state_dict 加载有未匹配键",
            extra={
                "n_missing": len(missing) if missing else 0,
                "n_unexpected": len(unexpected) if unexpected else 0,
            },
        )
    _log.info(
        "VideoSwin_tennis model loaded",
        extra={
            "checkpoint": str(checkpoint),
            "num_classes": up_cfg.MODEL.head.num_classes,
            "params_M": sum(int(p.numel()) for p in model.parameters()) / 1e6,
        },
    )
    return model, up_cfg
