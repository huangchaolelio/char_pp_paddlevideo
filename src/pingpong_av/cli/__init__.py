"""``pp`` CLI 主入口 (章程 VII: 端到端 ≤ 5 条命令的统一入口).

本模块仅做**注册与分发**, 不实现业务逻辑. 6 个子命令的具体实现在:

- ``env-check``    → :mod:`pingpong_av.cli.env_check`     (T028 ✓)
- ``data-prepare`` → :mod:`pingpong_av.cli.data_prepare`  (T039 ✓)
- ``train``        → :mod:`pingpong_av.cli.train`         (T044 ✓)
- ``eval``         → :mod:`pingpong_av.cli.eval`          (T049 ✓)
- ``infer-clip``   → :mod:`pingpong_av.cli.infer_clip`    (T054 ✓)
- ``infer-video``  → :mod:`pingpong_av.cli.infer_video`   (T063 ✓)

退出码统一约定 (contracts/cli.md):
    0 成功 · 1 用户输入错 · 2 环境问题 · 3 章程硬约束违反 · 4 运行时失败.
"""

from __future__ import annotations

import sys

import click

from pingpong_av import __version__


# --------------------------------------------------------------------------------------
# 顶层 group
# --------------------------------------------------------------------------------------

@click.group(
    name="pp",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", "-V")
def cli() -> None:
    """pp — 基于 PaddleVideo 的乒乓球视频动作识别 CLI.

    所有命令都必须在项目隔离的 Python 3.11 .venv 中运行 (章程 VIII).
    使用 `pp env-check --strict` 验证环境.
    """


# --------------------------------------------------------------------------------------
# 子命令实现
# 全部 6 个子命令均已通过专用模块接通 (T028/T039/T044/T049/T054/T063);
# env-check 保留 _stub_env_check 作为兜底, 在 cli.env_check 模块缺失时仍可降级运行.
# --------------------------------------------------------------------------------------


@cli.command(name="env-check")
@click.option("--strict", is_flag=True, default=False,
              help="启用时额外尝试 import paddle / paddlevideo, 验证上游已安装.")
def env_check_cmd(strict: bool) -> None:
    """验证 Python 3.11 + 项目隔离环境 + (--strict 时) PaddleVideo 可用 (章程 VIII)."""
    # 不要在这个 stub 里硬退出 — 真正的实现在 cli.env_check (T028); 但我们已经在
    # T012 完成了 utils.env, 所以这里直接接通到那个层, 让 T028 的"业务版"专门
    # 处理 JSON 输出 / 退出码 / 修复指引 — 即便 T028 还没专属文件, 这个 stub
    # 也能工作于 quickstart 第一步.
    try:
        from pingpong_av.cli import env_check as _real
        rc = _real.run(strict=strict)
    except ImportError:
        # T028 还没创建 pingpong_av/cli/env_check.py 时的兜底
        rc = _stub_env_check(strict=strict)
    sys.exit(rc)


def _stub_env_check(*, strict: bool) -> int:
    """env-check 的最小可用兜底实现 (在 T028 接通前直接基于 utils.env 工作).

    这样 quickstart 第 1 步在 T028 之前就能执行, 不会被任务依赖卡住.
    """
    import json

    from pingpong_av.utils import env as env_mod

    results = env_mod.collect_strict() if strict else env_mod.collect_basic()
    payload = {r.name: r.as_dict() for r in results}

    # 把人类可读 hint 打到 stderr, 把结构化结果打到 stdout (符合 contracts/cli.md)
    click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    for r in results:
        if not r.ok and r.hint:
            click.echo(f"[{r.name}] {r.hint}", err=True)

    return 0 if env_mod.all_passed(results) else 2


@cli.command(name="data-prepare")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="数据集配置 YAML (configs/datasets/*.yaml).")
@click.option("--force", is_flag=True, default=False,
              help="忽略已有产物, 强制重新整理.")
def data_prepare_cmd(config_path: str, force: bool) -> None:
    """拉取/整理公开乒乓球动作数据集, 生成训练/验证/测试 list 文件 (FR-004~007)."""
    from pingpong_av.cli import data_prepare as _real
    sys.exit(_real.run(config_path=config_path, force=force))


@cli.command(name="train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="模型 + 训练配置 YAML (configs/models/*.yaml).")
@click.option("--seed", type=int, default=None, help="覆盖配置中的随机种子.")
@click.option("--resume", "resume_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="从指定 checkpoint 恢复训练 (FR-010).")
@click.option("--allow-dirty", is_flag=True, default=False,
              help="允许 git 工作区有未提交修改启动训练 (本次结果不得作为正式指标, 章程 II).")
@click.option("--output-root", type=click.Path(file_okay=False), default="experiments/",
              help="实验输出根目录 (默认 experiments/).")
def train_cmd(config_path: str, seed: int | None, resume_path: str | None,
              allow_dirty: bool, output_root: str) -> None:
    """启动一次训练, 创建 experiments/<run_id>/ 并调用 PaddleVideo (FR-008/009/012/018)."""
    from pingpong_av.cli import train as _real
    sys.exit(_real.run(
        config_path=config_path,
        seed_override=seed,
        resume_path=resume_path,
        allow_dirty=allow_dirty,
        output_root=output_root,
    ))


@cli.command(name="eval")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False),
              help="checkpoint 路径 (.pdparams).")
@click.option("--split", default="test", type=click.Choice(["test", "val"]),
              help="评估使用的数据划分; 默认且只允许 test 或 val (章程 IV).")
@click.option("--batch-size", type=int, default=None, help="覆盖配置中的 test_batch_size.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), default=None,
              help="metrics.json 输出路径; 默认写入 checkpoint 同 run 目录.")
@click.option("--rerun", is_flag=True, default=False,
              help="允许对 split=test 重复评估 (章程 IV: 测试集禁止反复挑选).")
def eval_cmd(checkpoint: str, split: str, batch_size: int | None,
             output_path: str | None, rerun: bool) -> None:
    """在测试集上评估并输出 top1/top5/per-class/macro-avg (章程 V, FR-011)."""
    from pingpong_av.cli import eval as _real
    sys.exit(_real.run(
        checkpoint=checkpoint,
        split=split,
        batch_size=batch_size,
        output_path=output_path,
        rerun=rerun,
    ))


@cli.command(name="infer-clip")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="输入视频片段路径.")
@click.option("--topk", type=int, default=5, help="返回的 Top-K 数量 (默认 5).")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), default=None,
              help="结果 JSON 输出路径; 不指定时只输出到 stdout.")
def infer_clip_cmd(checkpoint: str, input_path: str, topk: int, output_path: str | None) -> None:
    """对单个已切分的视频片段做动作分类 (FR-013)."""
    from pingpong_av.cli import infer_clip as _real
    sys.exit(_real.run(
        checkpoint=checkpoint,
        input_path=input_path,
        topk=topk,
        output_path=output_path,
    ))


@cli.command(name="infer-pkl")
@click.option("--pkl", "pkl_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="PaddleVideo 官方乒乓球样例 pkl (含 video_name + 多任务标签 + JPEG 帧列表).")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False),
              help="VideoSwin_tennis.pdparams (380MB) — 上游 BCEBOS 提供的乒乓球训练权重.")
@click.option("--topk", type=int, default=3, help="返回 Top-K 个预测 (默认 3).")
@click.option("--num-seg", type=int, default=32,
              help="均匀采样的帧数 (默认 32, 与上游 videoswin_tabletennis.yaml runtime_cfg.test.num_seg 对齐).")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), default=None,
              help="结果 JSON 输出路径; 不指定时只输出到 stdout.")
def infer_pkl_cmd(pkl_path: str, checkpoint: str, topk: int, num_seg: int,
                  output_path: str | None) -> None:
    """用上游 VideoSwin TableTennis 模型推理一个 PaddleVideo 样例 pkl 文件."""
    from pingpong_av.cli import infer_pkl as _real
    sys.exit(_real.run(
        pkl_path=pkl_path,
        checkpoint=checkpoint,
        topk=topk,
        num_seg=num_seg,
        output_path=output_path,
    ))


@cli.command(name="infer-video")
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="输入长视频路径.")
@click.option("--inference-config", "inference_config", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="滑窗推理配置 YAML (configs/inference/sliding_window.yaml).")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False),
              help="输出目录, 将产生 <basename>.timeline.json + <basename>.viz.mp4.")
@click.option("--no-viz", is_flag=True, default=False,
              help="只产 JSON, 不渲染 MP4.")
def infer_video_cmd(checkpoint: str, input_path: str, inference_config: str,
                    output_dir: str, no_viz: bool) -> None:
    """长视频端到端推理 + 滑窗 + 阈值合并 + JSON 时间轴 + (默认) MP4 可视化 (FR-014~016)."""
    from pingpong_av.cli import infer_video as _real
    sys.exit(_real.run(
        checkpoint=checkpoint,
        input_path=input_path,
        inference_config=inference_config,
        output_dir=output_dir,
        no_viz=no_viz,
    ))


# --------------------------------------------------------------------------------------
# 002 feature 子命令: 原始视频 → BMN 端到端
# --------------------------------------------------------------------------------------


@cli.command(name="extract-feat")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="输入视频文件 (mp4/avi/mov/flv/mkv 等任意 ffmpeg 可解格式).")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), default=None,
              help="输出 .pkl 路径; 默认在视频同目录, 文件名为 <sha256(file_bytes)[:32]>.pkl.")
@click.option("--fps", type=int, default=None,
              help="强制抽帧 fps; 不传则从 yaml 读 (默认 25).")
@click.option("--batch-size", "batch_size", type=int, default=None,
              help="PP-TSM forward batch_size; 不传则从 yaml 读 (默认 32).")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              default="configs/models/pp_tsm_extractor.yaml",
              help="抽特征业务配置 YAML.")
@click.option("--allow-dirty", is_flag=True, default=False,
              help="git 工作区脏时仍允许运行.")
@click.option("--keep-frames", is_flag=True, default=False,
              help="保留 ffmpeg 抽帧临时目录 (默认结束时清理).")
def extract_feat_cmd(input_path: str, output_path: str | None, fps: int | None,
                     batch_size: int | None, config_path: str,
                     allow_dirty: bool, keep_frames: bool) -> None:
    """原始视频 → 2048-d PP-TSM 特征 pkl (002 feature, FR-033)."""
    from pingpong_av.cli import extract_feat as _real
    sys.exit(_real.run(
        input_path=input_path,
        output_path=output_path,
        fps=fps,
        batch_size=batch_size,
        config_path=config_path,
        allow_dirty=allow_dirty,
        keep_frames=keep_frames,
    ))


@cli.command(name="build-feature-pkls")
@click.option("--videos-dir", "videos_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="含视频文件的目录 (递归扫描 mp4/avi/mov/flv/mkv).")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False),
              help="输出根目录, 命令内部会建 Features_<name>/ + manifest.csv + (可选) label_cls14_<name>.json.")
@click.option("--gt-json", "gt_json", type=click.Path(exists=True, dir_okay=False), default=None,
              help="可选 GT JSON (按 label_cls14_train.json schema); 若提供会校验 + 重写 url → clip_id.")
@click.option("--name", type=str, default=None,
              help="数据集子集名; 默认从 --videos-dir basename 派生.")
@click.option("--workers", type=int, default=1,
              help="并发 workers (当前实现仅 1; 预留接口).")
@click.option("--config", "config_path",
              type=click.Path(exists=True, dir_okay=False),
              default="configs/models/pp_tsm_extractor.yaml",
              help="PP-TSM 抽特征 YAML.")
@click.option("--allow-dirty", is_flag=True, default=False,
              help="git 工作区脏时仍允许运行.")
@click.option("--force", is_flag=True, default=False,
              help="忽略已有 .pkl, 全部重抽.")
def build_feature_pkls_cmd(videos_dir: str, output_dir: str, gt_json: str | None,
                           name: str | None, workers: int, config_path: str,
                           allow_dirty: bool, force: bool) -> None:
    """批量原始视频 → Features_<name>/<clip_id>.pkl + (可选) 重写 GT JSON (002 feature, FR-034)."""
    from pingpong_av.cli import build_feature_pkls as _real
    sys.exit(_real.run(
        videos_dir=videos_dir,
        output_dir=output_dir,
        gt_json=gt_json,
        name=name,
        workers=workers,
        config_path=config_path,
        allow_dirty=allow_dirty,
        force=force,
    ))


@cli.command(name="infer-rawvideo")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="输入原始视频文件 (任意 ffmpeg 可解格式).")
@click.option("--bmn-checkpoint", "bmn_checkpoint", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="BMN .pdparams 路径 (本仓库 v0.2.x 训练产物).")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False),
              help="输出根目录, 将产生 timeline.json + visualized.mp4 + feature.pkl.")
@click.option("--threshold", type=float, default=0.0,
              help="BMN proposal score 过滤阈值; 默认 0 (不过滤).")
@click.option("--min-duration", "min_duration", type=float, default=0.3,
              help="最小区间时长秒, 过滤太短候选; 默认 0.3.")
@click.option("--allow-dirty", is_flag=True, default=False,
              help="git 工作区脏时仍允许运行.")
@click.option("--keep-frames", is_flag=True, default=False,
              help="保留 ffmpeg 抽帧临时目录.")
@click.option("--keep-features", is_flag=True, default=True,
              help="保留中间产物 feature.pkl (默认 ON, 便于 debug / 断点续算).")
@click.option("--no-visualize", is_flag=True, default=False,
              help="跳过可视化 mp4 渲染 (调试用).")
@click.option("--extractor-config", "extractor_config",
              type=click.Path(exists=True, dir_okay=False),
              default="configs/models/pp_tsm_extractor.yaml",
              help="PP-TSM 抽特征 YAML.")
@click.option("--bmn-config", "bmn_config",
              type=click.Path(exists=True, dir_okay=False),
              default="configs/models/bmn_pingpong.yaml",
              help="BMN 业务 YAML.")
def infer_rawvideo_cmd(input_path: str, bmn_checkpoint: str, output_dir: str,
                       threshold: float, min_duration: float,
                       allow_dirty: bool, keep_frames: bool, keep_features: bool,
                       no_visualize: bool, extractor_config: str, bmn_config: str) -> None:
    """端到端原始视频推理: mp4 → timeline.json + 可视化 mp4 (002 feature, FR-039/040)."""
    from pingpong_av.cli import infer_rawvideo as _real
    sys.exit(_real.run(
        input_path=input_path,
        bmn_checkpoint=bmn_checkpoint,
        output_dir=output_dir,
        threshold=threshold,
        min_duration=min_duration,
        allow_dirty=allow_dirty,
        keep_frames=keep_frames,
        keep_features=keep_features,
        no_visualize=no_visualize,
        extractor_config=extractor_config,
        bmn_config=bmn_config,
    ))


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    """``pyproject.toml`` 中 ``pp = "pingpong_av.cli:main"`` 指向此函数."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
