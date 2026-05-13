"""T206: PP-TSM 训练权重 → inference 双文件转换 (002 feature, FR-038a).

把 BCEBOS 公开下载的 `ppTSM_k400_dense.pdparams` (动态图 state_dict) 通过上游
`tools/export_model.py` 转换为 `ppTSM.pdmodel + ppTSM.pdiparams` (静态图 inference),
便于被上游 `extract_feat.py` 风格的 `paddle.inference.Predictor` 加载.

对应文档:
    - specs/002-raw-video-feature-bmn/research.md R10
    - specs/002-raw-video-feature-bmn/data-model.md PPTSMInferenceModel
    - specs/002-raw-video-feature-bmn/contracts/cli.md (scripts/export_pptsm_inference.py)

设计选择 (research.md R10):
    - **复用上游 `tools/export_model.py`** (subprocess 调), 不自己实现转换逻辑;
      上游已验证 PP-TSM + to_static + save 的全链路, 自己重写风险更高.
    - 本脚本是**薄 wrapper**: 幂等性检查 + marker.json 写入 + 输出整理.

幂等性:
    - 默认检查 `<out-dir>/.export_marker.json`; 若 `derived_from_train_weight_sha256` +
      `paddle_version` 都匹配, 直接返回 0 (skipped).
    - `--force` 跳过此检查.

退出码 (与 FR-047 对齐):
    0  成功 (含幂等跳过)
    1  用户输入错 (--src / --config 缺失)
    2  环境问题 (paddle / 上游不可导 / 训练权重 sha256 校验失败)
    3  章程硬约束违反 (保留, 本脚本内部不会触发)
    4  运行时失败 (export subprocess 返回非 0, 磁盘不足)

用法:
    python scripts/export_pptsm_inference.py                     # 用默认路径
    python scripts/export_pptsm_inference.py --force             # 强制重导
    python scripts/export_pptsm_inference.py --src <custom.pdparams> --out-dir <custom/>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 把 src/ 加到 path 以便 import pingpong_av.*
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pingpong_av.utils.config import load_config  # noqa: E402
from pingpong_av.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------------------


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """流式 sha256 整文件, 返回 64-hex 字符串."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _get_paddle_version() -> str:
    """懒加载 paddle, 返回版本号字符串."""
    try:
        import paddle
        return str(paddle.__version__)
    except ImportError as exc:
        print(f"ERROR: paddle 不可导入: {exc}", file=sys.stderr)
        sys.exit(2)


def _read_marker(marker_path: Path) -> dict | None:
    """读 .export_marker.json; 不存在或不合法 → None."""
    if not marker_path.is_file():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("marker 读失败, 视为不存在", extra={"path": str(marker_path), "error": str(exc)})
        return None


def _write_marker(
    marker_path: Path,
    *,
    train_weight_sha256: str,
    pdmodel_sha256: str,
    pdiparams_sha256: str,
    combined_sha256: str,
    paddle_version: str,
    upstream_yaml: str,
) -> None:
    """原子写 marker.json."""
    data = {
        "schema": "pp-tsm-export-marker-v1",
        "derived_from_train_weight_sha256": train_weight_sha256,
        "pdmodel_sha256": pdmodel_sha256,
        "pdiparams_sha256": pdiparams_sha256,
        "combined_sha256": combined_sha256,
        "paddle_version": paddle_version,
        "upstream_yaml": upstream_yaml,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = marker_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(marker_path)


def _combined_sha256(pdmodel: Path, pdiparams: Path) -> str:
    """两个文件拼接的 sha256, 作为 inference 双文件的联合身份."""
    h = hashlib.sha256()
    for p in [pdmodel, pdiparams]:
        with p.open("rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()


def _is_marker_fresh(
    marker: dict,
    *,
    current_train_sha256: str,
    current_paddle_version: str,
) -> bool:
    """判断现有 marker 是否仍对应当前 (训练权重 + paddle 版本) 组合."""
    return (
        marker.get("derived_from_train_weight_sha256") == current_train_sha256
        and marker.get("paddle_version") == current_paddle_version
    )


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def _find_upstream_pptsm_yaml(repo_root: Path) -> Path:
    """找上游 PP-TSM inference yaml (含 INFERENCE + model_name: ppTSM).

    优先使用 `configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml` — 它有完整
    INFERENCE 段, 且 MODEL 结构 (ResNetTweaksTSM depth=50) 与 dense.pdparams 100% 兼容.
    """
    candidates = [
        repo_root / "third_party" / "PaddleVideo" / "configs" / "recognition" / "pptsm" / "pptsm_k400_frames_uniform.yaml",
        repo_root / "third_party" / "PaddleVideo" / "configs" / "recognition" / "pptsm" / "pptsm_k400_videos_uniform.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "无法找到上游 PP-TSM inference yaml. 请确认 third_party/PaddleVideo submodule 已 init, "
        "预期位置: configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PP-TSM train-weight → inference-model converter")
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="源训练权重路径 (默认从 --config 的 pretrained.train_weight_path 读)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录 (默认从 --config 的 pretrained.inference_dir 读)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/pp_tsm_extractor.yaml",
        help="业务 yaml (默认 configs/models/pp_tsm_extractor.yaml)",
    )
    parser.add_argument(
        "--upstream-yaml",
        type=str,
        default=None,
        help="上游 PaddleVideo pptsm yaml (默认自动选 configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过 marker 检查, 强制重新导出",
    )
    args = parser.parse_args()

    # 1. 加载业务 yaml 拿路径默认值
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: --config 不存在: {cfg_path}", file=sys.stderr)
        return 1
    try:
        cfg = load_config(cfg_path).data
    except Exception as exc:
        print(f"ERROR: 加载业务 yaml 失败: {exc}", file=sys.stderr)
        return 1

    pretrained = cfg.get("pretrained") or {}
    src_path = Path(args.src or pretrained.get("train_weight_path", ""))
    out_dir = Path(args.out_dir or pretrained.get("inference_dir", ""))

    if not src_path or not str(src_path):
        print("ERROR: 训练权重路径未指定 (--src 或 yaml pretrained.train_weight_path)", file=sys.stderr)
        return 1
    # 把相对路径解析到 repo root (与 utils.config 的 find_repo_root 行为一致)
    if not src_path.is_absolute():
        src_path = _REPO_ROOT / src_path
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    if not src_path.is_file():
        url = pretrained.get("train_weight_url", "")
        print(
            f"ERROR: 训练权重不存在: {src_path}\n"
            f"  请下载 (~120MB):\n"
            f"    curl -fL -o {src_path} \\\n"
            f"      {url}",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 计算训练权重 sha256 (用于 marker + expected_sha256 校验)
    _log.info("sha256 scan train weight", extra={"path": str(src_path)})
    train_sha256 = _sha256_of_file(src_path)
    expected = pretrained.get("expected_train_sha256")
    if expected and expected != train_sha256:
        print(
            f"ERROR (章程 II 违反): 训练权重 sha256 不匹配:\n"
            f"  实际:  {train_sha256}\n"
            f"  预期:  {expected}\n"
            f"  请重新下载以保证实验可复现.",
            file=sys.stderr,
        )
        return 2

    # 3. 幂等性检查
    marker_path_str = pretrained.get("export_marker", str(out_dir / ".export_marker.json"))
    marker_path = Path(marker_path_str)
    if not marker_path.is_absolute():
        marker_path = _REPO_ROOT / marker_path

    pdmodel_name = pretrained.get("inference_pdmodel", "ppTSM.pdmodel")
    pdiparams_name = pretrained.get("inference_pdiparams", "ppTSM.pdiparams")
    pdmodel = out_dir / pdmodel_name
    pdiparams = out_dir / pdiparams_name

    paddle_version = _get_paddle_version()

    if not args.force:
        marker = _read_marker(marker_path)
        if marker and _is_marker_fresh(
            marker,
            current_train_sha256=train_sha256,
            current_paddle_version=paddle_version,
        ) and pdmodel.is_file() and pdiparams.is_file():
            out_summary = {
                "schema": "pp-tsm-export-v1",
                "src_train_weight": str(src_path),
                "src_train_weight_sha256": train_sha256,
                "out_pdmodel": str(pdmodel),
                "out_pdiparams": str(pdiparams),
                "combined_sha256": marker.get("combined_sha256", ""),
                "paddle_version": paddle_version,
                "exported_at": marker.get("exported_at"),
                "skipped_reason": "marker matches",
            }
            print(json.dumps(out_summary, ensure_ascii=False, sort_keys=True))
            print("✓ PP-TSM inference 模型已是最新, 跳过重导 (--force 可强制)", file=sys.stderr)
            return 0

    # 4. 自定义导出 (**不**用上游 tools/export_model.py, 因为它只保留 logits)
    #
    # 设计:
    #   上游 ppTSMHead.forward(x, num_seg) 最后 3 行是:
    #     x = paddle.reshape(x, [-1, self.in_channels])   # x 是 2048-d 特征
    #     score = self.fc(x)                              # 400 类 logits
    #     return score
    #   上游 tools/export_model.py 导出的 inference 模型只有 1 个 output (logits[N, 400]).
    #   我们要的是 2048-d 特征 (FR-037).
    #
    # 策略 (research.md R10):
    #   monkey-patch ppTSMHead.forward 让它返回元组 (feature, logits).
    #   然后 paddle.jit.to_static + save 会把两个 output 都记录到 .pdmodel.
    #   这样 PPTSMExtractor 取 output_names[1] 就是 logits (保留兼容)
    #   或 output_names[0] 就是 feature (我们要的).
    #
    # 与 SC-015 的关系: 不修改上游源码 (不写 patch 05), 只在本脚本运行时 monkey-patch.
    upstream_yaml = Path(args.upstream_yaml) if args.upstream_yaml else _find_upstream_pptsm_yaml(_REPO_ROOT)
    if not upstream_yaml.is_file():
        print(f"ERROR: 上游 yaml 不存在: {upstream_yaml}", file=sys.stderr)
        return 2

    # 切换到 repo root 再 import 上游 (避免 sys.path 污染)
    try:
        import paddle
        from paddle.jit import to_static
        from paddle.static import InputSpec

        # 先装上游 PaddleVideo 到 sys.path
        paddlevideo_root = _REPO_ROOT / "third_party" / "PaddleVideo"
        sys.path.insert(0, str(paddlevideo_root))

        from paddlevideo.modeling.builder import build_model
        from paddlevideo.modeling.heads.pptsm_head import ppTSMHead
        from paddlevideo.utils import get_config
    except ImportError as exc:
        print(f"ERROR: 无法导入上游 paddlevideo: {exc}", file=sys.stderr)
        return 2

    # 加载 cfg + trim + build model
    _log.info("building pp_tsm model", extra={"upstream_yaml": str(upstream_yaml)})
    cfg = get_config(str(upstream_yaml), show=False)
    if cfg.MODEL.get("backbone") and cfg.MODEL.backbone.get("pretrained"):
        cfg.MODEL.backbone.pretrained = ""

    model = build_model(cfg.MODEL)

    # ---- 关键: monkey-patch head.forward 返回 (feature, logits) ----
    original_forward = ppTSMHead.forward

    def _forward_with_feature(self, x, num_seg):
        """返回 (feature_2048d, logits_400) 元组."""
        x = self.avgpool2d(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = paddle.reshape(x, [-1, num_seg, x.shape[1]])
        x = paddle.mean(x, axis=1)
        x = paddle.reshape(x, shape=[-1, self.in_channels])
        # 此时 x 是 (N, 2048) — 就是我们要的特征
        feature = x
        score = self.fc(x)
        return feature, score

    ppTSMHead.forward = _forward_with_feature

    # 加载训练权重
    _log.info("loading train weight", extra={"path": str(src_path)})
    params = paddle.load(str(src_path))
    model.set_dict(params)
    model.eval()

    # to_static + save
    # PP-TSM 输入 spec 与上游 configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml 一致:
    # (B, num_seg, 3, target_size, target_size)
    inf = cfg.INFERENCE
    input_spec = [[
        InputSpec(
            shape=[None, int(inf.num_seg), 3, int(inf.target_size), int(inf.target_size)],
            dtype="float32",
        )
    ]]

    out_base = out_dir / pdmodel_name.replace(".pdmodel", "")
    _log.info("calling paddle.jit.save", extra={"out_base": str(out_base)})
    try:
        static_model = to_static(model, input_spec=input_spec)
        paddle.jit.save(static_model, str(out_base))
    except Exception as exc:
        # 恢复原 forward
        ppTSMHead.forward = original_forward
        print(
            f"ERROR: paddle.jit.save 失败: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4

    # 恢复原 forward (避免污染其它后续代码)
    ppTSMHead.forward = original_forward

    if not pdmodel.is_file() or not pdiparams.is_file():
        print(
            f"ERROR: paddle.jit.save 没写出预期文件:\n"
            f"  预期: {pdmodel} + {pdiparams}\n"
            f"  out_dir 内容: {list(out_dir.iterdir())}",
            file=sys.stderr,
        )
        return 4

    # 5. 校验导出结果并写 marker
    pdmodel_sha = _sha256_of_file(pdmodel)
    pdiparams_sha = _sha256_of_file(pdiparams)
    combined_sha = _combined_sha256(pdmodel, pdiparams)

    _write_marker(
        marker_path,
        train_weight_sha256=train_sha256,
        pdmodel_sha256=pdmodel_sha,
        pdiparams_sha256=pdiparams_sha,
        combined_sha256=combined_sha,
        paddle_version=paddle_version,
        upstream_yaml=str(upstream_yaml),
    )

    # 6. 结果 JSON 到 stdout
    out_summary = {
        "schema":                   "pp-tsm-export-v1",
        "src_train_weight":         str(src_path),
        "src_train_weight_sha256":  train_sha256,
        "out_pdmodel":              str(pdmodel),
        "out_pdiparams":            str(pdiparams),
        "combined_sha256":          combined_sha,
        "paddle_version":           paddle_version,
        "exported_at":              datetime.now(timezone.utc).isoformat(),
        "skipped_reason":           None,
    }
    print(json.dumps(out_summary, ensure_ascii=False, sort_keys=True))
    print(
        f"✓ PP-TSM inference 模型已导出:\n"
        f"  pdmodel:   {pdmodel}\n"
        f"  pdiparams: {pdiparams}\n"
        f"  combined:  {combined_sha[:16]}... (64-hex)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
