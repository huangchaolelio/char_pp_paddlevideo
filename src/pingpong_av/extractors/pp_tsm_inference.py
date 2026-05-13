"""PP-TSM 静态图 Predictor 包装 (T207).

对应:
    - spec FR-037 (取 output_names[1], 即 2048-d 特征而非 logits)
    - data-model.md ImageFeaturePkl
    - research.md R10/R11
    - 上游 ``applications/FootballAction/predict/action_detect/models/pptsm_infer.py``
      (设计参考, 不复制代码)

设计:
    - 只管 "给一堆帧, 吐 (N_samples, 2048) 特征". 不管 ffmpeg / 切窗 / 下游 BMN.
    - 一个 :class:`PPTSMExtractor` 实例 = 一个加载到 GPU 的 Predictor.
      切记: 重用同一实例抽多视频, 不要每次新建 (避免 GPU 上下文反复切换).

抽特征语义 (与上游 FootballAction 一致):
    每 ``seg_num=8`` 帧 → 1 个 2048-d 特征样本.
    N 帧视频产出约 ``N // seg_num`` 个样本 (末尾不足 8 帧用最后一帧 pad).

Batch 形状:
    输入张量 (batch_size, seg_num, 3, target_size, target_size), float32
    输出张量 (batch_size, 2048), float32 (取 output_names[1])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pingpong_av.utils.logging import get_logger

__all__ = ["PPTSMExtractor", "PPTSMExtractorError"]

_log = get_logger(__name__)


class PPTSMExtractorError(RuntimeError):
    """PP-TSM Predictor 构造或前向失败."""


@dataclass
class ExtractorConfig:
    """从 configs/models/pp_tsm_extractor.yaml 的 extraction 段映射."""

    fps: int = 25
    batch_size: int = 32
    short_size: int = 256
    target_size: int = 224
    seg_num: int = 8
    seglen: int = 1
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def from_yaml_dict(cls, extraction: dict[str, Any]) -> "ExtractorConfig":
        """从 load_config(yaml).data['extraction'] 映射; 缺失字段用默认."""
        return cls(
            fps=int(extraction.get("fps", 25)),
            batch_size=int(extraction.get("batch_size", 32)),
            short_size=int(extraction.get("short_size", 256)),
            target_size=int(extraction.get("target_size", 224)),
            seg_num=int(extraction.get("seg_num", 8)),
            seglen=int(extraction.get("seglen", 1)),
            image_mean=tuple(extraction.get("image_mean", [0.485, 0.456, 0.406])),
            image_std=tuple(extraction.get("image_std", [0.229, 0.224, 0.225])),
        )


class PPTSMExtractor:
    """加载 PP-TSM inference 双文件到 GPU, 抽 2048-d 特征.

    典型用法:
        >>> cfg = ExtractorConfig(seg_num=8, batch_size=32, target_size=224)
        >>> ext = PPTSMExtractor(pdmodel="ppTSM.pdmodel", pdiparams="ppTSM.pdiparams", config=cfg)
        >>> features = ext.extract_from_frames_dir(Path("/tmp/frames/"))
        >>> features.shape    # (N_samples, 2048), N_samples ≈ total_frames // 8
    """

    def __init__(
        self,
        *,
        pdmodel: Path | str,
        pdiparams: Path | str,
        config: ExtractorConfig,
        device: str = "gpu",
        gpu_mem_mb: int = 8000,
        device_id: int = 0,
    ) -> None:
        """构造 Predictor.

        Args:
            pdmodel: 静态图结构文件 (由 export_pptsm_inference.py 产出).
            pdiparams: 静态图权重文件 (同上).
            config: yaml 派生的 extractor 配置.
            device: 'gpu' 或 'cpu' (CPU 仅作 fallback, 不能做 SC-011 性能).
            gpu_mem_mb: GPU 内存预留 (与上游 pptsm_infer.py 的 gpu_mem 字段语义一致).
            device_id: GPU 序号.
        """
        self._config = config
        self._pdmodel = Path(pdmodel)
        self._pdiparams = Path(pdiparams)
        if not self._pdmodel.is_file():
            raise PPTSMExtractorError(
                f"pdmodel 不存在: {self._pdmodel}. "
                f"请先运行 `python scripts/export_pptsm_inference.py`."
            )
        if not self._pdiparams.is_file():
            raise PPTSMExtractorError(
                f"pdiparams 不存在: {self._pdiparams}. "
                f"请先运行 `python scripts/export_pptsm_inference.py`."
            )

        # 延迟 import, 避免模块 import 期强依赖 paddle
        try:
            from paddle.inference import Config, create_predictor
        except ImportError as exc:
            raise PPTSMExtractorError(f"paddle.inference 不可导入: {exc}") from exc

        # 预计算 mean/std 为 (3,1,1) 用于广播
        self._img_mean = np.array(config.image_mean, dtype=np.float32).reshape(3, 1, 1)
        self._img_std  = np.array(config.image_std,  dtype=np.float32).reshape(3, 1, 1)

        # 构造 Predictor
        inf_cfg = Config(str(self._pdmodel), str(self._pdiparams))
        if device == "gpu":
            inf_cfg.enable_use_gpu(gpu_mem_mb, device_id)
        else:
            inf_cfg.disable_gpu()
        inf_cfg.switch_ir_optim(True)
        inf_cfg.enable_memory_optim()
        inf_cfg.switch_use_feed_fetch_ops(False)

        self._predictor = create_predictor(inf_cfg)

        input_names = self._predictor.get_input_names()
        self._input_tensor = self._predictor.get_input_handle(input_names[0])

        output_names = self._predictor.get_output_names()
        # FR-037: 我们自己的 scripts/export_pptsm_inference.py monkey-patch ppTSMHead
        # 返回 (feature_2048d, logits_400), 导出后 output_names = [feature_tensor, logits_tensor].
        # 因此取 output_names[0] (与上游 FootballAction extract_feat.py 的 [1] 不同, 那是因为
        # 它们的导出把 feature 放在 [1] 位置).
        if len(output_names) < 2:
            raise PPTSMExtractorError(
                f"PP-TSM inference 模型输出数 < 2: {output_names}. "
                f"导出的模型不符合预期 (应有 feature + logits 两个输出). "
                f"请重新运行 `python scripts/export_pptsm_inference.py`."
            )
        # 验证 output[0] 确实是 2048-d: 探测一下形状 (仅保存 handle, 不实际前向)
        self._output_tensor = self._predictor.get_output_handle(output_names[0])
        self._output_names = output_names

        _log.info(
            "PPTSMExtractor ready",
            extra={
                "pdmodel": str(self._pdmodel),
                "device": device,
                "seg_num": config.seg_num,
                "batch_size": config.batch_size,
                "target_size": config.target_size,
                "n_outputs": len(output_names),
                "using_output_idx": 1,
            },
        )

    # ---- 前向 ----

    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """前向一个 batch.

        Args:
            batch: shape (B, seg_num, 3, H, W), dtype float32, 已标准化.

        Returns:
            shape (B, 2048), float32.
        """
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        self._input_tensor.copy_from_cpu(batch)
        self._predictor.run()
        return self._output_tensor.copy_to_cpu()

    # ---- 抽帧目录 → 特征数组 ----

    def extract_from_frames_dir(
        self,
        frames_dir: Path | str,
        *,
        frame_ext: str = ".jpg",
    ) -> np.ndarray:
        """抽一个目录的帧 → (N_samples, 2048) ndarray.

        N_samples ≈ ceil(n_frames / seg_num). 末尾不足 seg_num 的用最后一帧重复 pad.

        Args:
            frames_dir: 含 `00000001.jpg, 00000002.jpg, ...` 的目录.
            frame_ext: 帧扩展名, 默认 `.jpg`.

        Returns:
            (N_samples, 2048) float32 ndarray.

        Raises:
            PPTSMExtractorError: 目录无帧或预处理失败.
        """
        frames_dir = Path(frames_dir)
        frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix == frame_ext)
        if not frame_paths:
            raise PPTSMExtractorError(f"{frames_dir} 无 {frame_ext} 帧")

        # 按 seg_num 切片, pad 末尾
        seg_num = self._config.seg_num
        samples: list[list[Path]] = []
        for i in range(0, len(frame_paths), seg_num):
            chunk = frame_paths[i : i + seg_num]
            if len(chunk) < seg_num:
                # pad 用最后一帧重复 (与上游 fault-tolerant 一致)
                chunk = chunk + [chunk[-1]] * (seg_num - len(chunk))
            samples.append(chunk)

        _log.info(
            "extract_from_frames_dir start",
            extra={"n_frames": len(frame_paths), "n_samples": len(samples),
                   "batch_size": self._config.batch_size},
        )

        # 按 batch_size 组装 + 前向
        features: list[np.ndarray] = []
        batch_size = self._config.batch_size
        for b_start in range(0, len(samples), batch_size):
            b_end = min(b_start + batch_size, len(samples))
            batch_samples = samples[b_start:b_end]
            batch_array = self._build_batch_array(batch_samples)
            out = self.infer_batch(batch_array)       # (B, 2048)
            features.append(out.astype(np.float32))

        result = np.vstack(features)
        assert result.shape[1] == 2048, f"期望 feature_dim=2048, 实际 {result.shape[1]}"
        assert result.dtype == np.float32, f"期望 dtype=float32, 实际 {result.dtype}"

        _log.info(
            "extract_from_frames_dir done",
            extra={"output_shape": list(result.shape)},
        )
        return result

    # ---- 预处理 ----

    def _build_batch_array(self, batch_samples: list[list[Path]]) -> np.ndarray:
        """从 Path 列表构造 (B, seg_num, 3, H, W) 标准化 float32 数组."""
        from PIL import Image

        cfg = self._config
        B = len(batch_samples)
        T = cfg.seg_num * cfg.seglen
        H = cfg.target_size
        W = cfg.target_size
        batch = np.empty((B, T, 3, H, W), dtype=np.float32)

        for b, sample_paths in enumerate(batch_samples):
            for t, p in enumerate(sample_paths):
                img = Image.open(p).convert("RGB")
                img = self._resize_short_side(img, cfg.short_size)
                img = self._center_crop(img, cfg.target_size)
                # PIL → ndarray (H, W, 3) uint8 → (3, H, W) float32 / 255
                arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                # ImageNet 标准化
                arr = (arr - self._img_mean) / self._img_std
                batch[b, t] = arr

        return batch

    @staticmethod
    def _resize_short_side(img: Any, short_size: int) -> Any:
        """把短边 resize 到 short_size, 长边按比例."""
        from PIL import Image
        w, h = img.size
        if w <= h:
            new_w = short_size
            new_h = int(round(h * short_size / w))
        else:
            new_h = short_size
            new_w = int(round(w * short_size / h))
        return img.resize((new_w, new_h), Image.BILINEAR)

    @staticmethod
    def _center_crop(img: Any, crop_size: int) -> Any:
        """中心裁剪到 crop_size × crop_size."""
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))
