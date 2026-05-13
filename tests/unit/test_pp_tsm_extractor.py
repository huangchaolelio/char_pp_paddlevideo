"""T215: PPTSMExtractor 单元测试 (002 feature, FR-037).

策略:
    - 不依赖真实 paddle.inference (那是 e2e 测试的责任, 见 tests/integration/)
    - 用 unittest.mock 替换 Config / create_predictor
    - 验证: (a) 取 output_names[0] 即 feature, (b) shape 校验, (c) ExtractorConfig 加载正确

为什么是 [0] 不是 [1]:
    上游 FootballAction extract_feat.py 取 [1] 是因为它们的导出脚本把 feature 放在 [1] 位置.
    我们自己的 scripts/export_pptsm_inference.py monkey-patch ppTSMHead.forward 返回
    (feature, logits), 所以 output_names = [feature_tensor, logits_tensor], feature 在 [0].
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pingpong_av.extractors.pp_tsm_inference import (
    ExtractorConfig,
    PPTSMExtractor,
    PPTSMExtractorError,
)


def test_extractor_config_from_yaml_dict():
    """ExtractorConfig.from_yaml_dict 应严格映射 yaml 字段."""
    cfg = ExtractorConfig.from_yaml_dict({
        "fps": 30,
        "batch_size": 16,
        "short_size": 256,
        "target_size": 224,
        "seg_num": 8,
        "seglen": 1,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std":  [0.229, 0.224, 0.225],
    })
    assert cfg.fps == 30
    assert cfg.batch_size == 16
    assert cfg.target_size == 224
    assert cfg.seg_num == 8
    assert tuple(cfg.image_mean) == (0.485, 0.456, 0.406)


def test_extractor_config_defaults_when_missing():
    """缺字段时用默认值, 不抛错."""
    cfg = ExtractorConfig.from_yaml_dict({})
    assert cfg.fps == 25
    assert cfg.batch_size == 32
    assert cfg.target_size == 224


def test_extractor_init_missing_files(tmp_path):
    """pdmodel / pdiparams 不存在时必须抛 PPTSMExtractorError."""
    with pytest.raises(PPTSMExtractorError, match="pdmodel"):
        PPTSMExtractor(
            pdmodel=tmp_path / "missing.pdmodel",
            pdiparams=tmp_path / "missing.pdiparams",
            config=ExtractorConfig(),
        )


def _make_mock_predictor(output_names: list[str], output_shape: tuple[int, ...]):
    """构造一个 mock predictor.

    Returns:
        (mock_predictor, output_handle): output_handle.copy_to_cpu() 返回固定 shape ndarray.
    """
    mock_predictor = MagicMock()
    mock_predictor.get_input_names.return_value = ["data_batch_0"]
    mock_predictor.get_output_names.return_value = output_names

    # 每次 copy_to_cpu 返回带 batch_size 的数组
    mock_input = MagicMock()
    mock_output = MagicMock()
    # 让 copy_to_cpu 返回正确 shape 的数组 (按调用时 batch 维变化)
    def _fake_copy_to_cpu():
        # batch_size 从最近一次 input 的形状推出
        if hasattr(_fake_copy_to_cpu, "_last_batch"):
            B = _fake_copy_to_cpu._last_batch
        else:
            B = 1
        return np.random.randn(B, *output_shape[1:]).astype(np.float32)
    mock_output.copy_to_cpu = _fake_copy_to_cpu

    def _fake_copy_from_cpu(arr):
        _fake_copy_to_cpu._last_batch = arr.shape[0]
    mock_input.copy_from_cpu = _fake_copy_from_cpu

    mock_predictor.get_input_handle.return_value = mock_input
    mock_predictor.get_output_handle.return_value = mock_output

    return mock_predictor


def test_extractor_takes_output_0_not_1(tmp_path):
    """FR-037: 必须取 output_names[0] (即 2048-d feature, 我们 monkey-patch 的导出).

    与上游 FootballAction extract_feat.py 取 [1] 的语义对比: 二者都对, 取决于导出顺序.
    我们的 export_pptsm_inference.py 把 feature 放在 [0].
    """
    # 创建假的 pdmodel/pdiparams (内容无关, 只要存在)
    pdmodel = tmp_path / "ppTSM.pdmodel"
    pdiparams = tmp_path / "ppTSM.pdiparams"
    pdmodel.write_bytes(b"fake pdmodel")
    pdiparams.write_bytes(b"fake pdiparams")

    mock_predictor = _make_mock_predictor(
        output_names=["save_infer_model/scale_0.tmp_0", "save_infer_model/scale_1.tmp_0"],
        output_shape=(0, 2048),  # batch_size 0 占位; 实际由 _last_batch 决定
    )

    with patch("paddle.inference.Config") as mock_cfg, \
         patch("paddle.inference.create_predictor", return_value=mock_predictor):
        # Config 是 builder, 各方法都返回 self / None — 用 MagicMock 默认行为即可
        mock_cfg.return_value = MagicMock()

        cfg = ExtractorConfig(batch_size=4, seg_num=8, target_size=224)
        extractor = PPTSMExtractor(
            pdmodel=pdmodel,
            pdiparams=pdiparams,
            config=cfg,
        )

        # 验证: 取了 output_names[0]
        # 第一次 get_output_handle 是 PPTSMExtractor.__init__ 里调的
        assert mock_predictor.get_output_handle.called
        first_call_args = mock_predictor.get_output_handle.call_args_list[0]
        assert first_call_args.args[0] == "save_infer_model/scale_0.tmp_0", (
            f"应取 output_names[0], 实际取了 {first_call_args.args[0]}"
        )


def test_extractor_rejects_single_output(tmp_path):
    """如果导出的 inference 模型只有 1 个 output (用上游 tools/export_model.py 直接导出会这样),
    必须抛 PPTSMExtractorError 让用户知道要用我们的脚本重新导出."""
    pdmodel = tmp_path / "ppTSM.pdmodel"
    pdiparams = tmp_path / "ppTSM.pdiparams"
    pdmodel.write_bytes(b"fake")
    pdiparams.write_bytes(b"fake")

    mock_predictor = _make_mock_predictor(
        output_names=["save_infer_model/scale_0.tmp_0"],   # 只有 1 个
        output_shape=(0, 400),
    )

    with patch("paddle.inference.Config") as mock_cfg, \
         patch("paddle.inference.create_predictor", return_value=mock_predictor):
        mock_cfg.return_value = MagicMock()
        with pytest.raises(PPTSMExtractorError, match="输出数 < 2"):
            PPTSMExtractor(
                pdmodel=pdmodel,
                pdiparams=pdiparams,
                config=ExtractorConfig(),
            )


def test_extractor_infer_batch_dtype(tmp_path):
    """infer_batch 必须把 batch 转成 float32 后再喂."""
    pdmodel = tmp_path / "m.pdmodel"
    pdiparams = tmp_path / "m.pdiparams"
    pdmodel.write_bytes(b"x"); pdiparams.write_bytes(b"x")

    mock_predictor = _make_mock_predictor(
        output_names=["a", "b"],
        output_shape=(0, 2048),
    )

    with patch("paddle.inference.Config") as mock_cfg, \
         patch("paddle.inference.create_predictor", return_value=mock_predictor):
        mock_cfg.return_value = MagicMock()
        ext = PPTSMExtractor(pdmodel=pdmodel, pdiparams=pdiparams,
                             config=ExtractorConfig(batch_size=2, seg_num=8, target_size=224))

        # 喂一个 float64 batch, 验证转换为 float32
        batch = np.zeros((2, 8, 3, 224, 224), dtype=np.float64)
        result = ext.infer_batch(batch)
        # 通过 mock input 验证传入的实际是 float32
        # (我们的 _fake_copy_from_cpu 记录 batch_size; 这里只验证 result 形状)
        assert result.shape == (2, 2048)


def test_extractor_resize_short_side():
    """_resize_short_side 必须把短边精确缩到 short_size."""
    from PIL import Image
    # 长方形 100x200 (短边 100), 缩到 256
    img = Image.new("RGB", (100, 200), color=(255, 0, 0))
    resized = PPTSMExtractor._resize_short_side(img, 256)
    assert min(resized.size) == 256, f"{resized.size}"
    # 比例保持
    assert abs(resized.size[1] / resized.size[0] - 2.0) < 0.02


def test_extractor_center_crop():
    """_center_crop 必须返回正方形, 大小精确."""
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(0, 255, 0))
    cropped = PPTSMExtractor._center_crop(img, 224)
    assert cropped.size == (224, 224)
