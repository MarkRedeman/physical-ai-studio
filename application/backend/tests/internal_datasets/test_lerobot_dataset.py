from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from lerobot.configs import RGBEncoderConfig

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot streaming_encoding_settings
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.streaming_encoding_settings import StreamingEncodingSettings, StudioRGBEncoderConfig


def test_streaming_settings_translate_to_lerobot_kwargs() -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=4,
        encoder_queue_maxsize=60,
    )

    kwargs = settings.to_lerobot_write_kwargs()

    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 4
    assert kwargs["encoder_queue_maxsize"] == 60
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert "vcodec" not in kwargs


def test_create_uses_rgb_encoder_and_not_vcodec(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=False),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.create", return_value=MagicMock()
        ) as create_mock,
    ):
        dataset.create(fps=30, features={}, robot_type="so100")

    kwargs = create_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_load_dataset_is_read_only_and_does_not_pass_write_kwargs(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset",
            return_value=MagicMock(num_episodes=1),
        ) as init_mock,
    ):
        dataset.load_dataset()

    kwargs = init_mock.call_args.kwargs
    assert "rgb_encoder" not in kwargs
    assert "streaming_encoding" not in kwargs
    assert "encoder_threads" not in kwargs
    assert "encoder_queue_maxsize" not in kwargs
    assert "vcodec" not in kwargs


def test_resume_dataset_uses_write_kwargs_and_not_vcodec(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.RECORDING_MUTATION

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.resume",
            return_value=MagicMock(num_episodes=1),
        ) as resume_mock,
    ):
        dataset.resume_dataset()

    kwargs = resume_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_resume_dataset_raises_in_read_only_mode(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with patch.object(InternalLeRobotDataset, "_resume_for_writing") as resume_mock:
        try:
            dataset.resume_dataset()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "RECORDING_MUTATION" in str(exc)
    resume_mock.assert_not_called()


def test_streaming_settings_translate_extra_encoding_fields() -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        pix_fmt="yuv420p",
        g=10,
        crf=23,
        preset="medium",
        extra_options={"x264-params": "keyint=100:min-keyint=100"},
    )

    kwargs = settings.to_lerobot_write_kwargs()

    rgb_encoder = kwargs["rgb_encoder"]
    assert rgb_encoder.pix_fmt == "yuv420p"
    assert rgb_encoder.g == 10
    assert rgb_encoder.crf == 23
    assert rgb_encoder.preset == "medium"
    assert rgb_encoder.extra_options == {"x264-params": "keyint=100:min-keyint=100"}


def test_streaming_settings_hw_codec_defaults_to_nv12_pix_fmt() -> None:
    with patch.object(streaming_encoding_settings, "_is_vcodec_usable", return_value=True):
        settings = StreamingEncodingSettings(streaming_encoding=True, vcodec="h264_nvenc")
        kwargs = settings.to_lerobot_write_kwargs()

    assert kwargs["rgb_encoder"].pix_fmt == "nv12"


def test_vcodec_candidates_are_av1_first_and_exclude_native() -> None:
    candidates = streaming_encoding_settings.vcodec_candidates()

    assert candidates[0] == "av1_qsv"
    assert candidates.index("av1_qsv") < candidates.index("h264_qsv")
    assert candidates.index("av1_nvenc") < candidates.index("h264_nvenc")
    assert candidates.index("av1_vaapi") < candidates.index("h264_vaapi")
    assert candidates.index("h264_videotoolbox") < candidates.index("libsvtav1")
    assert candidates.index("libsvtav1") < candidates.index("libx265")
    assert "h264" not in candidates
    assert "hevc" not in candidates


def test_auto_vcodec_resolves_first_viable_candidate() -> None:
    # av1_qsv passes the probe but fails validation on a machine where it is
    # not registered, so it must be skipped in favor of the first viable codec.
    with patch.object(
        streaming_encoding_settings,
        "_is_vcodec_usable",
        side_effect=lambda vcodec: vcodec in {"av1_qsv", "libsvtav1"},
    ):
        config = StudioRGBEncoderConfig(vcodec="auto")

    assert config.vcodec == "libsvtav1"


def test_explicit_unusable_vcodec_raises() -> None:
    with (
        patch.object(streaming_encoding_settings, "_is_vcodec_usable", return_value=False),
        pytest.raises(ValueError, match="not usable"),
    ):
        StudioRGBEncoderConfig(vcodec="av1_qsv")


def _encoder_config_with_fields(**fields) -> StudioRGBEncoderConfig:
    config = StudioRGBEncoderConfig.__new__(StudioRGBEncoderConfig)
    defaults = {
        "vcodec": "libsvtav1",
        "pix_fmt": "yuv420p",
        "g": 2,
        "crf": 30,
        "preset": None,
        "fast_decode": 0,
        "video_backend": "pyav",
        "extra_options": {},
    }
    defaults.update(fields)
    for name, value in defaults.items():
        setattr(config, name, value)
    return config


def test_qsv_codec_options_use_global_quality() -> None:
    config = _encoder_config_with_fields(vcodec="av1_qsv", crf=25, preset=1)

    opts = config.get_codec_options()

    assert opts["global_quality"] == 25
    assert "crf" not in opts


def test_vaapi_codec_options_use_qp() -> None:
    config = _encoder_config_with_fields(vcodec="av1_vaapi", crf=25)

    opts = config.get_codec_options()

    assert opts["qp"] == 25
    assert "crf" not in opts


def test_software_codec_options_keep_crf() -> None:
    config = _encoder_config_with_fields(vcodec="libx264", crf=23, preset="medium")

    opts = config.get_codec_options()

    assert opts["crf"] == 23
    assert "global_quality" not in opts
