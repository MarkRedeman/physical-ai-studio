import logging
import tempfile
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

from lerobot.configs import RGBEncoderConfig
from lerobot.configs.video import VIDEO_CODECS_ALIASES
from pydantic import BaseModel, Field

# Codec-specific pixel formats. Hardware encoders generally require NV12 input
# (or auto-convert via FFmpeg); defaulting here avoids RGBEncoderConfig's pix_fmt
# validation rejecting a codec that only advertises yuv420p support.
_HW_CODEC_PIX_FMTS: dict[str, str] = {
    "av1_qsv": "nv12",
    "hevc_qsv": "nv12",
    "h264_qsv": "nv12",
    "av1_nvenc": "nv12",
    "hevc_nvenc": "nv12",
    "h264_nvenc": "nv12",
    "av1_vaapi": "nv12",
    "hevc_vaapi": "nv12",
    "h264_vaapi": "nv12",
    "h264_videotoolbox": "nv12",
    "hevc_videotoolbox": "nv12",
}

# Auto-selection preference order. Hardware acceleration is preferred over software.
# Intel QSV (AV1 first), then NVIDIA, then VA-API, then macOS videotoolbox (hardware,
# unavailable on other platforms), then software encoders. Native `h264`/`hevc`
# (testing-only encoders) and VP9 are intentionally excluded.
_VCODEC_CANDIDATES: tuple[str, ...] = (
    "av1_qsv",  # Intel QSV AV1 (preferred on Panther Lake)
    "hevc_qsv",  # Intel QSV HEVC
    "h264_qsv",  # Intel QSV H.264, max compatibility
    "av1_nvenc",  # NVIDIA AV1 (Ada and later)
    "hevc_nvenc",  # NVIDIA HEVC
    "h264_nvenc",  # NVIDIA H.264
    "av1_vaapi",  # VA-API AV1 (Linux Intel/AMD)
    "hevc_vaapi",  # VA-API HEVC
    "h264_vaapi",  # VA-API H.264
    "h264_videotoolbox",  # macOS hardware
    "hevc_videotoolbox",  # macOS hardware
    "libsvtav1",  # open-source software AV1 (LGPL)
    "libaom-av1",  # reference software AV1
    "libx265",  # software HEVC (GPL), offline compression
    "libx264",  # software H.264 (GPL), offline compression
)


def vcodec_candidates() -> list[str]:
    """Return the auto-selection candidates in preference order."""
    return list(_VCODEC_CANDIDATES)


@cache
def _is_vcodec_usable(vcodec: str) -> bool:
    """Probe whether *vcodec* can actually encode a frame.

    Mirrors lerobot's StreamingVideoEncoder path (av.open + add_stream) so that
    hardware device auto-initialization is exercised. The result is cached.
    """
    import av
    import numpy as np

    pix_fmt = _HW_CODEC_PIX_FMTS.get(vcodec, "yuv420p")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as probe_file:
        probe_path = Path(probe_file.name)
    try:
        container = av.open(str(probe_path), "w")
        stream = container.add_stream(vcodec, 30, options={"g": "2"})
        stream.pix_fmt = pix_fmt  # type: ignore[missing-attribute]
        stream.width = 320  # type: ignore[missing-attribute]
        stream.height = 240  # type: ignore[missing-attribute]
        stream.time_base = Fraction(1, 30)
        frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="rgb24")
        frame.pts = 0
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):  # type: ignore[missing-attribute]
            container.mux(packet)
        for packet in stream.encode():  # type: ignore[missing-attribute]
            container.mux(packet)
        container.close()
        return True
    except Exception as exc:
        logging.warning(f"Skipping unavailable vcodec '{vcodec}': {exc}")
        return False
    finally:
        probe_path.unlink(missing_ok=True)


class StudioRGBEncoderConfig(RGBEncoderConfig):
    """RGB encoder config with backend-owned auto codec resolution.

    Subclasses lerobot's RGBEncoderConfig so it stays fully compatible with the
    DatasetWriter API (asdict persistence, get_codec_options, isinstance checks)
    while letting us resolve ``vcodec="auto"`` with a real-encode probe instead
    of lerobot's registration-only check.
    """

    def resolve_vcodec(self) -> None:
        self.vcodec = VIDEO_CODECS_ALIASES.get(self.vcodec, self.vcodec)  # type: ignore[has-type]
        if self.vcodec != "auto":
            if not _is_vcodec_usable(self.vcodec):
                raise ValueError(f"Video codec {self.vcodec!r} is not usable for encoding (probe encode failed)")
            self._apply_default_pix_fmt()
            return

        for candidate in vcodec_candidates():
            if _candidate_is_viable(candidate):
                self.vcodec = candidate
                self._apply_default_pix_fmt()
                logging.info(f"Auto-selected vcodec '{candidate}'")
                return

        raise RuntimeError("No usable video encoder found for streaming encoding")

    def _apply_default_pix_fmt(self) -> None:
        # The base class defaults pix_fmt to "yuv420p"; treat that as "unset"
        # and pick the codec-appropriate format for hardware encoders.
        if self.pix_fmt == "yuv420p":  # type: ignore[has-type]
            self.pix_fmt = _HW_CODEC_PIX_FMTS.get(self.vcodec, self.pix_fmt)  # type: ignore[has-type]

    def get_codec_options(self, encoder_threads: int | None = None, as_strings: bool = False) -> dict[str, Any]:
        opts = super().get_codec_options(encoder_threads, as_strings)
        # lerobot's base mapping only knows h264_qsv/h264_vaapi. Map the generic
        # `crf` it emits for the other qsv/vaapi codecs to the option those
        # encoders actually expose.
        if self.vcodec.endswith("_qsv") and "crf" in opts:
            opts["global_quality"] = opts.pop("crf")
        elif self.vcodec.endswith("_vaapi") and "crf" in opts:
            opts["qp"] = opts.pop("crf")
        return opts


def _candidate_is_viable(vcodec: str) -> bool:
    """Whether *vcodec* works end to end (probe encode + config validation)."""
    if not _is_vcodec_usable(vcodec):
        return False
    try:
        StudioRGBEncoderConfig(vcodec=vcodec)
        return True
    except Exception as exc:
        logging.warning(f"Skipping vcodec '{vcodec}' (probe passed but validation failed): {exc}")
        return False


class StreamingEncodingSettings(BaseModel):
    streaming_encoding: bool = True
    vcodec: str = "auto"
    pix_fmt: str | None = None
    g: int | None = 2
    crf: int | float | None = None
    preset: int | str | None = None
    extra_options: dict[str, Any] = Field(default_factory=dict)
    encoder_threads: int | None = None
    encoder_queue_maxsize: int = 60

    def to_lerobot_write_kwargs(self) -> dict[str, Any]:
        return {
            "streaming_encoding": self.streaming_encoding,
            "encoder_threads": self.encoder_threads,
            "encoder_queue_maxsize": self.encoder_queue_maxsize,
            "rgb_encoder": self._build_rgb_encoder_config(),
        }

    def _build_rgb_encoder_config(self) -> StudioRGBEncoderConfig:
        params: dict[str, Any] = {"vcodec": self.vcodec}
        if self.pix_fmt is not None:
            params["pix_fmt"] = self.pix_fmt
        if self.g is not None:
            params["g"] = self.g
        if self.crf is not None:
            params["crf"] = self.crf
        if self.preset is not None:
            params["preset"] = self.preset
        if self.extra_options:
            params["extra_options"] = self.extra_options
        return StudioRGBEncoderConfig(**params)
