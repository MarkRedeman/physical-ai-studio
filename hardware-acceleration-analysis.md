# Hardware acceleration for streaming video encoding — analysis

Status of Intel (QSV) and NVIDIA (NVENC) hardware acceleration for dataset
recordings across the native/development setup and the Docker images.

## TL;DR

- **NVIDIA (NVENC): works out of the box.** The PyPI `av` wheel already registers
  `h264_nvenc`, `hevc_nvenc`, and `av1_nvenc`. It only needs the NVIDIA driver + GPU
  reachable at runtime (native driver install, or the CUDA image run with `--gpus all`
  so the container toolkit injects the host driver libraries).
- **Intel (QSV): does not work out of the box, native or Docker.** The `av` wheel’s
  bundled FFmpeg was built without `--enable-libvpl`/`--enable-vaapi`, so the
  `*_qsv`/`*_vaapi` codecs are not registered at all — even though the XPU image
  ships the full oneVPL runtime. Fix: rebuild PyAV from source against an FFmpeg that
  has `--enable-libvpl`.

## Evidence

Empirically probed with the `av` wheel pinned by `application/backend/uv.lock`
(`av==15.1.0`) in the backend venv, on a host with an Intel iGPU (`/dev/dri`,
iHD driver) and no NVIDIA GPU:

| Codec                                     | Registered in wheel | Runtime probe result                                               |
| ----------------------------------------- | ------------------- | ------------------------------------------------------------------ |
| `h264_qsv` / `hevc_qsv` / `av1_qsv`       | **No**              | `UnknownCodecError`                                                |
| `h264_vaapi` / `hevc_vaapi` / `av1_vaapi` | **No**              | `UnknownCodecError`                                                |
| `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` | **Yes**             | `avcodec_open2` fails only because no NVIDIA driver/GPU is present |
| `libsvtav1` / `libx264` / `libx265` (CPU) | Yes                 | works                                                              |

The real encode-probe used at record time (`_is_vcodec_usable` in
`application/backend/src/internal_datasets/lerobot/streaming_encoding_settings.py`)
matches this: QSV/VA-API always skipped, NVENC skipped without a GPU, CPU encoders
selected as fallback.

## Root cause

- PyAV builds FFmpeg with a fixed set of flags (`PyAV-Org/PyAV` `scripts/build-deps`):
  it enables NVENC only when a CUDA toolchain is present at build time, and never
  enables `--enable-libvpl` or `--enable-vaapi`. The published Linux wheels therefore
  carry NVENC but no QSV/VA-API.
- PyAV v15 builds from source against the **system** FFmpeg via `pkg-config`
  (`setup.py` → `get_config_from_pkg_config`). It does not build its own FFmpeg.
  So `uv pip install --no-binary av av` alone only gains QSV if the system FFmpeg
  dev libraries were built with `--enable-libvpl`.

## What changed

1. **`application/docker/Dockerfile` — `builder-xpu`:** installs `libvpl-dev` + the
   `libav*-dev` packages and rebuilds `av==15.1.0` from source against Debian’s
   QSV-enabled FFmpeg. `runtime-xpu` already provides the matching runtime libraries
   (`libvpl2`, `libmfx-gen1`, `intel-media-va-driver-non-free`, and the `ffmpeg`
   metapackage that supplies `libavcodec61` etc.), and `docker-compose.yaml` already
   maps `/dev/dri`. No runtime-stage change needed.
2. **`application/backend/src/internal_datasets/lerobot/streaming_encoding_settings.py`:**
   when an explicitly selected codec is unusable, log a warning and fall back to the
   first usable candidate instead of failing the recording (kept `RuntimeError` only
   when nothing at all is usable). Tests updated in
   `application/backend/tests/internal_datasets/test_lerobot_dataset.py`.
3. **`application/backend/docs/video_hardware_acceleration_intel.md`:** rewritten with
   a check-first step, a fast distro-package path (for distros whose packaged FFmpeg
   has QSV, e.g. Debian 13), and a corrected from-source path explaining that PyAV
   links the system FFmpeg via `pkg-config`.

## Notes / follow-ups

- The auto-selection order prefers `av1_qsv` → `hevc_qsv` → `h264_qsv` → NVENC →
  VA-API → CPU. With the XPU image rebuild, Intel boxes get QSV first; NVIDIA boxes
  (CUDA image, wheel) get NVENC.
- The CUDA image keeps the stock wheel (NVENC already present); it is not rebuilt, so
  it also keeps the wheel’s bundled FFmpeg.
- Recording only, not decoding — the same PyAV FFmpeg is used for decoding, so the
  wheel is fine for reading datasets on CPU images.
