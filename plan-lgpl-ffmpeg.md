# Plan: LGPL FFmpeg for the XPU Docker image

Status: proposed — not yet implemented.

## Why

The XPU image currently rebuilds PyAV against **Debian's FFmpeg**, which is
**GPL** (built with `--enable-gpl` + libx264/libx265). The repo is **Apache-2.0**;
Apache-2.0 is not compatible with GPLv2, and shipping a GPL FFmpeg is a recurring
legal flag. The PyPI `av` wheel (used by native installs and the CPU/CUDA images)
is even worse: it is **GPL + nonfree** (bundled FFmpeg built with
`--enable-nonfree`, `--enable-cuda-nvcc`, `--enable-libnpp`, plus NVENC).

Goal: make the XPU image ship an **LGPL v2.1**, **nonfree-free** FFmpeg so that
hardware acceleration still works (Intel QSV + VA-API) without the GPL/nonfree
baggage.

## What we build

An in-image FFmpeg built from source with only LGPL-safe features:

- `--enable-libvpl` (Intel oneVPL, MIT) → `*_qsv` encoders
- `--enable-vaapi --enable-libdrm` (VA-API, MIT) → `*_vaapi` encoders
- `--enable-libsvtav1 --enable-libaom` (both BSD-2) → CPU AV1 (fallback)
- **No** `--enable-gpl`, **no** libx264/libx265/libopenh264, **no** nonfree
  → the build license stays LGPL v2.1

Pinned to **FFmpeg 7.1.x** (same major the `av==15.1.0` wheel bundled — soname
`libavcodec61`), matching PyAV 15.1.0's expected ABI. PyAV v15 links the system
FFmpeg via `pkg-config`, so a source rebuild against this prefix picks it up.

## Dockerfile changes (`application/docker/Dockerfile`)

1. **New stage `ffmpeg-lgpl`** (based on `builder-base`):
   - `apt-get install --no-install-recommends nasm yasm libdrm-dev libva-dev
libvpl-dev libsvtav1-dev libaom-dev`
   - Download FFmpeg 7.1.x release tarball (pinned version + checksum, matching
     the repo's digest-pinning style), configure with the flags above (no
     `--enable-gpl`), `make -j`, `make install` into `/opt/ffmpeg-lgpl`.
2. **`builder-xpu`** — replace the current `libav*-dev` + `av==15.1.0` block
   (which links Debian's GPL FFmpeg):
   - `COPY --link --from=ffmpeg-lgpl /opt/ffmpeg-lgpl /opt/ffmpeg-lgpl`
   - `ENV PKG_CONFIG_PATH=/opt/ffmpeg-lgpl/lib/pkgconfig` and
     `LD_LIBRARY_PATH=/opt/ffmpeg-lgpl/lib`
   - `uv pip install --python .venv/bin/python --no-binary av av==15.1.0`
3. **`runtime-xpu`** (stage 6b):
   - `COPY --link --from=ffmpeg-lgpl /opt/ffmpeg-lgpl/lib /opt/ffmpeg-lgpl/lib`
   - `ENV LD_LIBRARY_PATH="/opt/ffmpeg-lgpl/lib"` so the rebuilt PyAV `.so`
     files find the LGPL libs at runtime
   - Keep the existing oneVPL / media-driver packages (`libvpl2`, `libmfx-gen1`,
     `intel-media-va-driver-non-free`, libva) — still required at runtime.

## Resulting behavior (XPU image)

- Registered encoders: `h264_qsv`/`hevc_qsv`/`av1_qsv`, `h264_vaapi`/`hevc_vaapi`/`av1_vaapi`, `libsvtav1`, `libaom-av1`.
- **No** NVENC, **no** libx264/libx265, **no** nonfree codecs.
- `auto` selection on an Intel box → QSV first; CPU fallback → SVT-AV1.
- **Product trade-off:** an LGPL FFmpeg has **no software H.264/HEVC encoder**.
  The "CPU H.264 (libx264)" / "CPU H.265 (libx265)" presets in the UI will not
  resolve in the XPU image; the explicit-codec fallback (already merged) will
  switch them to `libsvtav1`. Hardware paths (QSV/VA-API) are unaffected.

## Docs

- `application/backend/docs/video_hardware_acceleration_intel.md`: update the
  Docker section and add a licensing note — XPU image ships an LGPL, nonfree-free
  FFmpeg; the native "fast path" still uses the distro's GPL FFmpeg and is
  user-opt-in; include an H.264/HEVC/AV1 patent note.
- `hardware-acceleration-analysis.md`: reflect the LGPL decision.

## Open questions (pending confirmation)

1. Accept the loss of software H.264/H.265 presets in the XPU image (they fall
   back to SVT-AV1)?
2. Leave the unused system `ffmpeg` binary (GPL, pulled by the `ffmpeg`
   metapackage in `runtime-base`) in the image, or remove it for a fully
   GPL-free image?
3. Keep the FFmpeg version pin at 7.1.x?

## Notes

- Nothing in the application shells out to the system `ffmpeg` binary — all
  video goes through the PyAV module, so the swap is contained to the PyAV
  rebuild.
- This modifies the Dockerfile change already committed in `ac832fb3e`; it will
  land as a follow-up commit.
- Not legal advice; confirm the approach with the legal team.
