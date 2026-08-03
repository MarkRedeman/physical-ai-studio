# Enable hardware acceleration for video encoding on Intel devices

Intel supports hardware acceleration for video encoding on Intel devices using **oneVPL**.
pyAV is used as video backend. pyAV uses libav / ffmpeg.

For **QSV / oneVPL**, the right path is:

1. build/install **oneVPL**
2. build **FFmpeg** with `--enable-libvpl`
3. build **PyAV from source** against that FFmpeg

Intel’s FFmpeg guidance says to enable VPL is to simply compile with `--enable-libvpl`, and that this enables the `*_qsv` codecs such as `h264_qsv` and `hevc_qsv`.
It also notes that FFmpeg’s `*_qsv` codecs are implemented on top of VPL.

One important detail: Intel’s `libvpl` repo is only the **dispatcher + headers + samples**. You also need an **implementation** installed, such as `oneVPL-intel-gpu` for newer Intel Xe and newer hardware, or Media SDK for legacy graphics.

### 1) Install build dependencies

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake meson ninja-build pkg-config \
  python3-dev python3-venv python3-pip \
  yasm nasm \
  libdrm-dev libva-dev vainfo
```

### 2) Build and install oneVPL

```bash
git clone https://github.com/intel/libvpl
cd libvpl

export VPL_INSTALL_DIR="$HOME/opt/vpl"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$VPL_INSTALL_DIR"
cmake --build build -j"$(nproc)"
cmake --install build
```

Then export the pkg-config path so FFmpeg can find `vpl.pc`:

```bash
export PKG_CONFIG_PATH="$VPL_INSTALL_DIR/lib/pkgconfig:$VPL_INSTALL_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$VPL_INSTALL_DIR/lib:$VPL_INSTALL_DIR/lib64:$LD_LIBRARY_PATH"
```

That environment setup is necessary when `libvpl` is not installed to a standard location.

### 3) Make sure the Intel GPU runtime is installed

This is separate from the dispatcher. Without the runtime/implementation, FFmpeg may build but QSV will fail at runtime. [Intel’s install docs](https://github.com/intel/libvpl/blob/main/INSTALL.md) say the base package alone is not enough and you need an implementation as well.

### 4) Install dependencies for SW encoders
If the hardware accelerator is not available, you might want to fall back on SW encoders.
You can skip this step if you do not want to enable these encoders.

For libopenh264:
```bash
git clone https://github.com/cisco/openh264.git
make -j"$(nproc)"
sudo make install
```

For SVT-AV1:
```bash
git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
cd SVT-AV1
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
sudo cmake --install build

export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

### 5) Build FFmpeg with libvpl enabled

```bash
cd ~
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

./configure \
  --prefix="$HOME/opt/ffmpeg-vpl" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/opt/vpl/include" \
  --extra-ldflags="-L$HOME/opt/vpl/lib -L$HOME/opt/vpl/lib64" \
  --extra-libs="-lpthread -lm" \
  --enable-libvpl \
  --enable-vaapi \
  --enable-shared \
  --enable-libsvtav1 \
  --enable-libopenh264

make -j"$(nproc)"
make install
```

You can remove these lines if you did skip step 4:
```bash
--enable-libsvtav1 \
--enable-libopenh264
```

After install:

```bash
export PATH="$HOME/opt/ffmpeg-vpl/bin:$PATH"
export PKG_CONFIG_PATH="$HOME/opt/ffmpeg-vpl/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$HOME/opt/ffmpeg-vpl/lib:$LD_LIBRARY_PATH"
```

### 6) Verify FFmpeg sees QSV

```bash
ffmpeg -encoders | grep qsv
ffmpeg -h encoder=h264_qsv
```

You should see `h264_qsv`. Intel documents that `h264_qsv`, `hevc_qsv`, and other `*_qsv` codecs are the FFmpeg-facing names when using VPL-backed QSV.

### 7) Install PyAV from source

Inside your uv environment:

```bash
uv pip uninstall av
uv pip install --no-binary av av
```

## Configuring the streaming video encoder

The backend selects the encoder used when recording episodes through
`StreamingEncodingSettings`. The encoder is auto-detected at process start: `vcodec` defaults to
`auto`, which probes the candidate encoders and picks the first one that can **actually encode a
test frame**. The probe mirrors lerobot's own streaming-encode path (open a container, add a stream,
encode and mux one frame), so hardware device initialization is exercised end to end. A codec that
is registered in the pyAV build but has no usable device (for example `h264_nvenc` in a container
without GPU passthrough) is skipped instead of failing mid-recording.

Encoder preference order for `auto` (hardware before software, AV1 first within each family):

1. `av1_qsv`, `hevc_qsv`, `h264_qsv` — Intel QSV (AV1 preferred on Panther Lake, H.264 for maximum compatibility)
2. `av1_nvenc`, `hevc_nvenc`, `h264_nvenc` — NVIDIA
3. `av1_vaapi`, `hevc_vaapi`, `h264_vaapi` — VA-API (Linux Intel/AMD)
4. `h264_videotoolbox`, `hevc_videotoolbox` — macOS hardware
5. `libsvtav1`, `libaom-av1` — software AV1 (open source, LGPL)
6. `libx265`, `libx264` — software, offline compression (GPL)

Native `h264`/`hevc` encoders are excluded from auto-selection because they are testing-focused and
generally inferior to the alternatives. They can still be selected explicitly with `STREAMING_VCODEC`
if required. The codec surface is not limited by lerobot's `VALID_VIDEO_CODECS` whitelist, so
`av1_qsv` and `libx264`/`libx265` work as long as the local FFmpeg build provides them.

### Environment variables

All settings can be overridden through environment variables on the backend service:

| Variable | Default | Description |
| --- | --- | --- |
| `STREAMING_VCODEC` | `auto` | Encoder name. Valid values: `av1_qsv`, `hevc_qsv`, `h264_qsv`, `av1_nvenc`, `hevc_nvenc`, `h264_nvenc`, `av1_vaapi`, `hevc_vaapi`, `h264_vaapi`, `libsvtav1`, `libaom-av1`, `libx265`, `libx264`, `h264_videotoolbox`, `hevc_videotoolbox`, `h264`, `hevc`, `auto`. |
| `STREAMING_PIX_FMT` | unset | Pixel format passed to the encoder. For hardware encoders this defaults to `nv12`; otherwise `yuv420p`. |
| `STREAMING_CRF` | unset | Constant rate factor (quality) for codecs that support it (e.g. `libsvtav1` defaults to 30). |
| `STREAMING_PRESET` | unset | Codec preset (e.g. `12` for `libsvtav1`, `medium` for `h264`). |
| `STREAMING_EXTRA_OPTIONS` | unset | JSON object of extra codec options merged into the encoder config. |
| `STREAMING_ENCODER_THREADS` | unset | Number of encoder worker threads. |
| `STREAMING_ENCODER_QUEUE_MAXSIZE` | `60` | Max frames queued for encoding before frames are dropped. |

Example, force SVT-AV1 with a quality preset:

```bash
STREAMING_VCODEC=libsvtav1 STREAMING_PRESET=8 STREAMING_CRF=25 docker compose up -d
```

Or select the Intel QSV encoder explicitly:

```bash
STREAMING_VCODEC=h264_qsv docker compose up -d
```

### Known limitations

- `av1_qsv` requires an Intel GPU with AV1 encode support (Xe and later) and a oneVPL-based FFmpeg
  build. It is auto-detected and skipped when not usable.
- On macOS, `h264_videotoolbox`/`hevc_videotoolbox` are the first usable hardware encoders, so they
  are preferred over the software fallbacks automatically. Set `STREAMING_VCODEC=hevc_videotoolbox`
  to force HEVC instead of H.264.
- `libx264`/`libx265` are GPL-licensed. For an open-source distribution that bundles FFmpeg binaries,
  prefer AV1-only codecs (`av1_qsv`, `libsvtav1`, `libaom-av1`) and detect H.264/HEVC support at
  runtime rather than redistributing it. Physical AI Studio only uses the codecs the local FFmpeg
  build provides and does not bundle FFmpeg itself.
