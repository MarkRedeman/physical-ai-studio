# Physical AI Studio Distribution

This document describes how the backend is packaged as the `physicalai-studio` Python distribution, how to test the wheel locally before publishing, and how the GitHub Actions publishing workflows behave.

## Production Usage Overview

The initial PyPI rollout publishes one distribution:

```text
physicalai-studio
```

Users select the hardware runtime with package extras:

```text
physicalai-studio[cpu]
physicalai-studio[xpu]
physicalai-studio[cuda]
```

Because PyTorch publishes hardware-specific wheels on separate package indexes, users must pass the matching PyTorch index to `uvx`.

### Intel XPU

Use this on Intel GPU systems:

```bash
uvx \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]" \
  physicalai-studio serve
```

### CPU

Use this on systems without a supported GPU, or for simple smoke testing:

```bash
uvx \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cpu]" \
  physicalai-studio serve
```

### NVIDIA CUDA

Use this on NVIDIA GPU systems compatible with the CUDA 12.8 PyTorch wheels:

```bash
uvx \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cuda]" \
  physicalai-studio serve
```

All three commands start the same packaged application and serve the bundled production UI. The selected extra only changes the installed training/inference runtime dependencies.

For headless startup, add:

```bash
--no-browser
```

For example:

```bash
uvx \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]" \
  physicalai-studio serve --no-browser
```

## Package Shape

The Python package is built from `application/backend`.

Important files:

```text
application/backend/
├── pyproject.toml
├── scripts/build_package.sh
├── scripts/hatch_build.py
├── src/
│   ├── cli/
│   ├── main.py
│   ├── alembic.ini
│   ├── alembic/
│   └── ...
└── dist/
```

The wheel contains:

- Backend Python modules from `application/backend/src`.
- Alembic configuration and migrations.
- A `physicalai-studio` console script.
- The production UI build from `application/ui/dist`, packaged into the wheel as `webui/`.

The UI is included through Hatch `force-include`:

```toml
[tool.hatch.build.force-include]
"../ui/dist" = "webui"
```

`scripts/hatch_build.py` prevents publishing a wheel without the frontend by failing wheel builds when `application/ui/dist/index.html` is missing. Editable development installs are not blocked.

## Build The Wheel Locally

From the repository root:

```bash
bash application/backend/scripts/build_package.sh
```

The script performs the release build sequence:

1. Syncs backend dependencies with the XPU extra.
2. Generates the OpenAPI spec from the backend.
3. Installs UI dependencies.
4. Builds OpenAPI TypeScript definitions.
5. Builds the production UI.
6. Builds the backend wheel.
7. Runs `twine check` on the generated distribution.

The generated wheel is written to:

```text
application/backend/dist/
```

## Inspect Wheel Contents

Confirm that the wheel contains UI assets and migrations:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("application/backend/dist").glob("*.whl"))
with ZipFile(wheel) as zf:
    names = set(zf.namelist())

for name in (
    "webui/index.html",
    "alembic.ini",
    "alembic/env.py",
):
    print(f"{name}: {name in names}")

print("webui file count:", sum(name.startswith("webui/") for name in names))
PY
```

Expected:

```text
webui/index.html: True
alembic.ini: True
alembic/env.py: True
webui file count: <non-zero>
```

## Test The Wheel Locally With uvx

Use the wheel directly, without publishing to PyPI:

```bash
WHEEL="/home/mark/projects/intel/physical-ai-studio/application/backend/dist/physicalai_studio-0.1.0-py3-none-any.whl"

uvx --isolated --no-cache \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu] @ file://${WHEEL}" \
  physicalai-studio serve
```

Then open:

```text
http://127.0.0.1:7860
```

Use `--no-cache` while iterating locally. The version is usually unchanged during local testing, so `uv` may otherwise reuse a cached wheel.

Use `--isolated` to avoid reusing an installed tool environment.

The PyTorch XPU index is passed explicitly because wheel metadata cannot carry the local `[tool.uv.index]` configuration from `pyproject.toml`.

## Test From TestPyPI

When validating a release candidate from TestPyPI, include all three indexes:

- TestPyPI for `physicalai-studio`
- PyPI for normal dependencies (`torchao`, `fastapi`, etc.)
- PyTorch hardware index for `torch`/`torchvision` wheels

```bash
uvx \
  --index https://test.pypi.org/simple/ \
  --index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]==0.1.0" \
  physicalai-studio serve
```

For local iteration with the same version number, add `--no-cache --isolated`.

## Test The Wheel With uv pip

For a clean virtual environment test:

```bash
tmpdir="$(mktemp -d)"
uv venv "$tmpdir/venv"

uv pip install \
  --python "$tmpdir/venv/bin/python" \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  "physicalai-studio[xpu] @ file:///home/mark/projects/intel/physical-ai-studio/application/backend/dist/physicalai_studio-0.1.0-py3-none-any.whl"

uv pip check --python "$tmpdir/venv/bin/python"
"$tmpdir/venv/bin/physicalai-studio" serve --help
```

## Runtime Behavior

The console script is:

```text
physicalai-studio = cli:cli
```

`physicalai-studio serve` does the package-specific setup before importing the FastAPI app:

- Points `STATIC_FILES_DIR` at the installed `webui/` directory when it exists.
- Points Alembic at the installed `alembic.ini` and `alembic/` directory.
- Uses the platform default storage directory unless `STORAGE_DIR` is set.
- Runs storage migration and database migrations.
- Starts Uvicorn on `127.0.0.1:7860` by default.

Supported options:

```bash
physicalai-studio serve --host 127.0.0.1 --port 7860 --no-browser
```

## CPU, XPU, And CUDA Extras

The backend currently defines three hardware extras:

```toml
[project.optional-dependencies]
cpu = ["torch<2.13", "torchvision<0.26.0"]
cuda = ["torch<2.13", "torchvision<0.26.0"]
xpu = [
  "torch<2.13",
  "torchvision<0.26.0",
  "pytorch-triton-xpu ; sys_platform == 'linux' or sys_platform == 'win32'",
]
```

### XPU

XPU is supported in the initial PyPI rollout. Use the XPU PyTorch index for Intel GPU systems.

Published package:

```bash
uvx \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu]" \
  physicalai-studio serve
```

Local wheel:

```bash
uvx --isolated --no-cache \
  --index https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[xpu] @ file://${WHEEL}" \
  physicalai-studio serve
```

XPU requires the PyTorch XPU wheel index because `pytorch-triton-xpu` is not available on PyPI.

### CPU

CPU is supported in the initial PyPI rollout. It is useful for machines without a supported GPU and for smoke testing.

Published package:

```bash
uvx \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cpu]" \
  physicalai-studio serve --no-browser
```

Local wheel:

```bash
uvx --isolated --no-cache \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cpu] @ file://${WHEEL}" \
  physicalai-studio serve --no-browser
```

### CUDA

CUDA is supported in the initial PyPI rollout. It should be used on NVIDIA GPU systems compatible with the CUDA 12.8 PyTorch wheels.

Published package:

```bash
uvx \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cuda]" \
  physicalai-studio serve --no-browser
```

Local wheel:

```bash
uvx --isolated --no-cache \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --from "physicalai-studio[cuda] @ file://${WHEEL}" \
  physicalai-studio serve --no-browser
```

### Why Index Flags Are Required

The source project has `[tool.uv.index]` and `[tool.uv.sources]` entries for local development. Those settings are not standard wheel metadata. Once installed from a wheel, the package dependencies are standard Python requirements, and the installer must be told where hardware-specific PyTorch packages live.

For now, local and published CPU, XPU, and CUDA testing should pass the matching PyTorch index explicitly.

## GitHub Actions

Two app package publishing workflows exist:

```text
.github/workflows/publish-app-testpypi.yml
.github/workflows/publish-app-pypi.yml
```

### TestPyPI

`publish-app-testpypi.yml` runs on `workflow_dispatch`.

It:

1. Checks out the repository.
2. Sets up Python from `application/backend/.python-version`.
3. Installs `uv`.
4. Sets up Node from `application/ui/.nvmrc`.
5. Installs npm `11.14.0`.
6. Runs `application/backend/scripts/build_package.sh`.
7. Uploads `application/backend/dist/` as an artifact.
8. Smoke-tests the wheel metadata and console script entry point with `--no-deps`.
9. Publishes to TestPyPI with trusted publishing.

The smoke test intentionally does not install all runtime dependencies. Full dependency resolution, especially XPU, should be validated on a suitable XPU host.

### PyPI

`publish-app-pypi.yml` runs on stable app tags:

```text
app/vX.Y.Z
```

It also supports manual `workflow_dispatch`, but publishing only runs for `app/v...` tags.

The workflow validates that:

- The tag version matches `application/VERSION`.
- The tag version matches `application/backend/pyproject.toml`.

Then it builds, smoke-tests, uploads the artifact, and publishes to PyPI.

## Common Issues

### `No module named 'pytorch-triton-xpu'`

The PyTorch XPU index was not supplied.

Add:

```bash
--index https://download.pytorch.org/whl/xpu \
--index-strategy unsafe-best-match
```

### Stale Local Wheel Is Used

If you rebuild the wheel without changing the version, `uv` may reuse a cached artifact.

Use:

```bash
--no-cache --isolated
```

Or point directly to the wheel file:

```bash
--from "physicalai-studio[xpu] @ file://${WHEEL}"
```

### Missing UI Assets In The Wheel

Run:

```bash
bash application/backend/scripts/build_package.sh
```

Do not build the app wheel before `application/ui/dist/index.html` exists. The Hatch build hook should fail wheel builds when the UI production build is missing.

### Worker Spawn Circular Import Errors

The installed package uses Python's `spawn` multiprocessing mode. Package-level imports must stay lightweight. Avoid importing lifecycle or scheduler code from package `__init__.py` files, because worker subprocesses import modules during unpickling.

In particular, `core/__init__.py` should not import `core.lifecycle`.
