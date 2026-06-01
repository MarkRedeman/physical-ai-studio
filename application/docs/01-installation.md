# Installation

This guide helps you install Physical AI Studio and confirm that the UI is ready.

## Choose how to run
[//]: # (Screenshot suggestion: side-by-side view of the docs snippets showing Docker and native options, with Docker visually highlighted as recommended.)

Most users should start with Docker:

- Docker starts both backend and UI together.
- You open one URL and can begin setting up projects right away.

Use our native setup if you are planning to contribute to the project and want to make changes to either the backend or UI.

## Install with Docker (recommended)
[//]: # (Screenshot suggestion: terminal in application/docker showing cp .env.example .env and docker compose up completing successfully.)

From `application/docker/`:

```bash
./setup-devices.sh --xpu # or use --cuda, or --cpu
docker compose up -d
```

When startup is done, open `http://localhost:7860`.

## First UI check
[//]: # (Screenshot suggestion: first loaded app screen showing Projects and, after opening a project, the Robots/Datasets/Models navigation tabs.)

After opening the app, you should see:

- A Projects view.
- The main project navigation after you create/select a project:
  - Robots
  - Datasets
  - Models

If the page opens and you can create a project, your installation is ready.

## If something does not open
[//]: # (Screenshot suggestion: Docker Desktop or terminal status indicating containers running, plus browser URL bar at localhost:7860.)

- Confirm Docker is running.
- Confirm no other app is already using the same port.
- Restart the stack from `application/docker/`.

## Alternatively install natively 

You may choose to run the backend and UI directly on your system using your local python and node setup. This is mainly intended as a development setup, and requires installing prerequisites on your system.

### Prerequisites

Before starting the backend, install the system libraries.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libusb-1.0-0 \
  libusb-1.0-0-dev \
  libclang-dev \
  pkg-config \
  build-essential \
  g++ \
  git
```

If you are using another Linux distribution, install equivalent packages before running
the native setup steps below.

### Backend

Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/), then run the commands below,

```bash
cd backend
uv sync --extra xpu # or `--extra cpu` or `--extra cuda`
./run.sh
```

Backend runs at http://localhost:7860

If you plan to train Hugging Face Hub-backed policies (for example, SmolVLA, Pi0,
and others), configure `HF_TOKEN` in `backend/.env`. See
[Hugging Face Integration](./backend/docs/huggingface_integration.md).

### Frontend

Install [node v24](https://nodejs.org/en/download) (we recommend using nvm), and run the commands below,

```bash
cd ui
npm install
npm run start
```

UI runs at http://localhost:3000

## Next

- Continue with `application/docs/03-getting-started.md`.
- If you are upgrading an existing setup, use `application/docs/02-update-existing-installation.md`.
