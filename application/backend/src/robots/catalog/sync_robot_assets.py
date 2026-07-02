from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from robots.catalog.assets import get_builtin_robot_assets_root

SO101_REPO_URL = "https://github.com/TheRobotStudio/SO-ARM100.git"
WIDOWX_REPO_URL = "https://github.com/TrossenRobotics/trossen_arm_description.git"


def sync_robot_assets(target_dir: Path | None = None) -> None:
    """Sync SO101 and WidowX assets into backend static storage."""
    target_root = target_dir or get_builtin_robot_assets_root()
    target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        so101_repo_path = tmp_root / "so101-repo"
        _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                "--filter=blob:none",
                "--sparse",
                SO101_REPO_URL,
                str(so101_repo_path),
            ]
        )
        _run_git(["sparse-checkout", "set", "--no-cone", "Simulation/SO101"], cwd=so101_repo_path)

        so101_target = target_root / "SO101"
        if so101_target.exists():
            shutil.rmtree(so101_target)
        shutil.copytree(so101_repo_path / "Simulation" / "SO101", so101_target)

        widowx_repo_path = tmp_root / "widowx-repo"
        _run_git(["clone", "--depth", "1", "--branch", "main", WIDOWX_REPO_URL, str(widowx_repo_path)])

        widowx_target = target_root / "widowx"
        if widowx_target.exists():
            shutil.rmtree(widowx_target)
        shutil.copytree(widowx_repo_path, widowx_target, ignore=shutil.ignore_patterns(".git"))


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git executable was not found")
    subprocess.run([git, *args], cwd=cwd, check=True)  # noqa: S603
