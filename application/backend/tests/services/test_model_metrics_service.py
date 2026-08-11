import asyncio
import csv
import json
from pathlib import Path

from sse_starlette import ServerSentEvent

from services.model_metrics_service import ModelMetricsService


async def _collect_until(
    gen,
    stop_after_quiet: float = 1.5,
    max_wait: float = 6.0,
) -> list[dict]:
    """Collect events until the generator stays quiet for a while.

    Uses a timeout well above the generator's internal 0.5s poll so an idle
    generator is not cancelled mid-read. Returns all parsed rows.
    """
    rows: list[dict] = []
    start = asyncio.get_event_loop().time()
    last_event = start
    try:
        while asyncio.get_event_loop().time() - start < max_wait:
            try:
                event = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                break
            rows.append(json.loads(event.data))
            last_event = asyncio.get_event_loop().time()
            if last_event - start >= stop_after_quiet:
                break
    finally:
        await gen.aclose()
    return rows


def _write_initial_rows(path: Path, step: int, count: int) -> None:
    """Write a metrics.csv with the initial 4-column Lightning header."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr-AdamW", "step", "train/loss"])
        for _ in range(count):
            writer.writerow(["", "0.0001", step, ""])
            writer.writerow(["0", "", step, f"{1.0 - step / 1000:.6f}"])
            step += 50


def _rewrite_with_new_header(path: Path) -> None:
    """Mimic Lightning's ``_rewrite_with_new_header`` truncate+rewrite."""
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr-AdamW", "step", "train/loss", "val/loss"])
        writer.writeheader()
        writer.writerows(rows)


async def test_tail_csv_file_emits_existing_rows_first(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_initial_rows(path, step=49, count=3)

    rows = await _collect_until(ModelMetricsService.tail_csv_file(path))

    steps = {row["step"] for row in rows if row["train_loss"] is not None}
    assert steps == {49, 99, 149}


async def test_tail_csv_file_keeps_streaming_appended_rows(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_initial_rows(path, step=49, count=2)

    async def append_later() -> None:
        await asyncio.sleep(0.2)
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "0.0001", 199, ""])
            writer.writerow(["1", "", 199, "0.800000"])

    gen = ModelMetricsService.tail_csv_file(path)
    task = asyncio.create_task(append_later())
    rows = await _collect_until(gen)
    await task

    steps = {row["step"] for row in rows if row["train_loss"] is not None}
    assert steps == {49, 99, 199}


async def test_tail_csv_file_survives_header_rewrite(tmp_path) -> None:
    """Lightning truncates and rewrites metrics.csv when val/loss first appears.

    The stream must recover the full history (no lost rows, no duplicates)
    even though the on-disk header gains a column mid-stream.
    """
    path = tmp_path / "metrics.csv"
    _write_initial_rows(path, step=49, count=4)

    async def rewrite_later() -> None:
        await asyncio.sleep(0.2)
        _rewrite_with_new_header(path)
        await asyncio.sleep(0.2)
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "0.0001", 249, ""])
            writer.writerow(["2", "", 249, "0.750000"])

    gen = ModelMetricsService.tail_csv_file(path)
    task = asyncio.create_task(rewrite_later())
    rows = await _collect_until(gen, max_wait=8.0)
    await task

    loss_steps = [row["step"] for row in rows if row["train_loss"] is not None]
    assert loss_steps == [49, 99, 149, 199, 249]
