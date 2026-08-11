import asyncio
import csv
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import anyio
from loguru import logger
from sse_starlette import ServerSentEvent

from exceptions import InvalidResourceError, ResourceType
from schemas.job import Job, JobType
from schemas.model import Model
from settings import Settings


class ModelMetricsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @classmethod
    async def get_model_metrics_path(cls, model: Model) -> Path:
        return Path(model.path) / "version_0" / "metrics.csv"

    async def get_model_job_metrics_path(self, job: Job) -> Path:
        if job.type != JobType.TRAINING:
            raise InvalidResourceError(ResourceType.JOB, f"{str(job.id)} {job.type} is not of type {JobType.TRAINING}")
        return self.settings.cache_dir / str(job.id) / "version_0" / "metrics.csv"

    @classmethod
    async def tail_csv_file(cls, path: Path) -> AsyncGenerator[ServerSentEvent]:
        """Async generator that live-tails a metrics CSV and yields SSE events.

        Rows already present when the stream opens are emitted first, then new
        rows are streamed as they are appended. Lightning's ``CSVLogger``
        truncates and rewrites ``metrics.csv`` when a new metric column first
        appears mid-training (e.g. ``val/loss``); the on-disk header is
        re-checked on each EOF so the file is re-opened from the top after such
        a rewrite, with already-emitted rows deduplicated.
        """
        emitted: set[tuple] = set()
        try:
            while True:
                async with await anyio.open_file(path, encoding="utf-8") as f:
                    header_line = await f.readline()
                    if not header_line or not header_line.strip():
                        # File not written yet (or mid-rewrite); wait and retry.
                        await asyncio.sleep(0.5)
                        continue
                    headers = next(csv.reader([header_line]))
                    headers = [header.replace("/", "_") for header in headers]

                    while True:
                        line = await f.readline()
                        if not line:
                            # At EOF, detect a truncate+rewrite (Lightning header
                            # growth) by comparing the on-disk header, then
                            # re-open from the top to recover the full history.
                            try:
                                async with await anyio.open_file(path, encoding="utf-8") as peek:
                                    current_header = await peek.readline()
                            except OSError:
                                current_header = ""
                            if current_header != header_line:
                                break
                            await asyncio.sleep(0.5)
                            continue

                        row = next(csv.reader([line]))
                        data = {key: cls._parse_value(value) for key, value in zip(headers, row)}
                        key = tuple(
                            data.get(column) for column in ("step", "train_loss", "train_loss_step", "val_loss")
                        )
                        if key in emitted:
                            continue
                        emitted.add(key)
                        yield ServerSentEvent(data=json.dumps(data))
        except asyncio.CancelledError:
            logger.debug(f"SSE log stream cancelled for {path}")
        except GeneratorExit:
            logger.debug(f"SSE log stream closed for {path}")

    @staticmethod
    async def empty_metrics_stream() -> AsyncGenerator[ServerSentEvent]:
        """Yield a terminal SSE event for sources with no log file yet."""
        yield ServerSentEvent(data="DONE")

    @staticmethod
    def _parse_value(s: str) -> int | float | str | None:
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except ValueError:
            if s == "":
                return None
            return s
