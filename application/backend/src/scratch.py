from fastapi.requests import HTTPConnection
from fastapi.concurrency import asynccontextmanager
from collections.abc import AsyncGenerator
import os
from multiprocessing.pool import AsyncResult
from multiprocessing.synchronize import Event as EventClass
import asyncio
from abc import abstractmethod, ABC
from multiprocessing import Queue, Pool, Manager, Process, current_process
import time
import signal

from core.logging import setup_logging
from loguru import logger

class BaseWorker(ABC):
    ROLE: str = "Worker"

    def __init__(self):
        self._parent_pid = os.getpid()

    def start(self, terminate_event: EventClass) -> None:
        self._terminate_event = terminate_event
        self.name = self._auto_name()
        current_process().name = self.name
        setup_logging()
        with logger.contextualize(worker=self.name):
            self.setup()
            self.run_loop()

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def run_loop(self) -> None:
        pass

    def should_stop(self) -> bool:
        return self._terminate_event.is_set()

    def _auto_name(self) -> str:
        """Generate a name for the process based on its role and PIDs."""
        return "-".join([self.ROLE, str(self._parent_pid), str(os.getpid())])


class ExampleWorker(BaseWorker):
    ROLE: str = "ExampleWorker"

    def __init__(self, duration: float = 10):
        super().__init__()
        self.duration = duration

    def setup(self) -> None:
        pass

    def run_loop(self) -> None:
        start = time.perf_counter()
        logger.info(f"start example worker for {self.duration}")
        while not self.should_stop() and time.perf_counter() - start < self.duration:
            logger.info(f"{time.perf_counter() - start}")
            time.sleep(0.5)
        logger.info("done")



class WorkerPool:
    def __init__(self, number_of_workers: int):
        self.pool = Pool(number_of_workers, initializer = self._install_signal_policy)
        self.manager = Manager()
        self.terminate_event = self.manager.Event()

    @staticmethod
    def _install_signal_policy() -> None:
        """
        Ignore shutdown signals (SIGINT) in child processes.

        This function prevents child processes from handling shutdown signals directly,
        ensuring that cleanup is coordinated through the parent process via the stop_event
        mechanism.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def start_process(self, worker: BaseWorker) -> AsyncResult:
        return self.pool.apply_async(worker.start, [self.terminate_event])

    async def start_process_async(self, worker: BaseWorker) -> None:
        res = self.start_process(worker)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, res.get)

    def teardown(self) -> None:
        self.terminate_event.set()
        self.pool.close()
        self.pool.join()



from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    pool = WorkerPool(5)
    app.state.pool = pool

    yield

    app.state.pool.teardown()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root(request: HTTPConnection):
    pool = request.app.state.pool
    pool.start_process(ExampleWorker(5))
    #await loop.run_in_executor(None, res.get)
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn
    uvicorn_port = int(os.environ.get("HTTP_SERVER_PORT", 7861))
    uvicorn.run(app, host="0.0.0.0", port=uvicorn_port)
