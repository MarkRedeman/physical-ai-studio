import threading
import asyncio
import signal
from multiprocessing import Manager, Pool
from multiprocessing.pool import AsyncResult

from workers.base import BaseWorker


from loguru import logger

class WorkerPool:
    def __init__(self, number_of_workers: int):
        self.pool = Pool(number_of_workers, initializer=self._install_signal_policy)
        logger.info(f"Setup pool of {number_of_workers} workers")
        self.manager = Manager()
        self.terminate_event = self.manager.Event()
        self._lock = threading.Lock()
        self._num_workers = number_of_workers
        self._active = 0

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
        logger.info(f"Starting process: {worker.ROLE}")
        return self.pool.apply_async(worker.start, [self.terminate_event])

    async def start_process_async(self, worker: BaseWorker) -> None:
        res = self.start_process(worker)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, res.get)

    def teardown(self) -> None:
        self.terminate_event.set()
        self.pool.close()
        self.pool.join()

    @property
    def available_workers(self) -> int:
        with self._lock:
            return self._num_workers - self._active

    def get_status_summary(self) -> dict:
        """
        Generate a summary of the status of all registered workers.

        Returns:
            Dictionary containing total count and individual worker statuses.
        """
        return {
            "total_workers": self.available_workers,
            "max_workers": self._num_workers,
        }
