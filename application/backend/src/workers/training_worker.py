# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import datetime
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from core.logging.utils import job_logging_ctx
from models.utils import load_policy, setup_policy
from services.snapshot_service import SnapshotService
from settings import get_settings

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass


from loguru import logger
from physicalai.data import LeRobotDataModule
from physicalai.export import ExportablePolicyMixin
from physicalai.train import Trainer

from schemas import Job, Model, Snapshot
from schemas.base_job import JobStatus
from schemas.calibration import Calibration
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.job import TrainJobPayload
from services import DatasetService, ModelService
from services.environment_service import EnvironmentService
from services.event_processor import EventType
from services.job_service import JobService
from services.model_manifest_service import ModelManifestService
from services.robot_calibration_service import RobotCalibrationService
from services.training_service import (
    TrainingLogCallback,
    TrainingService,
    TrainingTrackingCallback,
    TrainingTrackingDispatcher,
)
from utils.device import get_lightning_strategy, get_torch_device
from workers.base import BaseProcessWorker

SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    ROLE = "TrainingWorker"

    def __init__(self, stop_event: EventClass, interrupt_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        job_service = JobService()
        logger.info("Training Worker is running")
        while not self.should_stop():
            settings = get_settings()

            job = await job_service.get_pending_train_job()
            if job is not None:
                with job_logging_ctx(job_id=str(job.id)):
                    payload = TrainJobPayload.model_validate(job.payload)
                    id = uuid4()

                    base_model = None
                    if payload.base_model_id is not None:
                        base_model = await ModelService.get_model_by_id(payload.base_model_id)

                    dataset = await DatasetService.get_dataset_by_id(payload.dataset_id)
                    model_dir = Path(str(settings.models_dir / str(id)))
                    snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name()
                    snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)

                    model = Model(
                        id=id,
                        project_id=payload.project_id,
                        dataset_id=payload.dataset_id,
                        path=str(model_dir),
                        name=payload.model_name,
                        snapshot_id=snapshot.id,
                        policy=payload.policy,
                        properties={},
                        train_job_id=job.id,
                        parent_model_id=payload.base_model_id,
                        version=base_model.version + 1 if base_model else 1,
                        created_at=None,
                    )

                    self.interrupt_event.clear()
                    await asyncio.create_task(self._train_model(job, model, dataset, snapshot, payload, base_model))
            self.stop_aware_sleep(0.5)

    async def setup(self) -> None:
        await super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def teardown(self) -> None:
        await super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def _train_model(
        self,
        job: Job,
        model: Model,
        dataset: Dataset,
        snapshot: Snapshot,
        payload: TrainJobPayload,
        base_model: Model | None = None,
    ):
        settings = get_settings()
        await JobService.update_job(
            job=job,
            update={
                "status": JobStatus.RUNNING,
                "message": "Training started",
                "start_time": datetime.datetime.now(tz=datetime.UTC),
            },
        )
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=self.interrupt_event,
        )
        try:
            path = Path(model.path)
            cache_path = settings.cache_dir / str(job.id)

            # Resolve training device -- explicit from payload or auto-detected
            device_type = payload.device.type if payload.device else None
            device_index = payload.device.index if payload.device else None

            accelerator = get_torch_device(device_type)

            l_dm = LeRobotDataModule(
                repo_id="snapshot",  # doesnt matter for loading the data.
                root=snapshot.path,
                train_batch_size=payload.batch_size,
                num_workers=payload.num_workers,
                val_split=payload.val_split,
            )

            if base_model is not None:
                policy = load_policy(base_model, compile_model=payload.compile_model)
            else:
                policy = setup_policy(model, compile_model=payload.compile_model)

            precision = str(payload.precision)
            strategy = get_lightning_strategy(device_type)
            devices = [device_index] if device_index is not None else 1

            checkpoint_callback = ModelCheckpoint(
                dirpath=cache_path,
                filename="model",  # filename without suffix
                save_top_k=1,
                monitor="val/loss",
                mode="min",
            )
            csv_logger = CSVLogger(cache_path.parent, name=cache_path.stem)

            def _create_trainer() -> Trainer:
                return Trainer(
                    logger=csv_logger,
                    callbacks=[
                        checkpoint_callback,
                        TrainingTrackingCallback(
                            shutdown_event=self._stop_event,
                            interrupt_event=self.interrupt_event,
                            dispatcher=dispatcher,
                        ),
                        TrainingLogCallback(),
                    ],
                    accelerator=accelerator,
                    strategy=strategy,
                    devices=devices,
                    max_steps=payload.max_steps,
                    auto_scale_batch_size=payload.auto_scale_batch_size,
                    precision=precision,
                    check_val_every_n_epoch=1,
                )

            trainer = _create_trainer()

            dispatcher.start()
            trainer.fit(model=policy, datamodule=l_dm)

            final_checkpoint = cache_path / "model.ckpt"
            trainer.save_checkpoint(final_checkpoint)

            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(cache_path, path)

            export_policy = policy
            if payload.compile_model and model.policy in ["act", "smolvla"]:
                try:
                    logger.info("Reloading non-compiled policy for export")
                    export_policy = load_policy(model, compile_model=False)
                except Exception as e:
                    logger.warning("Failed to reload non-compiled policy for export; falling back to trained policy")
                    logger.exception(e)

            await self._export_policy(policy=export_policy, path=path, job=job)
            environment = await self._get_training_environment(dataset)
            calibrations = await self._get_environment_calibrations(environment)
            self._write_model_metadata(
                path,
                model=model,
                dataset=dataset,
                environment=environment,
                calibrations=calibrations,
            )

            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
            )
            model = await ModelService.create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {e}"
            )
        self.interrupt_event.set()
        if dispatcher.is_alive():
            dispatcher.join(timeout=10)
        self.queue.put((EventType.JOB_UPDATE, job))

    async def _export_policy(self, policy: object, path: Path, job: Job) -> None:
        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        logger.info("Starting model export for trained policy")
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            try:
                logger.info("Exporting model to {} format", backend_name)
                await JobService.update_job_status(
                    job_id=job.id,
                    status=JobStatus.RUNNING,
                    message=f"Exporting to {backend_name} format",
                )
                export_dir = path / "exports" / backend
                policy.export(export_dir, backend=backend)
                logger.info("Model export to {} completed", backend_name)
            except Exception as e:
                logger.error("Failed exporting model to {} format", backend_name)
                logger.exception(e)

    @staticmethod
    async def _get_training_environment(dataset: Dataset) -> EnvironmentWithRelations | None:
        try:
            return await EnvironmentService.get_environment_by_id(dataset.project_id, dataset.environment_id)
        except Exception as e:
            logger.warning("Failed loading training environment for model metadata")
            logger.exception(e)
            return None

    @staticmethod
    async def _get_environment_calibrations(environment: EnvironmentWithRelations | None) -> dict[UUID, Calibration]:
        if environment is None:
            return {}

        calibration_service = RobotCalibrationService(robot_manager=None, settings=get_settings())
        calibrations: dict[UUID, Calibration] = {}
        for robot_config in environment.robots:
            robot = robot_config.robot
            if robot.active_calibration_id is None:
                continue
            try:
                calibrations[robot.id] = await calibration_service.get_calibration(robot.active_calibration_id)
            except Exception as e:
                logger.warning("Failed loading active calibration for robot {}", robot.name)
                logger.exception(e)

        return calibrations

    @staticmethod
    def _write_model_metadata(
        path: Path,
        model: Model,
        dataset: Dataset,
        environment: EnvironmentWithRelations | None,
        calibrations: dict[UUID, Calibration],
    ) -> None:
        try:
            manifest_path = ModelManifestService.write_root_manifest(path)
            if manifest_path is None:
                logger.warning("Skipping model metadata: torch export manifest not found or invalid")
                return

            logger.info("Root model manifest written to {}", manifest_path)

            if environment is not None:
                environment_path = ModelManifestService.write_environment_description(path, environment, calibrations)
                logger.info("Model environment description written to {}", environment_path)

                calibration_path = ModelManifestService.write_runtime_calibration(path, environment, calibrations)
                if calibration_path is not None:
                    logger.info("Model runtime calibration written to {}", calibration_path)

            readme_path = ModelManifestService.write_model_card(
                path,
                model=model,
                dataset=dataset,
                environment=environment,
                calibrations=calibrations,
            )
            if readme_path is None:
                logger.warning("Skipping model card README: root model manifest not found or invalid")
                return

            logger.info("Model card README written to {}", readme_path)
        except Exception as e:
            logger.warning("Failed writing model metadata")
            logger.exception(e)
