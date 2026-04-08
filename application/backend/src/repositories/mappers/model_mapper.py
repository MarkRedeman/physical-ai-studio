from db.schema import ModelDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Model


class ModelMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Model) -> ModelDB:
        return ModelDB(
            id=str(db_schema.id),
            name=db_schema.name,
            path=db_schema.path,
            policy=db_schema.policy,
            properties=db_schema.properties,
            created_at=db_schema.created_at,
            dataset_id=str(db_schema.dataset_id) if db_schema.dataset_id else None,
            project_id=str(db_schema.project_id),
            snapshot_id=str(db_schema.snapshot_id) if db_schema.snapshot_id else None,
            parent_model_id=str(db_schema.parent_model_id) if db_schema.parent_model_id else None,
            version=db_schema.version,
            train_job_id=str(db_schema.train_job_id) if db_schema.train_job_id else None,
        )

    @staticmethod
    def from_schema(model: ModelDB) -> Model:
        return Model.model_validate(model, from_attributes=True)
