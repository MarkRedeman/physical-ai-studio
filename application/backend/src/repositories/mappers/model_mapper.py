from datetime import datetime

from db.schema import ModelDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Model


class ModelMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Model) -> ModelDB:
        data = db_schema.model_dump(mode="json")
        # model_dump(mode="json") serialises datetime to ISO strings, but
        # SQLite requires native datetime objects.
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return ModelDB(**data)

    @staticmethod
    def from_schema(model: ModelDB) -> Model:
        return Model.model_validate(model, from_attributes=True)
