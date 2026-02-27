import http
from enum import StrEnum
from uuid import UUID


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    PROJECT = "Project"
    ROBOT = "Robot"
    ROBOT_CALIBRATION = "Robot calibration"
    CAMERA = "Camera"
    ENVIRONMENT = "Environment"
    DATASET = "Dataset"
    SNAPSHOT = "Snapshot"
    MODEL = "Model"
    JOB = "JOB"
    JOB_FILE = "JOB_FILE"


class BaseException(Exception):
    """
    Base class for PhysicalAI exceptions with a predefined HTTP error code.

    :param message: str message providing short description of error
    :param error_code: str id of error
    :param http_status: int default http status code to return to user
    """

    def __init__(self, message: str, error_code: str, http_status: int) -> None:
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(message)


class ResourceNotFoundError(BaseException):
    """
    Exception raised when a resource could not be found in database.

    :param resource_id: ID of the resource that was not found
    """

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = (
            message or f"The requested {resource_type} could not be found. {resource_type.title()} ID: `{resource_id}`."
        )

        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        )


class DuplicateJobException(BaseException):
    """
    Exception raised when attempting to submit a duplicate job.

    :param message: str containing a custom message about the duplicate job.
    """

    def __init__(self, message: str = "A job with the same payload is already running or queued") -> None:
        super().__init__(message=message, error_code="duplicate_job", http_status=http.HTTPStatus.CONFLICT)


class ResourceInUseError(BaseException):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str | UUID, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(
            message=msg,
            error_code=f"{resource_type}_not_found",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ResourceAlreadyExistsError(BaseException):
    """
    Exception raised when a resource already exists.

    :param resource_name: Name of the resource that was not found
    """

    def __init__(self, resource_name: str, detail: str) -> None:
        super().__init__(
            message=f"{resource_name} already exists. {detail}",
            error_code=f"{resource_name}_already_exists",
            http_status=http.HTTPStatus.CONFLICT,
        )


class ModelNotRetrainableError(BaseException):
    """Exception raised when attempting to retrain a model that does not support it.

    HuggingFace-imported models cannot be retrained because their checkpoint
    format (state_dict key structure) is incompatible with the native policy
    classes used for training.  See ``retrainable-imported-models.md`` for details.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=(
                f"Model '{model_name}' cannot be retrained. "
                "Models imported from HuggingFace are inference-only and do not support retraining."
            ),
            error_code="model_not_retrainable",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class ImportValidationError(BaseException):
    """Exception raised when a model import archive fails validation.

    Covers bad zip files, unrecognized formats, missing required files,
    invalid config content, and unsupported policy types.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="import_validation_error",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class ImportConversionError(BaseException):
    """Exception raised when HuggingFace model conversion fails.

    Covers errors during model loading (from_pretrained) or export
    to the Physical AI inference format.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="import_conversion_error",
            http_status=http.HTTPStatus.BAD_REQUEST,
        )


class ImportDependencyError(BaseException):
    """Exception raised when a required library for model import is unavailable."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="import_dependency_error",
            http_status=http.HTTPStatus.INTERNAL_SERVER_ERROR,
        )
