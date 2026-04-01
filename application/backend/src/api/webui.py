from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    """StaticFiles subclass that serves index.html for unknown paths.

    Serves actual files when they exist (JS, CSS, URDF models, etc.)
    and falls back to index.html for everything else, allowing the
    frontend router to handle client-side routes.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        request_path = scope.get("path", "")
        is_api_path = request_path == "/api" or request_path.startswith("/api/")

        try:
            response = await super().get_response(path, scope)

            if response.status_code == 404 and not is_api_path:
                response = await super().get_response("index.html", scope)

            return response
        except HTTPException as e:
            if e.status_code == 404 and not is_api_path:
                return await super().get_response("index.html", scope)
            raise e
