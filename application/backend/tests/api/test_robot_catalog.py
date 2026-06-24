from fastapi.testclient import TestClient

from api.dependencies import get_robot_catalog_service
from main import app
from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import RobotType
from services.robot_catalog_service import RobotCatalogService


def _override_catalog_service() -> RobotCatalogService:
    return RobotCatalogService(RobotCatalogRegistry())


app.dependency_overrides[get_robot_catalog_service] = _override_catalog_service
client = TestClient(app)


class TestRobotCatalog:
    def test_list_catalog(self):
        response = client.get("/api/robots/catalog")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        entry = data[0]
        assert "type" in entry
        assert "display_name" in entry
        assert "role" in entry

    def test_get_catalog_entry(self):
        response = client.get("/api/robots/catalog/SO101_Follower")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "SO101_Follower"
        assert data["role"] == "follower"

    def test_get_catalog_entry_not_found(self):
        response = client.get("/api/robots/catalog/NonExistent")
        assert response.status_code == 404

    def test_catalog_includes_so101(self):
        response = client.get("/api/robots/catalog")
        assert response.status_code == 200
        types = {e["type"] for e in response.json()}
        assert "SO101_Follower" in types
        assert "SO101_Leader" in types

    def test_catalog_includes_widowx(self):
        response = client.get("/api/robots/catalog")
        assert response.status_code == 200
        types = {e["type"] for e in response.json()}
        assert "Trossen_WidowXAI_Follower" in types
        assert "Trossen_WidowXAI_Leader" in types
        assert "Trossen_Bimanual_WidowXAI_Follower" in types
        assert "Trossen_Bimanual_WidowXAI_Leader" in types

    def test_discover_returns_port_list(self):
        response = client.get("/api/robots/catalog/SO101_Follower/discover")
        assert response.status_code == 200
        data = response.json()
        assert "ports" in data
        assert isinstance(data["ports"], list)

    def test_urdf_endpoint_missing_asset(self):
        response = client.get("/api/robots/catalog/SO101_Follower/urdf")
        assert response.status_code == 404

    def test_asset_endpoint_missing(self):
        response = client.get("/api/robots/catalog/SO101_Follower/missing.urdf")
        assert response.status_code == 404
