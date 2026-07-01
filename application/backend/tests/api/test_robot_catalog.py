from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class TestRobotCatalogApi:
    def test_list_catalog_returns_entries(self):
        client = TestClient(app)

        response = client.get("/api/robots/catalog")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(entry["type"] == "SO101_Follower" for entry in data)

    def test_get_single_catalog_entry(self):
        client = TestClient(app)

        response = client.get("/api/robots/catalog/ReBot_B601_DM_Follower")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "ReBot_B601_DM_Follower"
        assert data["role"] == "follower"
        assert "joint_map" in data

    def test_get_catalog_entry_not_found(self):
        client = TestClient(app)

        response = client.get("/api/robots/catalog/SO101_Unknown")

        assert response.status_code == 400

    def test_discover_for_rebot_type(self):
        client = TestClient(app)
        fake_devices = [
            {
                "connection_string": "/dev/ttyACM0",
                "serial_number": "REBOT-DM-001",
                "robot_type": "unknown",
            }
        ]

        with patch("services.robot_catalog_service.find_robots", return_value=fake_devices):
            response = client.get("/api/robots/catalog/ReBot_B601_DM_Follower/discover")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "ReBot_B601_DM_Follower"
        assert len(data["devices"]) == 1
        assert data["devices"][0]["serial_number"] == "REBOT-DM-001"

    def test_online_for_rebot_type(self):
        client = TestClient(app)
        fake_devices = [
            {
                "connection_string": "/dev/ttyACM0",
                "serial_number": "REBOT-DM-001",
                "robot_type": "unknown",
            }
        ]

        with patch("services.robot_catalog_service.find_robots", return_value=fake_devices):
            response = client.get("/api/robots/catalog/ReBot_B601_DM_Follower/online")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "ReBot_B601_DM_Follower"
        assert data["online"] is True

    def test_catalog_urdf_for_rebot_type(self, tmp_path: Path):
        client = TestClient(app)
        urdf = tmp_path / "stararm102" / "urdf" / "stararm102_description.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot />")

        with patch("robots.catalog.rebot_b601.get_urdf_path", return_value=tmp_path):
            response = client.get("/api/robots/catalog/ReBot_Arm102_Leader/urdf")

        assert response.status_code == 200

    def test_catalog_urdf_for_so101_type(self, tmp_path: Path):
        client = TestClient(app)
        urdf = tmp_path / "SO101" / "so101_new_calib.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot />")

        with patch("robots.catalog.assets.get_builtin_robot_assets_root", return_value=tmp_path):
            response = client.get("/api/robots/catalog/SO101_Follower/urdf")

        assert response.status_code == 200

    def test_catalog_asset_for_so101_type(self, tmp_path: Path):
        client = TestClient(app)
        asset = tmp_path / "SO101" / "assets" / "base_so101_v2.stl"
        asset.parent.mkdir(parents=True)
        asset.write_text("solid mesh")

        with patch("robots.catalog.assets.get_builtin_robot_assets_root", return_value=tmp_path):
            response = client.get("/api/robots/catalog/SO101_Follower/assets/base_so101_v2.stl")

        assert response.status_code == 200

    def test_catalog_urdf_for_trossen_type(self, tmp_path: Path):
        client = TestClient(app)
        urdf = tmp_path / "widowx" / "urdf" / "generated" / "wxai" / "wxai_follower.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot />")

        with patch("robots.catalog.assets.get_builtin_robot_assets_root", return_value=tmp_path):
            response = client.get("/api/robots/catalog/Trossen_WidowXAI_Follower/urdf")

        assert response.status_code == 200
