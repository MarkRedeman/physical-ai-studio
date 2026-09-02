from __future__ import annotations

from schemas.project_camera import CameraAdapter
from utils.camera_factory import build_camera_config


def test_ip_camera_config_uses_url_without_parsing_fingerprint() -> None:
    camera = CameraAdapter.validate_python(
        {
            "driver": "ipcam",
            "name": "overview",
            "fingerprint": "http://127.0.0.1:8080/cameras/overview/mjpeg",
            "hardware_name": None,
            "payload": {
                "url": "http://127.0.0.1:8080/cameras/overview/mjpeg",
                "width": 640,
                "height": 480,
                "fps": 30,
            },
        }
    )

    config = build_camera_config(camera)

    assert config.class_path == "physicalai.capture.IPCamera"
    assert config.init_args == {
        "url": "http://127.0.0.1:8080/cameras/overview/mjpeg",
        "width": 640,
        "height": 480,
        "fps": 30,
    }
