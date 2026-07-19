from schemas.robot import RobotAdapter, RobotWithConnectionStateAdapter


def test_bimanual_so101_plugin_model_includes_studio_robot_fields() -> None:
    robot = RobotAdapter.validate_python(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Bimanual follower",
            "type": "BimanualSO101_Follower",
            "payload": {
                "left_serial_number": "left-arm",
                "right_serial_number": "right-arm",
                "left_calibration_id": "00000000-0000-0000-0000-000000000002",
                "right_calibration_id": "00000000-0000-0000-0000-000000000003",
            },
        }
    )

    assert robot.type == "BimanualSO101_Follower"
    assert robot.name == "Bimanual follower"
    assert robot.payload.left_serial_number == "left-arm"


def test_bimanual_so101_plugin_model_supports_connection_status() -> None:
    robot = RobotWithConnectionStateAdapter.validate_python(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Bimanual follower",
            "type": "BimanualSO101_Follower",
            "payload": {
                "left_serial_number": "left-arm",
                "right_serial_number": "right-arm",
            },
            "connection_status": "online",
        }
    )

    assert robot.connection_status == "online"
