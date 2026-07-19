from __future__ import annotations


def test_imports() -> None:
    from physicalai_studio_plugin import (
        BuildRobotCallable,
        CatalogRobot,
        CatalogRobotFactory,
        PayloadContainer,
        PortScanner,
        RobotAdapterOptions,
        RobotAsset,
        RobotCatalogDefinition,
        RobotProbe,
        SerialPortInfo,
    )

    assert BuildRobotCallable
    assert CatalogRobot
    assert CatalogRobotFactory
    assert PayloadContainer
    assert PortScanner
    assert RobotAdapterOptions
    assert RobotAsset
    assert RobotCatalogDefinition
    assert RobotProbe
    assert SerialPortInfo
