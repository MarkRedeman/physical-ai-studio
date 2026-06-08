from fastapi.testclient import TestClient

from main import app


def test_get_policy_hyper_parameters_act():
    client = TestClient(app)

    response = client.get("/api/policies/act/hyper_parameters")

    assert response.status_code == 200
    body = response.json()
    assert body["policy"] == "act"
    groups = {param["name"]: param for param in body["hyper_parameters"]}
    io_params = {param["name"]: param for param in groups["io"]["hyper_parameters"]}
    vae_params = {param["name"]: param for param in groups["vae"]["hyper_parameters"]}
    optimizer_params = {param["name"]: param for param in groups["optimizer"]["hyper_parameters"]}
    vision_params = {param["name"]: param for param in groups["vision"]["hyper_parameters"]}

    assert groups["io"]["field_type"] == "group"
    assert "default_value" not in groups["io"]
    assert "allowed_values" not in groups["io"]
    assert io_params["chunk_size"] == {
        "name": "chunk_size",
        "field_type": "integer",
        "default_value": 100,
        "description": "Number of future action steps predicted per policy invocation.",
        "human_name": "Chunk Size",
    }
    assert vae_params["use_vae"]["field_type"] == "boolean"
    assert optimizer_params["optimizer_lr"]["field_type"] == "float"
    assert vision_params["vision_backbone"]["field_type"] == "string"
    assert "allowed_values" not in optimizer_params["optimizer_lr"]
    assert "hyper_parameters" not in optimizer_params["optimizer_lr"]
    assert "compile_model" not in {param["name"] for param in groups["optimizer"]["hyper_parameters"]}


def test_get_policy_hyper_parameters_pi05_and_smolvla():
    client = TestClient(app)

    pi05_response = client.get("/api/policies/pi05/hyper_parameters")
    smolvla_response = client.get("/api/policies/smolvla/hyper_parameters")

    assert pi05_response.status_code == 200
    assert smolvla_response.status_code == 200

    pi05_groups = {param["name"]: param for param in pi05_response.json()["hyper_parameters"]}
    smolvla_groups = {param["name"]: param for param in smolvla_response.json()["hyper_parameters"]}
    pi05_backbone_params = {param["name"]: param for param in pi05_groups["backbone"]["hyper_parameters"]}
    pi05_optimizer_params = {param["name"]: param for param in pi05_groups["optimizer"]["hyper_parameters"]}
    smolvla_architecture_params = {
        param["name"]: param for param in smolvla_groups["architecture"]["hyper_parameters"]
    }
    smolvla_fine_tuning_params = {
        param["name"]: param for param in smolvla_groups["fine_tuning"]["hyper_parameters"]
    }

    assert pi05_backbone_params["dtype"]["field_type"] == "choice"
    assert pi05_backbone_params["dtype"]["default_value"] == "bfloat16"
    assert pi05_backbone_params["dtype"]["allowed_values"] == ["bfloat16", "float32"]
    assert "hyper_parameters" not in pi05_backbone_params["dtype"]
    assert pi05_optimizer_params["optimizer_lr"]["field_type"] == "float"
    assert smolvla_architecture_params["vlm_model_name"]["field_type"] == "string"
    assert smolvla_fine_tuning_params["freeze_vision_encoder"]["field_type"] == "boolean"


def test_get_policy_hyper_parameters_unknown_policy():
    client = TestClient(app)

    response = client.get("/api/policies/unknown/hyper_parameters")

    assert response.status_code == 404
