"""Run with:  pip install pytest httpx  &&  pytest"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # "with" runs the startup code, so the model gets loaded
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "ML API is running"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
        ({"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}, "versicolor"),
        ({"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.1}, "virginica"),
    ],
)
def test_predict_known_flowers(client, payload, expected):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == expected
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_missing_field(client):
    r = client.post("/predict", json={"sepal_length": 5.1})
    assert r.status_code == 422


def test_predict_invalid_value(client):
    bad = {"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict", json=bad).status_code == 422


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
