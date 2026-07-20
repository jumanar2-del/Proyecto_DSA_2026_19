import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """El endpoint /health debe responder 200 con nombre, api_version y model_version."""
    response = client.get("http://localhost:8001/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Recomendacion Productos API"
    assert "api_version" in data
    assert "model_version" in data


def test_predict_returns_200(client: TestClient, test_data: pd.DataFrame) -> None:
    """El endpoint /predict debe responder 200 con predicciones."""
    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["errors"] is None
    assert data["predictions"] is not None
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0


def test_predict_top_n_per_client(client: TestClient, test_data: pd.DataFrame) -> None:
    """Cada cliente debe tener como máximo top_n recomendaciones."""
    from recomendacion.config.core import config

    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]

    # Contar recomendaciones por cliente
    from collections import Counter
    conteo = Counter(p[config.model_config.client_col] for p in predictions)
    assert all(v <= config.model_config.top_n for v in conteo.values()), (
        f"Hay clientes con más de {config.model_config.top_n} recomendaciones"
    )


def test_predict_has_ranking_column(client: TestClient, test_data: pd.DataFrame) -> None:
    """Cada recomendación debe incluir la columna 'ranking'."""
    payload = {
        "inputs": test_data.head(50).replace({np.nan: None}).to_dict(orient="records")
    }

    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert all("ranking" in p for p in predictions)
    assert all("prob_compra" in p for p in predictions)


def test_predict_missing_columns_returns_422(client: TestClient) -> None:
    """Input sin columnas requeridas debe retornar 422."""
    payload = {
        "inputs": [{"Cliente": "C001", "CodProducto": "P001"}]
    }

    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422
