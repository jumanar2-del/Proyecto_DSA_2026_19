from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recomendacion.config.core import config
from recomendacion.processing.data_manager import load_dataset
from recomendacion.processing.features import (
    agregar_features_temporales,
    codificar_clientes,
    construir_grid,
)
from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Construye el grid del último mes listo para enviar al endpoint /predict."""
    clientes = load_dataset(file_name=config.app_config.train_data_file)
    resumen = load_dataset(file_name="Resumen.csv")

    grid_df, meses, resumen_procesado = construir_grid(resumen)
    grid_df = agregar_features_temporales(grid_df, meses, resumen_procesado)
    grid_df = codificar_clientes(grid_df, clientes)

    ultimo_mes = sorted(resumen[config.model_config.month_col].unique())[-1]
    return grid_df[grid_df[config.model_config.month_col] == ultimo_mes].copy()


@pytest.fixture()
def client() -> Generator:
    """Cliente de prueba de FastAPI."""
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
