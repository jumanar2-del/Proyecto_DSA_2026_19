import pytest

from recomendacion.config.core import config
from recomendacion.processing.data_manager import load_dataset
from recomendacion.processing.features import (
    agregar_features_temporales,
    codificar_clientes,
    construir_grid,
)


@pytest.fixture()
def sample_clientes():
    """Carga el dataset de clientes (datos comerciales)."""
    return load_dataset(file_name=config.app_config.train_data_file)


@pytest.fixture()
def sample_resumen():
    """Carga el dataset de resumen de ventas por producto y mes."""
    return load_dataset(file_name="Resumen.csv")


@pytest.fixture()
def sample_input_data(sample_clientes, sample_resumen):
    """Construye el grid completo con features listo para predecir."""
    grid_df, meses, resumen_procesado = construir_grid(sample_resumen)
    grid_df = agregar_features_temporales(grid_df, meses, resumen_procesado)
    grid_df = codificar_clientes(grid_df, sample_clientes)
    return grid_df
