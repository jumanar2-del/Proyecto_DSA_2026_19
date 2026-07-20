import pandas as pd
import pytest

from recomendacion.config.core import config
from recomendacion.predict import make_prediction, recomendar_top_n


def test_make_prediction_returns_list(sample_input_data):
    """make_prediction debe devolver una lista de probabilidades sin errores."""
    result = make_prediction(input_data=sample_input_data)

    assert result.get("errors") is None
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert len(predictions) > 0


def test_prediction_probabilities_in_range(sample_input_data):
    """Las probabilidades deben estar en el rango [0, 1]."""
    result = make_prediction(input_data=sample_input_data)
    predictions = result.get("predictions")

    assert all(0.0 <= p <= 1.0 for p in predictions), (
        "Hay probabilidades fuera del rango [0, 1]"
    )


def test_prediction_version_present(sample_input_data):
    """El resultado debe incluir la versión del modelo."""
    result = make_prediction(input_data=sample_input_data)
    assert result.get("version") is not None


def test_recomendar_top_n_columns(sample_input_data):
    """recomendar_top_n debe devolver las columnas esperadas."""
    recomendaciones = recomendar_top_n(input_data=sample_input_data)

    expected_cols = {
        config.model_config.client_col,
        "ranking",
        config.model_config.product_col,
        "prob_compra",
    }
    assert expected_cols.issubset(set(recomendaciones.columns))


def test_recomendar_top_n_max_por_cliente(sample_input_data):
    """Cada cliente debe tener como máximo top_n recomendaciones."""
    top_n = config.model_config.top_n
    recomendaciones = recomendar_top_n(input_data=sample_input_data, top_n=top_n)

    conteo = recomendaciones.groupby(config.model_config.client_col).size()
    assert (conteo <= top_n).all(), (
        f"Hay clientes con más de {top_n} recomendaciones"
    )


def test_recomendar_no_incluye_ya_comprados(sample_input_data):
    """Los productos ya comprados (compro=1) no deben aparecer en las recomendaciones."""
    recomendaciones = recomendar_top_n(input_data=sample_input_data)

    productos_comprados = sample_input_data[sample_input_data["compro"] == 1][
        [config.model_config.client_col, config.model_config.product_col]
    ]

    merged = recomendaciones.merge(
        productos_comprados,
        on=[config.model_config.client_col, config.model_config.product_col],
        how="inner",
    )
    assert merged.empty, "Se están recomendando productos ya comprados por el cliente"
