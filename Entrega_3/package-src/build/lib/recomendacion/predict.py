import typing as t

import pandas as pd

from recomendacion._version import __version__ as _version
from recomendacion.config.core import config
from recomendacion.processing.data_manager import load_pipeline
from recomendacion.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_recomendacion_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Genera predicciones de compra usando el pipeline entrenado.

    Args:
        input_data: DataFrame o dict con las features del modelo.

    Returns:
        Dict con 'predictions' (lista de probabilidades), 'version' y 'errors'.
    """
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _recomendacion_pipe.predict_proba(
            validated_data[config.model_config.features]
        )[:, 1]
        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results


def recomendar_top_n(
    *,
    input_data: pd.DataFrame,
    top_n: int = None,
) -> pd.DataFrame:
    """Genera el ranking Top-N de productos recomendados por cliente.

    Args:
        input_data: DataFrame con features + columnas Cliente, CodProducto, compro.
        top_n: Número de recomendaciones por cliente. Usa config si es None.

    Returns:
        DataFrame con columnas Cliente, ranking, CodProducto, prob_compra.
    """
    if top_n is None:
        top_n = config.model_config.top_n

    cfg = config.model_config
    result = make_prediction(input_data=input_data)

    if result["errors"]:
        raise ValueError(f"Error en validación de inputs: {result['errors']}")

    input_data = input_data.copy()
    input_data["prob_compra"] = result["predictions"]

    # Excluir productos ya comprados en el período actual
    df_nuevos = input_data[input_data["compro"] == 0].copy()

    recomendaciones = (
        df_nuevos.sort_values(
            [cfg.client_col, "prob_compra"], ascending=[True, False]
        )
        .groupby(cfg.client_col)
        .head(top_n)
        .reset_index(drop=True)
    )
    recomendaciones["ranking"] = recomendaciones.groupby(cfg.client_col).cumcount() + 1

    return recomendaciones[[cfg.client_col, "ranking", cfg.product_col, "prob_compra"]]
