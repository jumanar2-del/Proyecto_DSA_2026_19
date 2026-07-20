import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from recomendacion import __version__ as model_version
from recomendacion.config.core import config
from recomendacion.predict import recomendar_top_n
from recomendacion.processing.features import (
    agregar_features_temporales,
    codificar_clientes,
    construir_grid,
)
from recomendacion.processing.data_manager import load_dataset

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Verifica que la API esté en funcionamiento."""
    health = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version,
    )
    return health.model_dump()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Genera el Top-N de productos recomendados para los clientes del input.

    El modelo subyacente es el mejor entre LogisticRegression, RandomForest y
    GradientBoosting, seleccionado por ROC-AUC durante el entrenamiento.

    Recibe un DataFrame con las features ya construidas (grid completo del último mes)
    y devuelve las recomendaciones ordenadas por probabilidad de compra.
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Recibiendo predicción para {len(input_df)} filas de entrada.")

    # Validar que las columnas esperadas estén presentes
    missing = [c for c in config.model_config.features if c not in input_df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan columnas requeridas: {missing}",
        )

    # Agregar columna compro si no viene (para filtrar ya comprados)
    if "compro" not in input_df.columns:
        input_df["compro"] = 0

    try:
        recomendaciones = recomendar_top_n(
            input_data=input_df.replace({np.nan: None}),
            top_n=config.model_config.top_n,
        )
    except Exception as e:
        logger.error(f"Error al generar recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Recomendaciones generadas para {recomendaciones[config.model_config.client_col].nunique()} clientes.")

    # Serializar resultado como lista de dicts
    predictions = recomendaciones.to_dict(orient="records")

    return schemas.PredictionResults(
        errors=None,
        version=model_version,
        predictions=predictions,
    )
