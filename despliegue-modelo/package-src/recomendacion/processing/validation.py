from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from recomendacion.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Filtra filas con valores nulos en las features del modelo."""
    validated_data = input_data.copy()
    features_con_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=features_con_na, inplace=True)
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Valida que las features del modelo tengan los tipos y valores esperados."""
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    compro_mes_ant: Optional[int]
    veces_comprado: Optional[float]
    freq_compra: Optional[float]
    ventas_cliente_ant: Optional[float]
    productos_ant: Optional[float]
    Frecuencia_de_visita: Optional[float] = None  # alias handled at merge time

    class Config:
        populate_by_name = True
        extra = "allow"


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
