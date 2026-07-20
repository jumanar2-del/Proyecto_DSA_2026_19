from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    """Esquema de respuesta del endpoint /predict."""
    errors: Optional[Any]
    version: str
    predictions: Optional[List[Dict[str, Any]]]


class DataInputSchema(BaseModel):
    """
    Features requeridas por el modelo para un par (cliente, producto, mes).
    Todas son opcionales para permitir valores nulos en el input.
    """
    compro_mes_ant: Optional[int]
    veces_comprado: Optional[float]
    freq_compra: Optional[float]
    ventas_cliente_ant: Optional[float]
    productos_ant: Optional[float]
    Frecuencia_de_visita: Optional[float] = None
    oficina_enc: Optional[int]
    seg5_enc: Optional[int]
    seg4_enc: Optional[int]
    producto_enc: Optional[int]
    # Columnas de contexto (no son features del modelo pero se usan para rankear)
    Cliente: Optional[Any]
    CodProducto: Optional[Any]
    compro: Optional[int] = 0

    class Config:
        extra = "allow"  # permite columnas adicionales sin error


class MultipleDataInputs(BaseModel):
    """Wrapper para múltiples filas de input."""
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Cliente": "C001",
                        "CodProducto": "P001",
                        "compro_mes_ant": 0,
                        "veces_comprado": 2.0,
                        "freq_compra": 0.4,
                        "ventas_cliente_ant": 150000.0,
                        "productos_ant": 3.0,
                        "Frecuencia_de_visita": 2.0,
                        "oficina_enc": 1,
                        "seg5_enc": 0,
                        "seg4_enc": 2,
                        "producto_enc": 5,
                        "compro": 0,
                    }
                ]
            }
        }
