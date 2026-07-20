"""
Script de prueba del paquete instalado.

Uso:
    1. Instalar el wheel:
       pip install recomendacion_productos-0.1.0-py3-none-any.whl

    2. Copiar los CSV al directorio de datasets del paquete instalado,
       o ajustar las rutas en la llamada a load_dataset.

    3. Ejecutar este script desde la carpeta test/:
       python test-package.py
"""

import pandas as pd
from recomendacion.config.core import config
from recomendacion.predict import make_prediction, recomendar_top_n
from recomendacion.processing.data_manager import load_dataset
from recomendacion.processing.features import (
    agregar_features_temporales,
    codificar_clientes,
    construir_grid,
)

# Cargar datos de prueba desde la carpeta test/
clientes = pd.read_csv("test/Clientes.csv", encoding="latin1", sep=";")
clientes.rename(columns={"Deudor": "Cliente"}, inplace=True)

resumen = pd.read_csv("test/Resumen.csv", encoding="latin1", sep=None, engine="python")
for col in ["Venta", "KG", "Ton"]:
    if col in resumen.columns:
        resumen[col] = (
            resumen[col].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

# Construir features
grid_df, meses, resumen_procesado = construir_grid(resumen)
grid_df = agregar_features_temporales(grid_df, meses, resumen_procesado)
grid_df = codificar_clientes(grid_df, clientes)

# Usar último mes como período de predicción
ultimo_mes = sorted(resumen[config.model_config.month_col].unique())[-1]
df_ultimo = grid_df[grid_df[config.model_config.month_col] == ultimo_mes].copy()

# Probar make_prediction
print("=== make_prediction ===")
result = make_prediction(input_data=df_ultimo)
print(f"Versión del modelo : {result['version']}")
print(f"Errores            : {result['errors']}")
print(f"Total predicciones : {len(result['predictions'])}")
print(f"Primeras 5 probs   : {result['predictions'][:5]}")

# Probar recomendar_top_n
print("\n=== recomendar_top_n (Top-3) ===")
recomendaciones = recomendar_top_n(input_data=df_ultimo, top_n=3)
print(f"Clientes con recomendaciones: {recomendaciones[config.model_config.client_col].nunique():,}")
print("\nEjemplo — primeros 3 clientes:")
primeros = recomendaciones[config.model_config.client_col].unique()[:3]
print(
    recomendaciones[
        recomendaciones[config.model_config.client_col].isin(primeros)
    ].to_string(index=False)
)
