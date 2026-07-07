"""
Sistema de Recomendación de Productos - Entrega 2
Proyecto DSA 2026 | Jessica Salazar & Juan Camilo Umaña

Pregunta de negocio:
¿Qué productos se le deberían recomendar a cada cliente según su historial
de compra y sus características comerciales, con el fin de aumentar la
probabilidad de compra y la activación del portafolio?

Enfoque:
- Variable objetivo binaria: ¿compró el cliente el producto en el siguiente mes?
- Unidad de análisis: (cliente, producto, mes)
- Salida: ranking Top-3 de productos recomendados por cliente
- Experimentos versionados con MLflow
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score
)

import mlflow
import mlflow.sklearn

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

clientes = pd.read_csv("Clientes.csv", encoding="latin1", sep=";")
resumen  = pd.read_csv("Resumen.csv",  encoding="latin1", sep=None, engine="python")

# Limpiar valores numéricos con coma decimal (formato colombiano)
for col in ["Venta", "KG", "Ton"]:
    resumen[col] = (
        resumen[col].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

clientes.rename(columns={"Deudor": "Cliente"}, inplace=True)

print(f"Clientes : {clientes.shape[0]:,} filas | {clientes.shape[1]} columnas")
print(f"Resumen  : {resumen.shape[0]:,} filas | {resumen.shape[1]} columnas")
print(f"Meses    : {sorted(resumen['Mes'].unique())}")
print(f"Productos: {sorted(resumen['CodProducto'].unique())}")

# ─────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE CARACTERÍSTICAS (vectorizada)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. INGENIERÍA DE CARACTERÍSTICAS")
print("=" * 60)

meses_ordenados = sorted(resumen["Mes"].unique())
productos       = sorted(resumen["CodProducto"].unique())
todos_clientes  = resumen["Cliente"].unique()
n_meses         = len(meses_ordenados)

# Compra binaria por (Mes, Cliente, CodProducto)
resumen["compro"] = (resumen["Venta"] > 0).astype(int)

pivot = (
    resumen.groupby(["Mes", "Cliente", "CodProducto"])["compro"]
    .max()
    .reset_index()
)

# Grid completo: todos los (Mes, Cliente, CodProducto)
idx_mes      = pd.Index(meses_ordenados, name="Mes")
idx_cliente  = pd.Index(todos_clientes,  name="Cliente")
idx_producto = pd.Index(productos,       name="CodProducto")

grid = pd.MultiIndex.from_product(
    [meses_ordenados, todos_clientes, productos],
    names=["Mes", "Cliente", "CodProducto"]
)
grid_df = pd.DataFrame(index=grid).reset_index()

# Unir compras reales (0 si no compró)
grid_df = grid_df.merge(pivot, on=["Mes", "Cliente", "CodProducto"], how="left")
grid_df["compro"] = grid_df["compro"].fillna(0).astype(int)

# Ordenar
grid_df = grid_df.sort_values(["Cliente", "CodProducto", "Mes"]).reset_index(drop=True)

# --- Features por (cliente, producto, mes) ---

# compro_mes_ant: compra en el mes anterior (shift dentro de grupo)
grid_df["compro_mes_ant"] = (
    grid_df.groupby(["Cliente", "CodProducto"])["compro"].shift(1).fillna(0).astype(int)
)

# veces_comprado: acumulado histórico (excluyendo mes actual)
grid_df["veces_comprado"] = (
    grid_df.groupby(["Cliente", "CodProducto"])["compro"]
    .transform(lambda x: x.shift(1).fillna(0).cumsum())
)

# Índice del mes (posición ordinal)
mes_idx = {m: i+1 for i, m in enumerate(meses_ordenados)}
grid_df["mes_idx"] = grid_df["Mes"].map(mes_idx)

# freq_compra: veces_comprado / meses transcurridos
grid_df["freq_compra"] = grid_df["veces_comprado"] / grid_df["mes_idx"].clip(lower=1)

# Ventas totales del cliente en el mes anterior
ventas_mes = (
    resumen.groupby(["Mes", "Cliente"])["Venta"].sum().reset_index()
    .rename(columns={"Venta": "ventas_cliente_ant", "Mes": "Mes_ant"})
)
grid_df["Mes_ant"] = grid_df.groupby(["Cliente", "CodProducto"])["Mes"].shift(1)
grid_df = grid_df.merge(ventas_mes, on=["Mes_ant", "Cliente"], how="left")
grid_df["ventas_cliente_ant"] = grid_df["ventas_cliente_ant"].fillna(0)

# Número de productos distintos comprados en el mes anterior
prods_mes = (
    resumen[resumen["Venta"] > 0]
    .groupby(["Mes", "Cliente"])["CodProducto"].nunique().reset_index()
    .rename(columns={"CodProducto": "productos_ant", "Mes": "Mes_ant"})
)
grid_df = grid_df.merge(prods_mes, on=["Mes_ant", "Cliente"], how="left")
grid_df["productos_ant"] = grid_df["productos_ant"].fillna(0)

# Target: compra actual (se usará como "siguiente mes" al hacer el split)
grid_df["target"] = grid_df["compro"]

# --- Codificación de variables de clientes ---
le_oficina  = LabelEncoder()
le_seg5     = LabelEncoder()
le_seg4     = LabelEncoder()
le_producto = LabelEncoder()

clientes["oficina_enc"] = le_oficina.fit_transform(clientes["Oficina de ventas"])
clientes["seg5_enc"]    = le_seg5.fit_transform(clientes["Denominación Gr#Clientes 5"])
clientes["seg4_enc"]    = le_seg4.fit_transform(clientes["Denominación Gr#Clientes 4"])

grid_df = grid_df.merge(
    clientes[["Cliente", "Frecuencia de visita", "oficina_enc", "seg5_enc", "seg4_enc"]],
    on="Cliente", how="left"
)
grid_df["producto_enc"] = le_producto.fit_transform(grid_df["CodProducto"])
grid_df.fillna(0, inplace=True)

print(f"Dataset construido: {grid_df.shape[0]:,} filas")
print(f"Tasa de positivos : {grid_df['target'].mean():.2%}")

# ─────────────────────────────────────────────────────────────
# 3. DIVISIÓN TRAIN / TEST  (split temporal: último mes = test)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DIVISIÓN TRAIN / TEST")
print("=" * 60)

FEATURES = [
    "compro_mes_ant",
    "veces_comprado",
    "freq_compra",
    "ventas_cliente_ant",
    "productos_ant",
    "Frecuencia de visita",
    "oficina_enc",
    "seg5_enc",
    "seg4_enc",
    "producto_enc",
]

ultimo_mes = meses_ordenados[-1]
# Necesitamos al menos mes anterior conocido → excluir primer mes del dataset
df_model = grid_df[grid_df["Mes"] >= meses_ordenados[1]].copy()

mask_test = df_model["Mes"] == ultimo_mes
X_train   = df_model.loc[~mask_test, FEATURES]
X_test    = df_model.loc[ mask_test, FEATURES]
y_train   = df_model.loc[~mask_test, "target"]
y_test    = df_model.loc[ mask_test, "target"]

print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")
print(f"Positivos train: {y_train.mean():.2%} | Positivos test: {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────
# 4. EXPERIMENTOS CON MLFLOW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EXPERIMENTOS CON MLFLOW")
print("=" * 60)

# ── Conexión a MLflow en EC2 ──────────────────────────────────────────────────
# IMPORTANTE: reemplaza la IP por la IPv4 pública de tu instancia EC2.
# La encuentras en: AWS Console → EC2 → Instancias → columna "IPv4 pública"
# El servidor MLflow debe estar corriendo en EC2 con:
#   mlflow server --host 0.0.0.0 --port 5000 \
#                 --backend-store-uri sqlite:///mlflow.db \
#                 --default-artifact-root ./mlartifacts
# ─────────────────────────────────────────────────────────────────────────────
EC2_IP = "98.91.21.105"   # <── CAMBIA ESTO por la IP pública de tu EC2
mlflow.set_tracking_uri(f"http://{EC2_IP}:8050")
mlflow.set_experiment("Recomendacion_Productos_DSA2026")

modelos = {
    "LogisticRegression": LogisticRegression(
        max_iter=500, 
        class_weight="balanced", 
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        class_weight="balanced",
        random_state=42, 
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.05,
        random_state=42
    ),
}

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(modelo.get_params())
        mlflow.log_param("features",   str(FEATURES))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("mes_test",   str(ultimo_mes))

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "roc_auc"  : roc_auc_score(y_test, y_prob),
            "pr_auc"   : average_precision_score(y_test, y_prob),
            "f1"       : f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test, y_pred, zero_division=0),
        }

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, artifact_path=nombre)

        resultados[nombre] = {"modelo": modelo, **metricas}

        print(f"  ROC-AUC  : {metricas['roc_auc']:.4f}")
        print(f"  PR-AUC   : {metricas['pr_auc']:.4f}")
        print(f"  F1       : {metricas['f1']:.4f}")
        print(f"  Precision: {metricas['precision']:.4f}")
        print(f"  Recall   : {metricas['recall']:.4f}")

# ─────────────────────────────────────────────────────────────
# 5. SELECCIÓN DEL MEJOR MODELO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SELECCIÓN DEL MEJOR MODELO")
print("=" * 60)

mejor_nombre = max(resultados, key=lambda k: resultados[k]["roc_auc"])
mejor_modelo  = resultados[mejor_nombre]["modelo"]

df_comp = pd.DataFrame(
    {k: {m: v for m, v in v.items() if m != "modelo"} for k, v in resultados.items()}
).T.round(4)

print("\nComparativa de modelos:")
print(df_comp.to_string())
print(f"\nMejor modelo (ROC-AUC): {mejor_nombre} → {resultados[mejor_nombre]['roc_auc']:.4f}")

# Importancia de variables (si aplica)
if hasattr(mejor_modelo, "feature_importances_"):
    fi = pd.Series(mejor_modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nImportancia de variables:")
    print(fi.to_string())

# ─────────────────────────────────────────────────────────────
# 6. GENERACIÓN DE RECOMENDACIONES (mes siguiente al último)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. GENERACIÓN DE RECOMENDACIONES")
print("=" * 60)

# Usamos el conjunto del último mes como contexto para predecir el mes siguiente
df_recomendar = grid_df[grid_df["Mes"] == ultimo_mes].copy()

df_recomendar["prob_compra"] = mejor_modelo.predict_proba(df_recomendar[FEATURES])[:, 1]

# Excluir productos ya comprados en el último mes
df_nuevos = df_recomendar[df_recomendar["compro"] == 0].copy()

# Top-3 por cliente ordenado por probabilidad descendente
recomendaciones = (
    df_nuevos.sort_values(["Cliente", "prob_compra"], ascending=[True, False])
    .groupby("Cliente")
    .head(3)
    .reset_index(drop=True)
)
recomendaciones["ranking"] = recomendaciones.groupby("Cliente").cumcount() + 1

# Enriquecer con datos del cliente
recomendaciones = recomendaciones.merge(
    clientes[["Cliente", "Oficina de ventas",
              "Denominación Gr#Clientes 4", "Denominación Gr#Clientes 5"]],
    on="Cliente", how="left"
)

print(f"Recomendaciones para {recomendaciones['Cliente'].nunique():,} clientes")

primeros5 = recomendaciones["Cliente"].unique()[:5]
cols = ["Cliente", "ranking", "CodProducto", "prob_compra",
        "Oficina de ventas", "Denominación Gr#Clientes 4"]
print("\nEjemplo Top-3 para los primeros 5 clientes:")
print(recomendaciones[recomendaciones["Cliente"].isin(primeros5)][cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. GUARDAR RESULTADOS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GUARDANDO RESULTADOS")
print("=" * 60)

recomendaciones.to_csv("recomendaciones_top3.csv", index=False)
df_recomendar[["Cliente", "CodProducto", "prob_compra", "compro"]].to_csv(
    "probabilidades_completas.csv", index=False
)
df_comp.to_csv("comparativa_modelos.csv")

print("Archivos guardados:")
print("  recomendaciones_top3.csv      → Top-3 recomendaciones por cliente")
print("  probabilidades_completas.csv  → Probabilidades para todos los pares")
print("  comparativa_modelos.csv       → Métricas de los 3 modelos")
print("\n¡Proceso completado exitosamente!")
