"""
Entrena los tres modelos candidatos (LogisticRegression, RandomForest,
GradientBoosting), los compara por ROC-AUC en el último mes (split temporal)
y persiste el mejor pipeline. Replica la lógica del script original
modelo_recomendacion.py.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from recomendacion.config.core import config
from recomendacion.pipeline import CANDIDATE_PIPELINES
from recomendacion.processing.data_manager import load_dataset, save_pipeline
from recomendacion.processing.features import (
    agregar_features_temporales,
    codificar_clientes,
    construir_grid,
)

import os


def run_training() -> None:
    """Entrena, compara y persiste el mejor pipeline de recomendación."""

    cfg = config.model_config

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print("=" * 60)
    print("1. CARGA DE DATOS")
    print("=" * 60)
    clientes = load_dataset(file_name=config.app_config.train_data_file)
    resumen = load_dataset(file_name="Resumen.csv")
    print(f"Clientes : {clientes.shape[0]:,} filas | Resumen: {resumen.shape[0]:,} filas")

    # ── 2. Construir features ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. INGENIERÍA DE CARACTERÍSTICAS")
    print("=" * 60)
    grid_df, meses, resumen_procesado = construir_grid(resumen)
    grid_df = agregar_features_temporales(grid_df, meses, resumen_procesado)
    grid_df = codificar_clientes(grid_df, clientes)
    print(f"Dataset construido: {grid_df.shape[0]:,} filas")
    print(f"Tasa de positivos : {grid_df[cfg.target_col].mean():.2%}")

    # ── 3. Split temporal ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. DIVISIÓN TRAIN / TEST (split temporal: último mes = test)")
    print("=" * 60)
    meses_ordenados = sorted(resumen[cfg.month_col].unique())
    df_model = grid_df[grid_df[cfg.month_col] >= meses_ordenados[1]].copy()
    ultimo_mes = meses_ordenados[-1]
    mask_test = df_model[cfg.month_col] == ultimo_mes

    X_train = df_model.loc[~mask_test, cfg.features]
    X_test  = df_model.loc[ mask_test, cfg.features]
    y_train = df_model.loc[~mask_test, cfg.target_col]
    y_test  = df_model.loc[ mask_test, cfg.target_col]

    print(f"Train : {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")
    print(f"Positivos train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # ── 4. Experimentos con MLflow ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. EXPERIMENTOS CON MLFLOW")
    print("=" * 60)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Recomendacion_Productos_DSA2026")

    resultados = {}

    for nombre, pipe in CANDIDATE_PIPELINES.items():
        print(f"\nEntrenando {nombre}...")

        with mlflow.start_run(run_name=nombre):
            mlflow.log_params(pipe.steps[-1][1].get_params())
            mlflow.log_param("features",   str(cfg.features))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size",  len(X_test))
            mlflow.log_param("mes_test",   str(ultimo_mes))

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metricas = {
                "roc_auc"  : roc_auc_score(y_test, y_prob),
                "pr_auc"   : average_precision_score(y_test, y_prob),
                "f1"       : f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall"   : recall_score(y_test, y_pred, zero_division=0),
            }

            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(pipe, artifact_path=nombre)

            resultados[nombre] = {"pipeline": pipe, **metricas}

            print(f"  ROC-AUC  : {metricas['roc_auc']:.4f}")
            print(f"  PR-AUC   : {metricas['pr_auc']:.4f}")
            print(f"  F1       : {metricas['f1']:.4f}")
            print(f"  Precision: {metricas['precision']:.4f}")
            print(f"  Recall   : {metricas['recall']:.4f}")

    # ── 5. Seleccionar mejor modelo ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. SELECCIÓN DEL MEJOR MODELO")
    print("=" * 60)

    mejor_nombre = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    mejor_pipeline = resultados[mejor_nombre]["pipeline"]

    print("\nComparativa de modelos:")
    for nombre, vals in resultados.items():
        print(
            f"  {nombre:<22} ROC-AUC={vals['roc_auc']:.4f}  "
            f"F1={vals['f1']:.4f}  PR-AUC={vals['pr_auc']:.4f}"
        )
    print(f"\nMejor modelo: {mejor_nombre} → ROC-AUC={resultados[mejor_nombre]['roc_auc']:.4f}")

    # ── 6. Persistir el mejor pipeline ───────────────────────────────────────
    save_pipeline(pipeline_to_persist=mejor_pipeline)
    print(f"Pipeline '{mejor_nombre}' guardado correctamente.")


if __name__ == "__main__":
    run_training()
