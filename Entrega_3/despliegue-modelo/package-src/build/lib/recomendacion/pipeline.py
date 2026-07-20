"""
Define los tres pipelines candidatos del modelo de recomendación:
  - LogisticRegression
  - RandomForest
  - GradientBoosting

El pipeline ganador se selecciona en train_pipeline.py comparando ROC-AUC
sobre el último mes (split temporal), replicando la lógica del script original.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from recomendacion.config.core import config

_cfg = config.model_config

# ── Pipeline 1: Logistic Regression ─────────────────────────────────────────
logistic_pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=_cfg.random_state,
            ),
        ),
    ]
)

# ── Pipeline 2: Random Forest ────────────────────────────────────────────────
random_forest_pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=_cfg.n_estimators,
                max_depth=_cfg.max_depth,
                class_weight="balanced",
                random_state=_cfg.random_state,
                n_jobs=-1,
            ),
        ),
    ]
)

# ── Pipeline 3: Gradient Boosting ────────────────────────────────────────────
gradient_boosting_pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=_cfg.gb_n_estimators,
                max_depth=_cfg.gb_max_depth,
                learning_rate=_cfg.gb_learning_rate,
                random_state=_cfg.random_state,
            ),
        ),
    ]
)

# Diccionario de candidatos para iterar en train_pipeline.py
CANDIDATE_PIPELINES = {
    "LogisticRegression": logistic_pipe,
    "RandomForest": random_forest_pipe,
    "GradientBoosting": gradient_boosting_pipe,
}
