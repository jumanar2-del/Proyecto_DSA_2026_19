from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import recomendacion

# Directorios del proyecto
PACKAGE_ROOT = Path(recomendacion.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


class AppConfig(BaseModel):
    """Configuración a nivel de aplicación."""

    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """Configuración del modelo y feature engineering."""

    client_col: str
    product_col: str
    month_col: str
    sales_col: str
    target_col: str
    features: List[str]
    categorical_vars: List[str]
    test_size: float
    random_state: int
    # RandomForest
    n_estimators: int
    max_depth: int
    # GradientBoosting
    gb_n_estimators: int
    gb_max_depth: int
    gb_learning_rate: float
    # Selección
    best_model_metric: str
    top_n: int


class Config(BaseModel):
    """Objeto maestro de configuración."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Localiza el archivo de configuración."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config no encontrado en {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parsea el YAML de configuración del paquete."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r", encoding="utf-8") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"No se encontró el archivo de config en: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Valida los valores del archivo de configuración."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )
    return _config


config = create_and_validate_config()
