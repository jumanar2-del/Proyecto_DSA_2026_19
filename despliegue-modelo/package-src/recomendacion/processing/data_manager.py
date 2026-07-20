import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from recomendacion._version import __version__ as _version
from recomendacion.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un CSV desde la carpeta datasets del paquete."""
    file_path = Path(f"{DATASET_DIR}/{file_name}")

    if file_name == config.app_config.train_data_file:
        # Clientes: separador punto y coma, encoding latin1
        dataframe = pd.read_csv(file_path, encoding="latin1", sep=";")
        dataframe.rename(columns={"Deudor": "Cliente"}, inplace=True)
    else:
        # Resumen: separador automático
        dataframe = pd.read_csv(file_path, encoding="latin1", sep=None, engine="python")
        for col in ["Venta", "KG", "Ton"]:
            if col in dataframe.columns:
                dataframe[col] = (
                    dataframe[col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persiste el pipeline entrenado con versión en el nombre del archivo.

    Sobreescribe modelos anteriores para garantizar un único modelo por versión.
    """
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print(f"Modelo guardado en: {save_path}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Carga un pipeline persistido."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """Elimina versiones antiguas del modelo entrenado."""
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
