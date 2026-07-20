import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from recomendacion.config.core import config


def construir_grid(resumen: pd.DataFrame) -> pd.DataFrame:
    """Construye el grid completo (Mes x Cliente x CodProducto) con compra binaria."""
    cfg = config.model_config
    mes_col = cfg.month_col
    cliente_col = cfg.client_col
    producto_col = cfg.product_col
    ventas_col = cfg.sales_col

    meses = sorted(resumen[mes_col].unique())
    productos = sorted(resumen[producto_col].unique())
    todos_clientes = resumen[cliente_col].unique()

    # Compra binaria por (Mes, Cliente, CodProducto)
    resumen = resumen.copy()
    resumen["compro"] = (resumen[ventas_col] > 0).astype(int)

    pivot = (
        resumen.groupby([mes_col, cliente_col, producto_col])["compro"]
        .max()
        .reset_index()
    )

    grid = pd.MultiIndex.from_product(
        [meses, todos_clientes, productos],
        names=[mes_col, cliente_col, producto_col],
    )
    grid_df = pd.DataFrame(index=grid).reset_index()
    grid_df = grid_df.merge(pivot, on=[mes_col, cliente_col, producto_col], how="left")
    grid_df["compro"] = grid_df["compro"].fillna(0).astype(int)
    grid_df = grid_df.sort_values([cliente_col, producto_col, mes_col]).reset_index(drop=True)

    return grid_df, meses, resumen


def agregar_features_temporales(grid_df: pd.DataFrame, meses: list, resumen: pd.DataFrame) -> pd.DataFrame:
    """Agrega features históricas por (Cliente, CodProducto, Mes)."""
    cfg = config.model_config
    mes_col = cfg.month_col
    cliente_col = cfg.client_col
    producto_col = cfg.product_col
    ventas_col = cfg.sales_col

    # compro_mes_ant
    grid_df["compro_mes_ant"] = (
        grid_df.groupby([cliente_col, producto_col])["compro"]
        .shift(1).fillna(0).astype(int)
    )

    # veces_comprado (acumulado histórico, excluyendo mes actual)
    grid_df["veces_comprado"] = (
        grid_df.groupby([cliente_col, producto_col])["compro"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )

    # mes_idx ordinal
    mes_idx_map = {m: i + 1 for i, m in enumerate(meses)}
    grid_df["mes_idx"] = grid_df[mes_col].map(mes_idx_map)

    # freq_compra
    grid_df["freq_compra"] = grid_df["veces_comprado"] / grid_df["mes_idx"].clip(lower=1)

    # ventas_cliente_ant
    ventas_mes = (
        resumen.groupby([mes_col, cliente_col])[ventas_col]
        .sum()
        .reset_index()
        .rename(columns={ventas_col: "ventas_cliente_ant", mes_col: "Mes_ant"})
    )
    grid_df["Mes_ant"] = grid_df.groupby([cliente_col, producto_col])[mes_col].shift(1)
    grid_df = grid_df.merge(ventas_mes, on=["Mes_ant", cliente_col], how="left")
    grid_df["ventas_cliente_ant"] = grid_df["ventas_cliente_ant"].fillna(0)

    # productos_ant
    prods_mes = (
        resumen[resumen[ventas_col] > 0]
        .groupby([mes_col, cliente_col])[producto_col]
        .nunique()
        .reset_index()
        .rename(columns={producto_col: "productos_ant", mes_col: "Mes_ant"})
    )
    grid_df = grid_df.merge(prods_mes, on=["Mes_ant", cliente_col], how="left")
    grid_df["productos_ant"] = grid_df["productos_ant"].fillna(0)

    grid_df["target"] = grid_df["compro"]

    return grid_df


def codificar_clientes(grid_df: pd.DataFrame, clientes: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas de clientes y productos."""
    cfg = config.model_config
    cliente_col = cfg.client_col
    producto_col = cfg.product_col

    le_oficina = LabelEncoder()
    le_seg5 = LabelEncoder()
    le_seg4 = LabelEncoder()
    le_producto = LabelEncoder()

    clientes = clientes.copy()
    clientes["oficina_enc"] = le_oficina.fit_transform(clientes["Oficina de ventas"])
    clientes["seg5_enc"] = le_seg5.fit_transform(clientes["Denominación Gr#Clientes 5"])
    clientes["seg4_enc"] = le_seg4.fit_transform(clientes["Denominación Gr#Clientes 4"])

    grid_df = grid_df.merge(
        clientes[[cliente_col, "Frecuencia de visita", "oficina_enc", "seg5_enc", "seg4_enc"]],
        on=cliente_col,
        how="left",
    )
    grid_df["producto_enc"] = le_producto.fit_transform(grid_df[producto_col])
    grid_df.fillna(0, inplace=True)

    return grid_df
