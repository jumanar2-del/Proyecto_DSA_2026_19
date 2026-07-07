"""
Tablero de Recomendación de Productos
Proyecto DSA 2026 | Jessica Salazar · Juan Camilo Umaña

Dos vistas (según la maqueta del prototipo):
  Vista 1 — Por SKU   : qué clientes tienen mayor oportunidad de compra
  Vista 2 — Por Cliente: qué productos se recomienda ofrecer a ese cliente

Ejecutar:
    streamlit run 03_tablero.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Recomendador de Productos",
    page_icon="🛒",
    layout="wide",
)

# ── Estilos ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")

# ── Carga de datos ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def cargar_datos():
    recom = pd.read_csv(os.path.join(BASE_DIR, "recomendaciones_top3.csv"))
    probs = pd.read_csv(os.path.join(BASE_DIR, "probabilidades_completas.csv"))
    clientes = pd.read_csv(os.path.join(BASE_DIR, "Clientes.csv"),
                           encoding="latin1", sep=";")
    clientes.rename(columns={"Deudor": "Cliente"}, inplace=True)
    resumen = pd.read_csv(os.path.join(BASE_DIR, "Resumen.csv"),
                          encoding="latin1", sep=None, engine="python")
    for col in ["Venta", "KG", "Ton"]:
        resumen[col] = (
            resumen[col].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    return recom, probs, clientes, resumen

recom, probs, clientes_df, resumen_df = cargar_datos()

ultimo_mes = resumen_df["Mes"].max()
productos_lista = sorted(probs["CodProducto"].unique())
clientes_lista  = sorted(probs["Cliente"].unique())

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shop.png", width=70
)
st.sidebar.title("Recomendador de Productos")
st.sidebar.caption("DSA 2026 · Consumo Masivo")

vista = st.sidebar.radio(
    "Selecciona la vista:",
    ["🔍 Por SKU — ¿A quién ofrecer este producto?",
     "👤 Por Cliente — ¿Qué ofrecerle a este cliente?"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Datos al mes:** `{ultimo_mes}`  \n"
    f"**Clientes activos:** {probs['Cliente'].nunique():,}  \n"
    f"**Productos en portafolio:** {len(productos_lista)}"
)

# ════════════════════════════════════════════════════════════
# VISTA 1 — Por SKU
# ════════════════════════════════════════════════════════════
if vista.startswith("🔍"):

    st.title("🔍 Vista por SKU")
    st.markdown("Selecciona un producto para ver qué clientes tienen mayor probabilidad de comprarlo.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sku_sel = st.selectbox("Producto (SKU)", productos_lista)
    with col_f2:
        oficinas = ["Todas"] + sorted(clientes_df["Oficina de ventas"].unique())
        oficina_sel = st.selectbox("Oficina de ventas", oficinas)
    with col_f3:
        segmentos = ["Todos"] + sorted(clientes_df["Denominación Gr#Clientes 4"].unique())
        segmento_sel = st.selectbox("Segmento (Gr4)", segmentos)

    # Filtrar probabilidades para el SKU seleccionado
    df_sku = probs[probs["CodProducto"] == sku_sel].copy()
    df_sku = df_sku.merge(
        clientes_df[["Cliente", "Oficina de ventas",
                     "Denominación Gr#Clientes 4", "Denominación Gr#Clientes 5",
                     "Frecuencia de visita", "Población"]],
        on="Cliente", how="left"
    )

    # Calcular última compra
    ultima_compra = (
        resumen_df[resumen_df["CodProducto"] == sku_sel]
        .groupby("Cliente")["Mes"].max()
        .reset_index()
        .rename(columns={"Mes": "ultima_compra"})
    )
    df_sku = df_sku.merge(ultima_compra, on="Cliente", how="left")

    # Aplicar filtros
    if oficina_sel != "Todas":
        df_sku = df_sku[df_sku["Oficina de ventas"] == oficina_sel]
    if segmento_sel != "Todos":
        df_sku = df_sku[df_sku["Denominación Gr#Clientes 4"] == segmento_sel]

    # Excluir clientes que ya compraron en el último mes
    df_nuevos = df_sku[df_sku["compro"] == 0].sort_values("prob_compra", ascending=False)

    # ── KPIs ────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes con oportunidad", f"{len(df_nuevos):,}")
    k2.metric("Probabilidad promedio", f"{df_nuevos['prob_compra'].mean():.1%}")
    k3.metric("Clientes con prob > 30%", f"{(df_nuevos['prob_compra'] > 0.3).sum():,}")
    venta_hist = resumen_df[resumen_df["CodProducto"] == sku_sel]["Venta"].sum()
    k4.metric("Venta histórica total", f"${venta_hist:,.0f}")

    # ── Tabla de clientes recomendados ───────────────────────
    st.subheader(f"Clientes recomendados para {sku_sel}")
    top_n = st.slider("Número de clientes a mostrar", 10, 200, 50, step=10)

    tabla = df_nuevos.head(top_n)[[
        "Cliente", "prob_compra", "ultima_compra",
        "Oficina de ventas", "Denominación Gr#Clientes 4",
        "Frecuencia de visita", "Población"
    ]].copy()
    tabla["prob_compra"] = tabla["prob_compra"].apply(lambda x: f"{x:.1%}")
    tabla["Prioridad"] = ["🔴 Alta" if i < top_n * 0.2
                          else "🟡 Media" if i < top_n * 0.5
                          else "🟢 Normal"
                          for i in range(len(tabla))]
    tabla.columns = ["Cliente", "Prob. Compra", "Última Compra",
                     "Oficina", "Segmento", "Frec. Visita", "Población", "Prioridad"]

    st.dataframe(tabla, use_container_width=True, height=350)

    # ── Gráficas ────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Distribución de probabilidades")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df_nuevos["prob_compra"], bins=20, color="steelblue", edgecolor="white")
        ax.axvline(df_nuevos["prob_compra"].mean(), color="red", linestyle="--",
                   label=f"Promedio: {df_nuevos['prob_compra'].mean():.1%}")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("Probabilidad de compra")
        ax.set_ylabel("Clientes")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.subheader("Top 10 clientes — mayor oportunidad")
        top10 = df_nuevos.head(10).copy()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(
            top10["Cliente"].astype(str),
            top10["prob_compra"],
            color="steelblue"
        )
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("Probabilidad de compra")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Descarga
    csv = df_nuevos.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar lista completa (CSV)",
        csv,
        file_name=f"oportunidad_{sku_sel}.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════════════════════════
# VISTA 2 — Por Cliente
# ════════════════════════════════════════════════════════════
else:
    st.title("👤 Vista por Cliente")
    st.markdown("Selecciona un cliente para ver sus productos recomendados.")

    cliente_sel = st.selectbox(
        "Código de cliente",
        clientes_lista,
        format_func=lambda x: f"{x}"
    )

    # Info del cliente
    info_cliente = clientes_df[clientes_df["Cliente"] == cliente_sel]

    if not info_cliente.empty:
        info = info_cliente.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Oficina de ventas", info["Oficina de ventas"])
        c2.metric("Segmento", info["Denominación Gr#Clientes 4"])
        c3.metric("Canal", info["Denominación Gr#Clientes 5"])
        c4.metric("Frec. visita", info["Frecuencia de visita"])

    st.markdown("---")

    # Historial del cliente
    hist = resumen_df[resumen_df["Cliente"] == cliente_sel]
    skus_activos = hist[hist["Venta"] > 0]["CodProducto"].unique()
    ultima_compra_gral = hist["Mes"].max() if len(hist) > 0 else "Sin compras"

    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.info(f"**Última compra:** {ultima_compra_gral}")
    col_i2.info(f"**SKUs activos:** {', '.join(skus_activos) if len(skus_activos) > 0 else 'Ninguno'}")
    ventas_tot = hist["Venta"].sum()
    col_i3.info(f"**Venta histórica total:** ${ventas_tot:,.0f}")

    # Recomendaciones Top-3
    st.subheader("Productos recomendados")
    rec_cliente = recom[recom["Cliente"] == cliente_sel].sort_values("prob_compra", ascending=False)

    if rec_cliente.empty:
        st.warning("No hay recomendaciones disponibles para este cliente.")
    else:
        for _, row in rec_cliente.iterrows():
            razon = (
                "Alta frecuencia histórica de compra de productos similares."
                if row["prob_compra"] > 0.3
                else "Perfil comercial compatible con compradores activos de este SKU."
            )
            with st.container():
                r1, r2, r3 = st.columns([1, 2, 4])
                r1.metric(f"#{int(row['ranking'])} — {row['CodProducto']}",
                          f"{row['prob_compra']:.1%}", delta="prob. compra")
                r2.markdown(f"**Razón:** {razon}")
                r3.progress(min(row["prob_compra"], 1.0))
            st.markdown("")

    # Todas las probabilidades del cliente
    st.subheader("Probabilidad de compra por producto")
    probs_cliente = probs[probs["Cliente"] == cliente_sel].sort_values(
        "prob_compra", ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#2196F3" if p in skus_activos else "#90CAF9"
              for p in probs_cliente["CodProducto"]]
    ax.bar(probs_cliente["CodProducto"], probs_cliente["prob_compra"], color=colors)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Producto")
    ax.set_ylabel("Probabilidad de compra")
    ax.set_title(f"Probabilidades para cliente {cliente_sel}", fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Compra actual"),
        Patch(facecolor="#90CAF9", label="No compra actualmente"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Historial de compra por mes
    if len(hist) > 0:
        st.subheader("Historial de compras por mes")
        hist_pivot = hist.groupby(["Mes", "CodProducto"])["Venta"].sum().reset_index()
        hist_wide  = hist_pivot.pivot(index="Mes", columns="CodProducto", values="Venta").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 3))
        hist_wide.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"Ventas por producto — cliente {cliente_sel}", fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Venta ($)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Producto", fontsize=7, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Proyecto DSA 2026 · Sistema de Recomendación de Productos · Jessica Salazar · Juan Camilo Umaña")
