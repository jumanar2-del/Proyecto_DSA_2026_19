from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

BASE_DIR = Path(__file__).resolve().parent
TOP3_PATH = BASE_DIR / "data" / "recomendaciones_top3.csv"
FULL_PATH = BASE_DIR / "data" / "probabilidades_completas.csv"

COLOR_HEADER = "#141b2a"
COLOR_TEXT = "#172033"
COLOR_MUTED = "#6b7280"
COLOR_TEAL = "#139b92"
COLOR_ORANGE = "#f2a900"
COLOR_RED = "#e53935"
COLOR_ROW = "#c7d1e2"
COLOR_ROW_ALT = "#edf1f7"


def prioridad_desde_probabilidad(probabilidad: float) -> str:
    if pd.isna(probabilidad):
        return "Sin datos"
    if probabilidad >= 0.70:
        return "Alta"
    if probabilidad >= 0.40:
        return "Media"
    return "Baja"


def razon_recomendacion(row: pd.Series) -> str:
    """Genera una razón simple y explicable a partir de las variables disponibles."""
    if row.get("compro_mes_ant", 0) == 1:
        return "Compra reciente"
    if row.get("veces_comprado", 0) > 0:
        return "Histórico"
    if row.get("freq_compra", 0) > 0:
        return "Frecuencia"
    return "Modelo"


def formatear_periodo(valor) -> str:
    try:
        valor = int(float(valor))
        anio = str(valor)[:4]
        mes = str(valor)[4:]
        meses = {
            "01": "Enero", "02": "Febrero", "03": "Marzo", "04": "Abril",
            "05": "Mayo", "06": "Junio", "07": "Julio", "08": "Agosto",
            "09": "Septiembre", "10": "Octubre", "11": "Noviembre", "12": "Diciembre",
        }
        return f"{meses.get(mes, mes)} {anio}"
    except Exception:
        return str(valor)


def porcentaje_texto(probabilidad: float) -> str:
    if pd.isna(probabilidad):
        return "0%"
    return f"{probabilidad * 100:.0f}%"


def load_data():
    top3 = pd.read_csv(TOP3_PATH)
    completas = pd.read_csv(FULL_PATH)

    # Estandarización de nombres para el tablero
    top3 = top3.rename(
        columns={
            "CodProducto": "SKU",
            "prob_compra": "Probabilidad",
            "Oficina de ventas": "Oficina",
            "Denominación Gr#Clientes 5": "Canal",
            "Denominación Gr#Clientes 4": "Segmento",
        }
    )
    completas = completas.rename(
        columns={
            "CodProducto": "SKU",
            "prob_compra": "Probabilidad",
        }
    )

    top3["Cliente"] = top3["Cliente"].astype(str)
    top3["SKU"] = top3["SKU"].astype(str)
    completas["Cliente"] = completas["Cliente"].astype(str)
    completas["SKU"] = completas["SKU"].astype(str)

    for data in [top3, completas]:
        data["Probabilidad"] = pd.to_numeric(data["Probabilidad"], errors="coerce")
        data["Probabilidad_texto"] = data["Probabilidad"].apply(porcentaje_texto)
        data["Prioridad"] = data["Probabilidad"].apply(prioridad_desde_probabilidad)

    # Campos usados en la visual por cliente
    top3["Producto"] = top3["SKU"]
    top3["Categoria"] = top3["Segmento"].fillna("Sin categoría")
    top3["Razon"] = top3.apply(razon_recomendacion, axis=1)
    top3["Periodo"] = top3["Mes"].apply(formatear_periodo)
    top3["Compra_mes_anterior"] = top3["compro_mes_ant"].map({1: "Sí", 0: "No"}).fillna("No")
    top3["Comprado_antes"] = top3["compro"].map({1: "Sí", 0: "No"}).fillna("No")

    # Metadata de cliente para enriquecer la visual por SKU
    meta_cols = [
        "Cliente", "Oficina", "Canal", "Segmento", "Frecuencia de visita",
        "productos_ant", "ventas_cliente_ant", "Periodo"
    ]
    meta_cliente = (
        top3[meta_cols]
        .drop_duplicates(subset=["Cliente"])
        .copy()
    )

    sku_view = completas.merge(meta_cliente, on="Cliente", how="left")
    sku_view["Oficina"] = sku_view["Oficina"].fillna("Sin info")
    sku_view["Canal"] = sku_view["Canal"].fillna("Sin info")
    sku_view["Segmento"] = sku_view["Segmento"].fillna("Sin info")
    sku_view["Periodo"] = sku_view["Periodo"].fillna(formatear_periodo(top3["Mes"].iloc[0]) if not top3.empty else "Sin periodo")
    sku_view["Comprado_antes"] = sku_view["compro"].map({1: "Sí", 0: "No"}).fillna("No")
    sku_view["Producto"] = sku_view["SKU"]

    return top3, sku_view


df_top3, df_sku = load_data()

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Recomendador de clientes"


def build_options(series, include_all=False, all_label="Todos"):
    values = sorted(series.dropna().astype(str).unique())
    options = [{"label": value, "value": value} for value in values]
    if include_all:
        return [{"label": all_label, "value": all_label}] + options
    return options


def empty_figure(title="Sin datos"):
    fig = px.bar(title=title)
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


TABLE_STYLE = {
    "style_table": {"overflowX": "auto", "border": "1px solid #dbe2ec", "borderRadius": "4px"},
    "style_cell": {
        "textAlign": "left",
        "fontFamily": "Arial, sans-serif",
        "fontSize": "15px",
        "padding": "14px 12px",
        "border": "none",
        "color": COLOR_TEXT,
        "whiteSpace": "normal",
        "height": "auto",
    },
    "style_header": {
        "backgroundColor": "#f1f4f9",
        "fontWeight": "700",
        "fontSize": "15px",
        "color": "#4b5563",
        "borderBottom": "2px solid #dbe2ec",
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": COLOR_ROW_ALT},
        {"if": {"row_index": "even"}, "backgroundColor": COLOR_ROW},
        {"if": {"state": "selected"}, "backgroundColor": "#d9e2f3", "border": "1px solid #8ba3c7"},
        {"if": {"column_id": "Probabilidad_texto", "filter_query": "{Prioridad} = 'Alta'"}, "color": COLOR_TEAL, "fontWeight": "700"},
        {"if": {"column_id": "Probabilidad_texto", "filter_query": "{Prioridad} = 'Media'"}, "color": COLOR_ORANGE, "fontWeight": "700"},
        {"if": {"column_id": "Prioridad", "filter_query": "{Prioridad} = 'Alta'"}, "fontWeight": "700"},
    ],
}


def kpi_card(label, component_id=None, value=None, class_name=""):
    return html.Div(
        className="card",
        children=[
            html.P(label),
            html.H3(value if value is not None else "", id=component_id, className=class_name),
        ],
    )


def visual_por_sku():
    default_sku = sorted(df_sku["SKU"].dropna().astype(str).unique())[0]
    return html.Div(
        className="visual-block",
        children=[
            html.Div(className="title-block", children=[html.H1("Recomendador de Clientes por SKU")]),
            html.Div(
                className="filters-bar",
                children=[
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Label("SKU / PRODUCTO"),
                            dcc.Dropdown(
                                id="sku-filtro-sku",
                                options=build_options(df_sku["SKU"]),
                                value=default_sku,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Label("CANAL"),
                            dcc.Dropdown(
                                id="sku-filtro-canal",
                                options=build_options(df_sku["Canal"], include_all=True, all_label="Todos"),
                                value="Todos",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Label("OFICINA"),
                            dcc.Dropdown(
                                id="sku-filtro-oficina",
                                options=build_options(df_sku["Oficina"], include_all=True, all_label="Todas"),
                                value="Todas",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Label("ESTADO"),
                            dcc.Dropdown(
                                id="sku-filtro-comprado",
                                options=[
                                    {"label": "Solo oportunidad", "value": "No"},
                                    {"label": "Ya compró", "value": "Sí"},
                                    {"label": "Todos", "value": "Todos"},
                                ],
                                value="No",
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="cards",
                children=[
                    kpi_card("Clientes con oportunidad", "sku-kpi-clientes"),
                    kpi_card("Prob. promedio", "sku-kpi-prob"),
                    kpi_card("No han comprado", "sku-kpi-no-comprados"),
                    kpi_card("Prioridad", "sku-kpi-prioridad", class_name="priority-text"),
                ],
            ),
            html.Div(
                className="section-simple",
                children=[
                    html.H2("Clientes con mayor oportunidad para el SKU seleccionado"),
                    dash_table.DataTable(
                        id="tabla-por-sku",
                        columns=[
                            {"name": "Cliente", "id": "Cliente"},
                            {"name": "Canal", "id": "Canal"},
                            {"name": "Oficina", "id": "Oficina"},
                            {"name": "Compró antes", "id": "Comprado_antes"},
                            {"name": "Prob. compra", "id": "Probabilidad_texto"},
                            {"name": "Prioridad", "id": "Prioridad"},
                        ],
                        data=[],
                        page_size=8,
                        sort_action="native",
                        **TABLE_STYLE,
                    ),
                ],
            ),
        ],
    )


def visual_por_cliente():
    default_cliente = sorted(df_top3["Cliente"].dropna().astype(str).unique())[0]
    return html.Div(
        className="visual-block",
        children=[
            html.Div(className="title-block", children=[html.H1("Detalle del cliente: Top SKUs recomendados")]),
            html.Div(
                className="filters-bar detail-filters",
                children=[
                    html.Div(
                        className="filter-item",
                        children=[
                            html.Label("CLIENTE"),
                            dcc.Dropdown(
                                id="cliente-filtro-cliente",
                                options=build_options(df_top3["Cliente"]),
                                value=default_cliente,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(className="filter-item", children=[html.Label("CANAL"), html.Div(id="cliente-canal", className="fake-select")]),
                    html.Div(className="filter-item", children=[html.Label("OFICINA"), html.Div(id="cliente-oficina", className="fake-select")]),
                    html.Div(className="filter-item", children=[html.Label("PERIODO"), html.Div(id="cliente-periodo", className="fake-select")]),
                ],
            ),
            html.Div(
                className="cards detail-cards",
                children=[
                    kpi_card("SKUs recomendados", "cliente-kpi-recomendados"),
                    kpi_card("SKUs actuales", "cliente-kpi-actuales"),
                    kpi_card("Prob. máxima", "cliente-kpi-prob"),
                    kpi_card("Oportunidad", "cliente-kpi-oportunidad", class_name="priority-text"),
                ],
            ),
            html.Div(
                className="detail-grid",
                children=[
                    html.Div(
                        className="section-simple",
                        children=[
                            html.H2("Top SKUs para ofrecer al cliente"),
                            dash_table.DataTable(
                                id="tabla-por-cliente",
                                columns=[
                                    {"name": "Rank", "id": "ranking"},
                                    {"name": "SKU", "id": "SKU"},
                                    {"name": "Categoría", "id": "Categoria"},
                                    {"name": "Prob.", "id": "Probabilidad_texto"},
                                    {"name": "Razón", "id": "Razon"},
                                ],
                                data=[],
                                page_size=5,
                                sort_action="native",
                                **TABLE_STYLE,
                            ),
                        ],
                    ),
                    html.Div(
                        className="chart-card",
                        children=[
                            html.Div("▮  Probabilidad por SKU", className="chart-title"),
                            dcc.Graph(id="grafico-por-cliente", config={"displayModeBar": False}),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = html.Div(
    className="page",
    children=[
        dcc.Tabs(
            id="tabs",
            value="tab-sku",
            className="tabs",
            children=[
                dcc.Tab(label="Visual por SKU", value="tab-sku", className="tab", selected_className="tab-selected"),
                dcc.Tab(label="Visual por cliente", value="tab-cliente", className="tab", selected_className="tab-selected"),
            ],
        ),
        html.Div(id="tab-content"),
    ],
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-cliente":
        return visual_por_cliente()
    return visual_por_sku()


@app.callback(
    Output("tabla-por-sku", "data"),
    Output("sku-kpi-clientes", "children"),
    Output("sku-kpi-prob", "children"),
    Output("sku-kpi-no-comprados", "children"),
    Output("sku-kpi-prioridad", "children"),
    Input("sku-filtro-sku", "value"),
    Input("sku-filtro-canal", "value"),
    Input("sku-filtro-oficina", "value"),
    Input("sku-filtro-comprado", "value"),
    prevent_initial_call=False,
)
def actualizar_visual_sku(sku, canal, oficina, comprado):
    df = df_sku.copy()
    df = df[df["SKU"].astype(str) == str(sku)]

    if canal != "Todos":
        df = df[df["Canal"].astype(str) == str(canal)]
    if oficina != "Todas":
        df = df[df["Oficina"].astype(str) == str(oficina)]
    if comprado != "Todos":
        df = df[df["Comprado_antes"].astype(str) == str(comprado)]

    df = df.sort_values("Probabilidad", ascending=False).head(50)

    table_data = df[[
        "Cliente", "Canal", "Oficina", "Comprado_antes",
        "Probabilidad", "Probabilidad_texto", "Prioridad",
    ]].to_dict("records")

    n_clientes = df["Cliente"].nunique()
    prob_prom = "0%" if df.empty else porcentaje_texto(df["Probabilidad"].mean())
    no_comprados = str((df["Comprado_antes"] == "No").sum())
    prioridad = "Sin datos" if df.empty else prioridad_desde_probabilidad(df["Probabilidad"].mean())

    return table_data, str(n_clientes), prob_prom, no_comprados, prioridad


@app.callback(
    Output("cliente-canal", "children"),
    Output("cliente-oficina", "children"),
    Output("cliente-periodo", "children"),
    Output("cliente-kpi-recomendados", "children"),
    Output("cliente-kpi-actuales", "children"),
    Output("cliente-kpi-prob", "children"),
    Output("cliente-kpi-oportunidad", "children"),
    Output("tabla-por-cliente", "data"),
    Output("grafico-por-cliente", "figure"),
    Input("cliente-filtro-cliente", "value"),
    prevent_initial_call=False,
)
def actualizar_visual_cliente(cliente):
    df = df_top3[df_top3["Cliente"].astype(str) == str(cliente)].copy()

    if df.empty:
        return "-", "-", "-", "0", "0", "0%", "Sin datos", [], empty_figure("Sin datos")

    df = df.sort_values("ranking")
    canal = str(df.iloc[0]["Canal"])
    oficina = str(df.iloc[0]["Oficina"])
    periodo = str(df.iloc[0]["Periodo"])
    skus_recomendados = str(df["SKU"].nunique())
    skus_actuales = int(df["productos_ant"].max()) if pd.notna(df["productos_ant"].max()) else 0
    prob_max = df["Probabilidad"].max()
    oportunidad = prioridad_desde_probabilidad(prob_max)

    table_data = df[[
        "ranking", "SKU", "Categoria", "Probabilidad", "Probabilidad_texto", "Razon", "Prioridad"
    ]].to_dict("records")

    df_graph = df.sort_values("Probabilidad", ascending=True).copy()
    fig = px.bar(
        df_graph,
        x="Probabilidad",
        y="SKU",
        orientation="h",
        text="Probabilidad_texto",
        range_x=[0, max(1, df_graph["Probabilidad"].max() + 0.08)],
    )
    fig.update_traces(marker_color=COLOR_TEAL, textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        height=310,
        margin={"l": 70, "r": 45, "t": 10, "b": 10},
        xaxis={"visible": False},
        yaxis={"title": None},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial", "size": 13, "color": COLOR_TEXT},
        showlegend=False,
    )

    return (
        canal,
        oficina,
        periodo,
        skus_recomendados,
        str(skus_actuales),
        porcentaje_texto(prob_max),
        oportunidad,
        table_data,
        fig,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
