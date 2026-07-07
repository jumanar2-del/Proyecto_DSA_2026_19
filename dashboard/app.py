from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "recomendaciones_modelo.csv"


# -----------------------------------------------------------------------------
# Carga de datos
# Este archivo representa la salida del modelo para la Entrega 2.
# Cuando tu compañero tenga el modelo final, solo debe reemplazar este CSV
# manteniendo las mismas columnas.
# -----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    numeric_cols = ["Ultima_compra_dias", "Probabilidad", "Venta_potencial"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Probabilidad_pct"] = (df["Probabilidad"] * 100).round(1)
    df["Ultima_compra"] = "Hace " + df["Ultima_compra_dias"].astype(int).astype(str) + " días"
    return df


df_recomendaciones = load_data()

app = Dash(__name__)
server = app.server
app.title = "Recomendador de clientes"


def format_money(value: float) -> str:
    if pd.isna(value):
        return "$0,0 M"
    return f"${value / 1_000_000:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")


def build_options(series):
    values = sorted(series.dropna().astype(str).unique())
    return [{"label": value, "value": value} for value in values]


app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="header",
            children=[
                html.Div(
                    children=[
                        html.H1("Recomendador de Clientes por SKU"),
                        html.P(
                            "Tablero de apoyo comercial para priorizar clientes y productos con mayor probabilidad de compra."
                        ),
                    ]
                )
            ],
        ),
        html.Div(
            className="filters",
            children=[
                html.Div(
                    className="filter-item",
                    children=[
                        html.Label("SKU"),
                        dcc.Dropdown(
                            id="filtro-sku",
                            options=build_options(df_recomendaciones["SKU"]),
                            value=sorted(df_recomendaciones["SKU"].unique())[0],
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    className="filter-item",
                    children=[
                        html.Label("Categoría"),
                        dcc.Dropdown(
                            id="filtro-categoria",
                            options=[{"label": "Todas", "value": "Todas"}]
                            + build_options(df_recomendaciones["Categoria"]),
                            value="Todas",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    className="filter-item",
                    children=[
                        html.Label("Canal"),
                        dcc.Dropdown(
                            id="filtro-canal",
                            options=[{"label": "Todos", "value": "Todos"}]
                            + build_options(df_recomendaciones["Canal"]),
                            value="Todos",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    className="filter-item",
                    children=[
                        html.Label("Oficina"),
                        dcc.Dropdown(
                            id="filtro-oficina",
                            options=[{"label": "Todas", "value": "Todas"}]
                            + build_options(df_recomendaciones["Oficina"]),
                            value="Todas",
                            clearable=False,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="cards",
            children=[
                html.Div(className="card", children=[html.H3(id="kpi-clientes"), html.P("Clientes con oportunidad")]),
                html.Div(className="card", children=[html.H3(id="kpi-probabilidad"), html.P("Prob. promedio")]),
                html.Div(className="card", children=[html.H3(id="kpi-venta"), html.P("Venta potencial")]),
                html.Div(className="card", children=[html.H3(id="kpi-prioridad"), html.P("Prioridad")]),
            ],
        ),
        html.Div(
            className="section",
            children=[
                html.Div(
                    className="section-title",
                    children=[
                        html.H2("Clientes con mayor oportunidad para el SKU seleccionado"),
                        html.P("Seleccione una fila para ver el detalle del cliente y sus SKUs recomendados."),
                    ],
                ),
                dash_table.DataTable(
                    id="tabla-clientes",
                    columns=[
                        {"name": "Cliente", "id": "Cliente"},
                        {"name": "Canal", "id": "Canal"},
                        {"name": "Oficina", "id": "Oficina"},
                        {"name": "Última compra", "id": "Ultima_compra"},
                        {"name": "Prob. compra", "id": "Probabilidad_pct", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Prioridad", "id": "Prioridad"},
                    ],
                    data=[],
                    row_selectable="single",
                    selected_rows=[0],
                    page_size=8,
                    sort_action="native",
                    style_cell={"textAlign": "left", "fontFamily": "Arial", "padding": "10px"},
                    style_header={"fontWeight": "bold"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{Prioridad} = Alta", "column_id": "Prioridad"}, "fontWeight": "bold"},
                    ],
                ),
            ],
        ),
        html.Div(
            className="section",
            children=[
                html.Div(
                    className="section-title",
                    children=[
                        html.H2("Detalle del cliente: Top SKUs recomendados"),
                        html.P(id="cliente-seleccionado"),
                    ],
                ),
                html.Div(
                    className="mini-cards",
                    children=[
                        html.Div(className="mini-card", children=[html.H4(id="detalle-ultima"), html.P("Última compra")]),
                        html.Div(className="mini-card", children=[html.H4(id="detalle-skus"), html.P("SKUs actuales")]),
                        html.Div(className="mini-card", children=[html.H4(id="detalle-categorias"), html.P("Categorías activas")]),
                        html.Div(className="mini-card", children=[html.H4(id="detalle-oportunidad"), html.P("Oportunidad")]),
                    ],
                ),
                dash_table.DataTable(
                    id="tabla-skus",
                    columns=[
                        {"name": "SKU", "id": "SKU"},
                        {"name": "Producto", "id": "Producto"},
                        {"name": "Categoría", "id": "Categoria"},
                        {"name": "Prob.", "id": "Probabilidad_pct", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Razón", "id": "Razon"},
                    ],
                    data=[],
                    page_size=5,
                    sort_action="native",
                    style_cell={"textAlign": "left", "fontFamily": "Arial", "padding": "10px"},
                    style_header={"fontWeight": "bold"},
                ),
                dcc.Graph(id="grafico-skus"),
            ],
        ),
    ],
)


def filtrar_base(sku, categoria, canal, oficina):
    df = df_recomendaciones.copy()

    if sku:
        df = df[df["SKU"].astype(str) == str(sku)]
    if categoria != "Todas":
        df = df[df["Categoria"].astype(str) == str(categoria)]
    if canal != "Todos":
        df = df[df["Canal"].astype(str) == str(canal)]
    if oficina != "Todas":
        df = df[df["Oficina"].astype(str) == str(oficina)]

    return df.sort_values("Probabilidad", ascending=False)


@app.callback(
    Output("tabla-clientes", "data"),
    Output("kpi-clientes", "children"),
    Output("kpi-probabilidad", "children"),
    Output("kpi-venta", "children"),
    Output("kpi-prioridad", "children"),
    Input("filtro-sku", "value"),
    Input("filtro-categoria", "value"),
    Input("filtro-canal", "value"),
    Input("filtro-oficina", "value"),
)
def actualizar_tabla_clientes(sku, categoria, canal, oficina):
    df = filtrar_base(sku, categoria, canal, oficina)

    clientes = df[["Cliente", "Canal", "Oficina", "Ultima_compra", "Probabilidad_pct", "Prioridad"]].to_dict("records")
    n_clientes = df["Cliente"].nunique()
    prob_prom = "0%" if df.empty else f"{df['Probabilidad'].mean() * 100:.1f}%"
    venta_pot = format_money(df["Venta_potencial"].sum())
    prioridad = "Sin datos" if df.empty else df.iloc[0]["Prioridad"]

    return clientes, str(n_clientes), prob_prom, venta_pot, prioridad


@app.callback(
    Output("cliente-seleccionado", "children"),
    Output("detalle-ultima", "children"),
    Output("detalle-skus", "children"),
    Output("detalle-categorias", "children"),
    Output("detalle-oportunidad", "children"),
    Output("tabla-skus", "data"),
    Output("grafico-skus", "figure"),
    Input("tabla-clientes", "derived_virtual_data"),
    Input("tabla-clientes", "selected_rows"),
)
def actualizar_detalle_cliente(tabla_clientes, selected_rows):
    if not tabla_clientes:
        empty_fig = px.bar(title="Probabilidad por SKU")
        return "Cliente seleccionado: sin datos", "-", "-", "-", "-", [], empty_fig

    selected_idx = selected_rows[0] if selected_rows else 0
    selected_idx = min(selected_idx, len(tabla_clientes) - 1)
    cliente = tabla_clientes[selected_idx]["Cliente"]

    df_cliente = df_recomendaciones[df_recomendaciones["Cliente"].astype(str) == str(cliente)].copy()
    df_cliente = df_cliente.sort_values("Probabilidad", ascending=False)

    ultima = f"{int(df_cliente['Ultima_compra_dias'].min())} días"
    skus_actuales = str(df_cliente["SKU"].nunique())
    categorias = str(df_cliente["Categoria"].nunique())
    oportunidad = df_cliente.iloc[0]["Prioridad"]

    table_data = df_cliente[["SKU", "Producto", "Categoria", "Probabilidad_pct", "Razon"]].to_dict("records")

    fig = px.bar(
        df_cliente,
        x="SKU",
        y="Probabilidad",
        text="Probabilidad_pct",
        title="Probabilidad por SKU",
        labels={"Probabilidad": "Probabilidad de compra", "SKU": "SKU"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", margin={"l": 40, "r": 20, "t": 50, "b": 40})

    return f"Cliente seleccionado: {cliente}", ultima, skus_actuales, categorias, oportunidad, table_data, fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
