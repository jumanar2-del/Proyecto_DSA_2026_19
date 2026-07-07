# Sistema de Recomendación de Productos — Entrega 2
**Proyecto DSA 2026** | Jessica Salazar · Juan Camilo Umaña

## Descripción

Sistema de recomendación de productos para clientes comerciales de una empresa de consumo masivo. Utiliza modelos de machine learning para predecir la probabilidad de compra de cada producto por cliente y genera recomendaciones Top-3 personalizadas.

## Estructura del proyecto

```
Proyecto_DSA_2026_19/
├── Clientes.csv                       # Datos de clientes (6,316 registros)
├── Resumen.csv                        # Transacciones de ventas (84,676 registros)
├── 01_EDA.ipynb                       # Análisis exploratorio de datos (Entrega 1)
├── 02_modelo_recomendacion.ipynb      # Desarrollo y evaluación de modelos (⭐ NUEVO)
├── 03_tablero.py                      # Dashboard interactivo (⭐ NUEVO)
├── modelo_recomendacion.py            # Script Python del modelo
├── recomendaciones_top3.csv           # Top-3 productos por cliente
├── probabilidades_completas.csv       # Todas las probabilidades
├── comparativa_modelos.csv            # Métricas de los 3 modelos
├── mlflow.db                          # Base de datos MLflow con experimentos
└── requirements.txt                   # Dependencias
```

## Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/jumanar2-del/Proyecto_DSA_2026_19.git
cd Proyecto_DSA_2026_19
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## Uso

### 1. Ejecutar el notebook de modelado

Abre `02_modelo_recomendacion.ipynb` en Jupyter Lab/Notebook y ejecuta todas las celdas:

```bash
jupyter lab 02_modelo_recomendacion.ipynb
```

**El notebook incluye:**
- Carga y limpieza de datos
- Ingeniería de características (10 features + target binario)
- Split temporal train/test (último mes = test)
- Entrenamiento de 3 modelos (Logistic Regression, Random Forest, Gradient Boosting)
- Registro de experimentos en MLflow
- Evaluación y selección del mejor modelo (Gradient Boosting — ROC-AUC 0.91)
- Generación de recomendaciones Top-3 por cliente
- Visualizaciones y análisis de resultados

**Archivos generados:**
- `recomendaciones_top3.csv`
- `probabilidades_completas.csv`
- `comparativa_modelos.csv`
- `mlflow.db` (base de datos SQLite con experimentos)
- Figuras: `comparativa_modelos.png`, `importancia_variables.png`, `distribucion_recomendaciones.png`

### 2. Ver experimentos en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abre http://localhost:5000 en tu navegador.

### 3. Ejecutar el tablero interactivo

```bash
streamlit run 03_tablero.py
```

El tablero se abrirá automáticamente en http://localhost:8501

**Vista 1 — Por SKU:**
- Selecciona un producto (SKU)
- Filtra por oficina de ventas y segmento
- Ver clientes con mayor probabilidad de compra
- KPIs: clientes con oportunidad, probabilidad promedio, venta histórica
- Tabla de clientes recomendados con prioridad (Alta/Media/Normal)
- Gráficas de distribución de probabilidades y top 10 clientes

**Vista 2 — Por Cliente:**
- Selecciona un cliente
- Ver información comercial (oficina, segmento, canal, frecuencia de visita)
- SKUs activos y última compra
- Top-3 productos recomendados con probabilidades y razones
- Gráfica de probabilidad de compra por producto
- Historial de ventas por mes

## Resultados principales

### Mejor modelo: Gradient Boosting

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.912 |
| PR-AUC | 0.737 |
| F1-Score | 0.660 |
| Precisión | 0.719 |
| Recall | 0.610 |

### Importancia de variables

1. **freq_compra** (57.5%) — frecuencia histórica de compra del producto
2. **veces_comprado** (30.0%) — número de meses con compra
3. **producto_enc** (4.4%) — tipo de producto
4. **ventas_cliente_ant** (3.8%) — ventas totales del cliente

### Conclusiones

- El historial de compra del cliente es el predictor más fuerte (87% de importancia acumulada)
- La segmentación comercial aporta valor incremental (~3%)
- Gradient Boosting supera a Random Forest en precisión (72% vs 53%), reduciendo recomendaciones incorrectas
- El modelo discrimina bien entre clientes que comprarán vs no comprarán (ROC-AUC = 0.91)
- Se generaron recomendaciones para 6,261 clientes activos

## Tecnologías utilizadas

- **Python 3.13**
- **Pandas** — manipulación de datos
- **Scikit-learn** — modelos de machine learning
- **MLflow** — versionamiento y registro de experimentos
- **Streamlit** — dashboard interactivo
- **Matplotlib / Seaborn** — visualizaciones

## Autores

- Jessica Salazar
- Juan Camilo Umaña

**Universidad de los Andes** | Maestría en Inteligencia Analítica de Datos | DSA 2026
