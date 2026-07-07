# Tablero - Entrega 2

Este tablero corresponde a la parte de visualización del proyecto **Sistema de recomendación de productos para clientes comerciales**.

El objetivo es mostrar, de acuerdo con la maqueta definida en la Entrega 1, los clientes con mayor oportunidad para un SKU seleccionado y el detalle de los SKUs recomendados para un cliente.

## Archivos incluidos

```text
dashboard/
├── app.py
├── requirements.txt
├── Dockerfile
├── assets/
│   └── style.css
└── data/
    └── recomendaciones_modelo.csv
```

## Cómo ejecutar localmente

Desde la carpeta `dashboard`:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el tablero:

```bash
python app.py
```

Abrir en el navegador:

```text
http://127.0.0.1:8050
```

## Cómo ejecutar con Docker

Desde la carpeta `dashboard`:

```bash
docker build -t tablero-recomendador .
docker run -p 8050:8050 tablero-recomendador
```

Abrir en el navegador:

```text
http://127.0.0.1:8050
```

## Datos esperados

El archivo `data/recomendaciones_modelo.csv` representa la salida del modelo. Cuando se tenga el modelo final, se puede reemplazar ese archivo manteniendo las siguientes columnas:

```text
Cliente, Canal, Oficina, SKU, Producto, Categoria, Ultima_compra_dias, Probabilidad, Prioridad, Venta_potencial, Razon
```

No se desarrolla la API dentro de esta carpeta. Esta carpeta corresponde únicamente a las fuentes del tablero.
