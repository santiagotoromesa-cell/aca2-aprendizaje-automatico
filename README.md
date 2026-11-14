# Pronóstico de demanda de productos con una red neuronal recurrente

Aplicación interactiva en Streamlit que estima las unidades que se venderán de un producto
durante los próximos siete días, a partir de su historial reciente de ventas.

**ACA N° 2 — Aprendizaje Automático**
Especialización en Inteligencia Artificial · Escuela de Ingeniería
Corporación Unificada Nacional de Educación Superior (CUN) · 2026

Integrantes: Santiago Toro Mesa · Darwin Lozano Lozano
Docente: Dr. Hamilton Rivera Flor

---

## Qué hace

1. Permite elegir uno de los 200 productos modelados y la fecha desde la cual se quiere pronosticar.
2. Entrega las unidades esperadas para los siguientes siete días, el nivel de demanda (bajo, medio o
   alto) y una **cantidad sugerida de pedido** con margen de seguridad ajustable.
3. Para fechas pasadas muestra además la demanda que realmente ocurrió, de modo que el pronóstico se
   puede contrastar.
4. Reporta el desempeño del modelo frente a las demás arquitecturas evaluadas.

## Datos y modelo

- **Dataset:** UCI Online Retail (Chen, 2015). 541.909 transacciones de un comercio electrónico del
  Reino Unido entre diciembre de 2010 y diciembre de 2011.
- **Gestión de datos:** SQLite. La depuración y la agregación a demanda diaria se hacen con consultas SQL.
- **Entrada del modelo:** ventana de 28 días con 10 canales (demanda normalizada, medias móviles de 7 y
  28 días, y día de la semana), más un *embedding* de identidad del producto.
- **Arquitectura:** GRU apilada (64 y 32 unidades) con dropout, capa densa y calibración posterior del
  sesgo estimada sobre el conjunto de validación.
- **Partición:** estrictamente temporal. Entrenamiento hasta 2011-09-14, validación hasta 2011-10-31 y
  prueba desde esa fecha.

El entrenamiento completo, con la comparación de arquitecturas y todas las métricas, está documentado en
el notebook `ACA_2_Aprendizaje_Automatico.ipynb` del archivo comprimido del proyecto.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz de Streamlit |
| `procesamiento.py` | Consultas SQL y preparación de secuencias, idénticas a las del notebook |
| `modelo_demanda.keras` | Red entrenada |
| `configuracion_modelo.json` | Ventana, horizonte, catálogo, parámetros de normalización y calibración |
| `metricas.json` | Métricas del modelo que muestra la interfaz |
| `backtest_ejemplo.csv` | Seguimiento del pronóstico del producto representativo |
| `demanda.db` | Base SQLite con la demanda diaria de los 200 productos |
| `requirements.txt` | Dependencias |

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota sobre el alcance

El modelo es un apoyo a la decisión de compra, no un reemplazo del criterio del responsable de
inventarios. Al estar entrenado con un solo año de historia, no aprende la estacionalidad anual y tiende
a subestimar la temporada de fin de año; esa limitación está documentada en el notebook.
