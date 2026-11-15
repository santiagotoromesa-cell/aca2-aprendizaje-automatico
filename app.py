"""Pronostico de demanda con red neuronal recurrente.

ACA 2 - Aprendizaje Automatico | Especializacion en Inteligencia Artificial, CUN
Aplicacion interactiva que consume el modelo entrenado en el notebook del proyecto.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import procesamiento as prc

RUTA = Path(__file__).parent

st.set_page_config(page_title="Pronóstico de demanda · CUN",
                   page_icon="📦", layout="wide")

COLOR_REAL, COLOR_PRON = "#1f4e79", "#e07b39"
SEMAFORO = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}


# --------------------------------------------------------------- carga diferida
@st.cache_resource(show_spinner="Cargando la red neuronal…")
def cargar_modelo():
    from tensorflow import keras
    return keras.models.load_model(RUTA / "modelo_demanda.keras")


@st.cache_data
def cargar_base():
    return prc.cargar_configuracion(), prc.consultar_catalogo()


@st.cache_data
def cargar_serie(codigo):
    return prc.consultar_serie(codigo)


@st.cache_data
def cargar_metricas():
    import json
    with open(RUTA / "metricas.json", encoding="utf-8") as f:
        return json.load(f)


cfg, catalogo = cargar_base()
metricas = cargar_metricas()

# ------------------------------------------------------------------ encabezado
st.title("📦 Pronóstico de demanda de productos")
st.markdown(
    f"**Red neuronal recurrente ({cfg['arquitectura']})** entrenada sobre "
    f"{len(cfg['productos'])} productos de un comercio electrónico del Reino Unido. "
    f"A partir de los últimos **{cfg['ventana']} días** de ventas, la red estima "
    f"las **unidades que se venderán en los próximos {cfg['horizonte']} días**."
)

# --------------------------------------------------------------------- controles
with st.sidebar:
    st.header("Parámetros")
    etiquetas = {f"{r.descripcion.title()} · {r.codigo}": r.codigo
                 for r in catalogo.itertuples()}
    elegido = st.selectbox("Producto", list(etiquetas), index=0)
    codigo = etiquetas[elegido]

    serie = cargar_serie(codigo)
    f_min = serie.fecha.min() + pd.Timedelta(days=cfg["ventana"])
    f_max = serie.fecha.max()
    # Por defecto se ubica en la última fecha cuyos 7 días siguientes sí están
    # registrados, para que el usuario pueda contrastar el pronóstico con lo ocurrido.
    por_defecto = f_max - pd.Timedelta(days=cfg["horizonte"])
    corte = st.slider("Pronosticar a partir de", min_value=f_min.to_pydatetime(),
                      max_value=f_max.to_pydatetime(), value=por_defecto.to_pydatetime(),
                      format="DD/MM/YYYY")
    corte = pd.Timestamp(corte).normalize()

    cobertura = st.slider("Margen de seguridad del pedido (%)", 0, 50, 15, step=5,
                          help="Porcentaje adicional sobre el pronóstico para "
                               "absorber la variabilidad de la demanda.")
    st.divider()
    st.caption(
        f"Datos: UCI Online Retail · {cfg['fecha_fin_datos']}\n\n"
        f"Entrenamiento hasta {cfg['corte_entrenamiento']} · "
        f"validación hasta {cfg['corte_validacion']}"
    )

# ---------------------------------------------------------------------- calculo
modelo = cargar_modelo()
try:
    x_sec, x_pro = prc.construir_entrada(serie, codigo, cfg, corte)
except ValueError as e:
    st.error(str(e)); st.stop()

pred_norm = float(modelo.predict([x_sec, x_pro], verbose=0).ravel()[0])
pronostico = prc.a_unidades(pred_norm, codigo, cfg)
nivel = prc.clasificar_nivel(pronostico, codigo, cfg)
real = prc.demanda_real(serie, corte, cfg["horizonte"])
pedido = pronostico * (1 + cobertura / 100)

# --------------------------------------------------------------------- tarjetas
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Pronóstico próximos {cfg['horizonte']} días", f"{pronostico:,.0f} und".replace(",", "."))
c2.metric("Nivel de demanda", f"{SEMAFORO[nivel]} {nivel}")
c3.metric(f"Pedido sugerido (+{cobertura}%)", f"{pedido:,.0f} und".replace(",", "."))
if real is None:
    c4.metric("Demanda observada", "sin dato")
else:
    error = pronostico - real
    c4.metric("Demanda observada", f"{real:,.0f} und".replace(",", "."),
              delta=f"{error:+,.0f} de diferencia".replace(",", "."), delta_color="off")

st.divider()

# ----------------------------------------------------------------------- grafico
izq, der = st.columns([3, 2])

with izq:
    st.subheader("Historia reciente y pronóstico")
    hist = serie[(serie.fecha <= corte) &
                 (serie.fecha > corte - pd.Timedelta(days=90))].copy()
    hist["serie"] = "Venta diaria observada"
    base = alt.Chart(hist).mark_area(
        line={"color": COLOR_REAL}, color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#ffffff", offset=0),
                   alt.GradientStop(color=COLOR_REAL, offset=1)],
            x1=1, x2=1, y1=1, y2=0), opacity=0.25
    ).encode(
        x=alt.X("fecha:T", title="Fecha"),
        y=alt.Y("unidades:Q", title="Unidades por día"),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
                 alt.Tooltip("unidades:Q", title="Unidades", format=".0f")]
    )
    futuro = pd.DataFrame({
        "fecha": pd.date_range(corte + pd.Timedelta(days=1), periods=cfg["horizonte"]),
        "unidades": [pronostico / cfg["horizonte"]] * cfg["horizonte"]})
    capa_pron = alt.Chart(futuro).mark_bar(color=COLOR_PRON, opacity=.85).encode(
        x="fecha:T", y="unidades:Q",
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
                 alt.Tooltip("unidades:Q", title="Promedio diario pronosticado", format=".0f")])
    st.altair_chart((base + capa_pron).properties(height=300), width="stretch")
    st.caption("En azul, la venta diaria observada de los últimos 90 días. En naranja, "
               "el pronóstico de la red repartido en promedio sobre los días siguientes.")

with der:
    st.subheader("Comportamiento del modelo")
    comp = pd.DataFrame(metricas["comparacion"])[["modelo", "MAE"]]
    comp["MAE"] = comp.MAE.round(0)
    graf = alt.Chart(comp).mark_bar().encode(
        x=alt.X("MAE:Q", title="Error absoluto medio (unidades/semana)"),
        y=alt.Y("modelo:N", sort="x", title=None),
        color=alt.condition(
            alt.datum.modelo == metricas["arquitectura_seleccionada"],
            alt.value(COLOR_PRON), alt.value("#b8c4cf")),
        tooltip=["modelo", "MAE"]
    ).properties(height=260)
    st.altair_chart(graf, width="stretch")
    cl = metricas["clasificacion"]
    st.markdown(
        f"Al traducir el pronóstico a un nivel de demanda, el modelo acierta en "
        f"**{cl['exactitud']*100:.0f} %** de los casos "
        f"(F1 macro **{cl['f1_macro']:.2f}**)."
    )

# ------------------------------------------------------------------ comparativa
st.divider()
st.subheader("Revisión del pronóstico en todo el periodo de prueba")
bt = pd.read_csv(RUTA / "backtest_ejemplo.csv", parse_dates=["fecha"])
bt = bt.melt("fecha", var_name="serie", value_name="unidades").replace(
    {"real": "Demanda real", "pronostico": "Pronóstico de la red"})
linea = alt.Chart(bt).mark_line(point=True, strokeWidth=2.5).encode(
    x=alt.X("fecha:T", title="Fecha del pronóstico"),
    y=alt.Y("unidades:Q", title=f"Unidades en {cfg['horizonte']} días"),
    color=alt.Color("serie:N", title=None,
                    scale=alt.Scale(domain=["Demanda real", "Pronóstico de la red"],
                                    range=[COLOR_REAL, COLOR_PRON])),
    tooltip=[alt.Tooltip("fecha:T", format="%d/%m/%Y"), "serie:N",
             alt.Tooltip("unidades:Q", format=".0f")]
).properties(height=280)
st.altair_chart(linea, width="stretch")
demo = cfg.get("producto_demo", {})
st.caption(
    f"Producto representativo {demo.get('codigo', '')} "
    f"({demo.get('descripcion', '').title()}), elegido por quedar en la mediana del error "
    "porcentual entre los productos de alta rotación, sobre datos que la red nunca vio "
    "durante el entrenamiento."
)

with st.expander("Ficha técnica del proyecto"):
    st.markdown(f"""
**Dataset.** UCI Online Retail: {metricas['muestras']['entrenamiento'] + metricas['muestras']['validacion'] + metricas['muestras']['prueba']:,} secuencias
construidas a partir de 541.909 transacciones reales de un comercio electrónico del Reino Unido
entre diciembre de 2010 y diciembre de 2011. Los datos se gestionan en una base **SQLite**,
desde la cual se depuran y agregan con consultas SQL.

**Entrada de la red.** Ventana de {cfg['ventana']} días con {metricas['canales_entrada']} canales:
demanda normalizada, medias móviles de 7 y 28 días, y codificación del día de la semana.
Cada producto aporta además un vector de identidad aprendido (*embedding*).

**Arquitectura seleccionada.** {cfg['arquitectura']}, elegida por su error en el conjunto de
validación frente a una red densa, una LSTM y una GRU.

**Partición temporal.** Entrenamiento hasta {cfg['corte_entrenamiento']},
validación hasta {cfg['corte_validacion']} y prueba desde esa fecha hasta
{cfg['fecha_fin_datos']}. No hay fugas de información del futuro al pasado.
""".replace(",", "."))

st.caption("Corporación Unificada Nacional de Educación Superior · Especialización en "
           "Inteligencia Artificial · Asignatura Aprendizaje Automático")
