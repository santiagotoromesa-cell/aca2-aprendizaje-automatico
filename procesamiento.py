"""Carga de datos y preparacion de secuencias para la aplicacion.

Replica exactamente la transformacion usada durante el entrenamiento:
logaritmo, normalizacion con las estadisticas del periodo de entrenamiento,
medias moviles de 7 y 28 dias y codificacion del dia de la semana.
"""
from __future__ import annotations
import json, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RUTA = Path(__file__).parent
BD = RUTA / "demanda.db"
CONFIG = RUTA / "configuracion_modelo.json"


def cargar_configuracion() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def consultar_catalogo() -> pd.DataFrame:
    """Catalogo de productos modelados, leido con SQL."""
    with sqlite3.connect(BD) as con:
        return pd.read_sql_query(
            "SELECT codigo, descripcion, unidades_totales, dias_con_venta "
            "FROM productos ORDER BY unidades_totales DESC", con)


def consultar_serie(codigo: str) -> pd.DataFrame:
    """Serie diaria de un producto, con los dias sin venta completados en cero."""
    with sqlite3.connect(BD) as con:
        df = pd.read_sql_query(
            "SELECT fecha, unidades FROM demanda_diaria WHERE codigo = ? ORDER BY fecha",
            con, params=(codigo,))
        rango = pd.read_sql_query(
            "SELECT MIN(fecha) ini, MAX(fecha) fin FROM demanda_diaria", con).iloc[0]
    df["fecha"] = pd.to_datetime(df.fecha)
    calendario = pd.date_range(rango.ini, rango.fin, freq="D")
    serie = (df.set_index("fecha").unidades.reindex(calendario).fillna(0.0)
               .rename("unidades").rename_axis("fecha").reset_index())
    return serie


def construir_entrada(serie: pd.DataFrame, codigo: str, cfg: dict, hasta: pd.Timestamp):
    """Arma el tensor de entrada (1, ventana, canales) con la historia previa a `hasta`."""
    ventana = cfg["ventana"]
    j = cfg["productos"].index(codigo)
    mu, sigma = cfg["mu"][j], cfg["sigma"][j]

    s = serie[serie.fecha <= hasta].copy()
    if len(s) < ventana:
        raise ValueError("Historia insuficiente para esa fecha de corte.")
    z = (np.log1p(s.unidades.values) - mu) / sigma
    z_serie = pd.Series(z)
    r7 = z_serie.rolling(7, min_periods=1).mean().values
    r28 = z_serie.rolling(28, min_periods=1).mean().values

    fechas = s.fecha.iloc[-ventana:]
    dow = np.zeros((ventana, 7), dtype="float32")
    dow[np.arange(ventana), fechas.dt.dayofweek.values] = 1.0

    x = np.concatenate([z[-ventana:, None], r7[-ventana:, None],
                        r28[-ventana:, None], dow], axis=1)
    return x[None, ...].astype("float32"), np.array([j], dtype="int32")


def a_unidades(valor_normalizado: float, codigo: str, cfg: dict) -> float:
    """Deshace la normalizacion y el logaritmo, y aplica la calibracion del sesgo.

    El factor de calibracion se estimo sobre el conjunto de validacion en el
    notebook y se transporta en la configuracion; recalcularlo aqui dejaria el
    pronostico descalibrado.
    """
    j = cfg["productos"].index(codigo)
    mu_o, sg_o = cfg["mu_objetivo"][j], cfg["sigma_objetivo"][j]
    bruto = max(float(np.expm1(valor_normalizado * sg_o + mu_o)), 0.0)
    calibrado = np.expm1(np.log1p(bruto) + cfg.get("calibracion_log", 0.0))
    return float(max(calibrado, 0.0))


def clasificar_nivel(unidades: float, codigo: str, cfg: dict) -> str:
    bajo, alto = cfg["cortes_nivel"][codigo]
    if unidades <= bajo:
        return "Bajo"
    return "Medio" if unidades <= alto else "Alto"


def demanda_real(serie: pd.DataFrame, desde: pd.Timestamp, horizonte: int):
    """Demanda observada en los `horizonte` dias siguientes, si existe en los datos."""
    tramo = serie[(serie.fecha > desde) &
                  (serie.fecha <= desde + pd.Timedelta(days=horizonte))]
    if len(tramo) < horizonte:
        return None
    return float(tramo.unidades.sum())
