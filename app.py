"""
Options Analytics Dashboard - Panel 1: Pricing & Validacion
Primera version real (no mockup) usando SOLO pipeline.py y heston_engine.py
ya construidos y probados.

Correr con: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl
import heston_engine as he

st.set_page_config(page_title="Options Analytics Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cache: no volver a pegarle a yfinance/FRED en cada rerun de Streamlit
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Descargando y limpiando cadena de opciones...")
def cached_snapshot(ticker, expiry, spread_max_rel, moneyness_bounds):
    return pl.build_snapshot(ticker, expiry, spread_max_rel=spread_max_rel,
                              moneyness_bounds=moneyness_bounds)


@st.cache_data(ttl=300, show_spinner="Calibrando Heston a la sonrisa completa...")
def cached_calibration(ticker, expiry, snapshot_hash, S0, r, q, strikes_tuple,
                        mids_tuple, spreads_tuple, tau):
    quotes_by_tau = {tau: (np.array(strikes_tuple), np.array(mids_tuple),
                           np.array(spreads_tuple))}
    return he.calibrate_heston(S0, r, q, quotes_by_tau)


def recommend_engine(moneyness, tau_years, spread_rel):
    """Bandera de motor recomendado por contrato, con justificacion de una linea.
    Regla del trader (brief): ATM y corto plazo -> B&S basta; alas o largo
    plazo -> se necesita el skew de Heston."""
    if spread_rel is not None and spread_rel > 0.15:
        return ("AVISO", "Spread > 15% del mid: quote iliquida, desconfia del "
                          "precio observado antes de comparar motores.")
    atm = abs(moneyness - 1.0) <= 0.03
    corto = tau_years <= 60/365
    if atm and corto:
        return ("B&S", "ATM y corto plazo: la vol plana basta, es el benchmark "
                        "mas rapido y no hay skew relevante que capturar.")
    ala = abs(moneyness - 1.0) > 0.05
    largo = tau_years > 180/365
    if ala:
        return ("HESTON", "Fuera del dinero: el precio vive en el skew, se "
                           "necesita la estructura rho/xi que B&S no tiene.")
    if largo:
        return ("HESTON", "Largo plazo: la reversion a la media de la varianza "
                           "(kappa, theta) domina la dinamica.")
    return ("HESTON", "Zona intermedia: eleccion conservadora, el skew empieza "
                       "a pesar en cuanto el strike se aleja del ATM.")


# ---------------------------------------------------------------------------
# Sidebar: configuracion del contrato
# ---------------------------------------------------------------------------
st.sidebar.header("Configuracion del contrato")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()

with st.sidebar.expander("Limpieza (avanzado)"):
    spread_max_rel = st.slider("Spread relativo maximo", 0.05, 0.50, 0.20, 0.01)
    m_lo = st.slider("Moneyness minimo", 0.5, 1.0, 0.7, 0.05)
    m_hi = st.slider("Moneyness maximo", 1.0, 1.6, 1.3, 0.05)

if not ticker:
    st.stop()

try:
    under = pl.get_underlying_data(ticker)
except Exception as e:
    st.sidebar.error(f"No se pudo validar '{ticker}': {e}")
    st.stop()

expiry = st.sidebar.selectbox("Vencimiento (expiry)", under["expiries"])
tipo = st.sidebar.radio("Tipo de contrato", ["call", "put"], horizontal=True)

try:
    snap = cached_snapshot(ticker, expiry, spread_max_rel, (m_lo, m_hi))
except Exception as e:
    st.sidebar.error(f"Pipeline fallo: {e}")
    st.stop()

lado = snap[snap["type"] == tipo]
if lado.empty:
    st.error(f"No quedaron contratos '{tipo}' liquidos para {ticker} {expiry} "
             f"despues de limpiar. Relaja los filtros en el sidebar.")
    st.stop()

strike = st.sidebar.selectbox("Strike (K)", sorted(lado["strike"].unique()))
row = lado[lado["strike"] == strike].iloc[0]

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Options Analytics Dashboard")
st.caption(f"{ticker} · {tipo.upper()} K={strike:g} · exp {expiry} · "
           f"Panel 1 -- Pricing & Validacion")

tab1, tab2 = st.tabs(["Pricing & Validacion", "Griegas (proximamente)"])

with tab1:
    S0, K, r, q, tau = row["spot"], row["strike"], row["r"], row["q"], row["tau"]
    mid, iv_mkt = row["mid"], row["iv_mkt"]

    st.markdown(
        "B&S se pricea con la **IV implicita del propio contrato** (convencion "
        "de mesa): reproduce el mid por construccion, asi que su error es ~0% "
        "-- la IV se define exactamente como la sigma que hace que B&S iguale "
        "el precio observado. Esto **no valida el modelo**, es una tautologia; "
        "la prueba real de B&S vive en el Panel 3 (una sola sigma ATM contra "
        "toda la sonrisa)."
    )

    if pd.isna(iv_mkt) or iv_mkt <= 0:
        st.warning("Este contrato no trae IV valida de yfinance; no se puede "
                   "priciar B&S con la convencion de mesa para esta fila.")
        st.stop()

    bs = he.bs_price(S0, K, r, q, tau, iv_mkt)
    bs_price = bs[0]

    # Calibracion Heston sobre TODA la sonrisa del vencimiento (ambos lados,
    # una sola Theta para todo el snapshot -- ver decision de diseño pendiente
    # sobre calibracion global vs por vencimiento)
    strikes_cal = tuple(snap["strike"].values)
    mids_cal = tuple(snap["mid"].values)
    spreads_cal = tuple((snap["ask"] - snap["bid"]).values)

    with st.spinner("Calibrando Heston..."):
        result = cached_calibration(ticker, expiry, len(snap), S0, r, q,
                                     strikes_cal, mids_cal, spreads_cal, tau)

    p = result["params"]
    heston_price = he.cos_price_single(S0, K, r, q, tau, p["v0"], p["kappa"],
                                        p["theta"], p["xi"], p["rho"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Mercado (mid)", f"${mid:,.2f}")
    c2.metric("Black-Scholes", f"${bs_price:,.2f}",
              f"{bs_price-mid:+.2f} vs mid")
    c3.metric("Heston (calibrado)", f"${heston_price:,.2f}",
              f"{heston_price-mid:+.2f} vs mid")

    feller = result["feller"]
    if not feller["cumple"]:
        st.warning(f"Feller VIOLADA: 2*kappa*theta={feller['lhs']:.4f} < "
                    f"xi^2={feller['rhs']:.4f}. Diagnostico, no error -- el "
                    f"skew observado puede exigir esto.")

    motor, razon = recommend_engine(row["moneyness"], tau, row["spread_rel"])
    st.info(f"**Motor recomendado: {motor}** -- {razon}")

    tabla = pd.DataFrame({
        "Motor": ["Black-Scholes", "Heston"],
        "Precio": [bs_price, heston_price],
        "Mercado (mid)": [mid, mid],
        "Error absoluto": [bs_price-mid, heston_price-mid],
        "Error relativo (%)": [100*(bs_price-mid)/mid, 100*(heston_price-mid)/mid],
    })
    st.dataframe(tabla, hide_index=True, use_container_width=True)

    with st.expander("Parametros de Heston calibrados"):
        st.json({k: round(v, 4) for k, v in p.items()})
        st.caption(f"Loss final: {result['loss_final']:.6f}  |  "
                   f"Quotes usadas en la calibracion: {len(snap)}")

with tab2:
    st.info("Panel de Griegas: siguiente paso, usa bs_greeks() y "
            "heston_greeks_fd() que ya estan probados en heston_engine.py.")
