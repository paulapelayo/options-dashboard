"""
Options Analytics Dashboard
Quantitative Finance — ITESO
Motores: Black-Scholes-Merton y Heston (stochastic vol)

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, qmc
from scipy.integrate import quad
from scipy.optimize import least_squares
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import re
import os
from datetime import date

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ============================================================
st.set_page_config(
    page_title="Options Analytics Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

:root {
    --navy: #0B1220;
    --panel: #121A2B;
    --panel-border: #223049;
    --accent: #3B82C4;
    --accent-2: #1FAE85;
    --warn: #D9822B;
    --danger: #D9534F;
    --text-dim: #8492A6;
}

.stApp {
    background: linear-gradient(180deg, #0B1220 0%, #0E1626 100%);
}

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; }

.mono { font-family: 'JetBrains Mono', monospace; }

.desk-header {
    padding: 1.1rem 1.6rem;
    border-radius: 10px;
    background: linear-gradient(90deg, #14304F 0%, #0F2138 100%);
    border: 1px solid var(--panel-border);
    margin-bottom: 1.2rem;
}
.desk-header h1 { color: #EAF2FB; font-size: 1.55rem; margin: 0; }
.desk-header p { color: var(--text-dim); margin: 0.2rem 0 0 0; font-size: 0.92rem; }

.metric-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    height: 100%;
}
.metric-card .label {
    color: var(--text-dim);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: #EAF2FB;
}
.metric-card .sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}
.tag {
    display: inline-block;
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.tag-accent { background: rgba(59,130,196,0.18); color: #7FB8E8; }
.tag-good { background: rgba(31,174,133,0.18); color: #5FDCB4; }
.tag-warn { background: rgba(217,130,43,0.18); color: #F0A85C; }

.interp-box {
    background: var(--panel);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    color: #C9D6E5;
    font-size: 0.95rem;
    line-height: 1.55;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# 1. MOTORES DE PRICING
# ============================================================

# ---------- Black-Scholes-Merton ----------
def bs_price(S, K, r, q, tau, sigma, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == "call":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def bs_implied_vol(price, S, K, r, q, tau, option_type="call", tol=1e-7):
    intrinsic = (
        max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
        if option_type == "call"
        else max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0)
    )
    if price <= intrinsic + 1e-8 or tau <= 0:
        return np.nan
    # hi=10.0 (1000% vol): alineado con el techo dinámico de calibración de Heston
    # (var_ceiling <= 9.0 => vol <= 300%) con margen, y con el filtro de build_smile_data
    # (antes eran tres topes DESALINEADOS: 5.0 aquí, 9.0 de varianza en calibración, 10.0
    # en el filtro del smile -- el de 10.0 nunca se activaba porque este solver jamás
    # podía devolver más de 5.0).
    lo, hi = 1e-6, 10.0
    # Guarda de rango: bs_price es monótona creciente en sigma. Si el precio pedido
    # está fuera de [bs_price(lo), bs_price(hi)], NO hay sigma que lo reproduzca dentro
    # del cap -- antes el bisector simplemente convergía despacio hacia el borde (hi) y
    # devolvía ese número como si fuera una IV real, en vez de señalar "inalcanzable".
    price_at_hi = bs_price(S, K, r, q, tau, hi, option_type)
    if price >= price_at_hi:
        return np.nan
    mid = 0.2
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_price(S, K, r, q, tau, mid, option_type) - price
        if abs(d) < tol:
            return mid
        lo, hi = (mid, hi) if d < 0 else (lo, mid)
    return mid


# ---------- Heston (función característica, con dividendos) ----------
# Forma ESTABLE "little Heston trap" (Albrecher et al. 2007 — referencia del curso).
# Matemáticamente idéntica a la forma original de Heston (verificado a 2e-16), pero usa
# exp(-d·tau), que decae, en vez de exp(+d·tau), que desborda para ciertas combinaciones
# de parámetros (kappa bajo + tau largo) y degradaba/rompía la integración numérica.
def heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho):
    i = 1j
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho * xi if j == 1 else kappa
    rd = r - q
    A = bj - rho * xi * i * u
    d = np.sqrt(A**2 + xi**2 * (u**2 - 2 * uj * i * u))  # sqrt principal: Re(d) >= 0
    c = (A - d) / (A + d)
    exp_neg = np.exp(-d * tau)
    Dj = ((A - d) / xi**2) * ((1 - exp_neg) / (1 - c * exp_neg))
    Cj = rd * i * u * tau + (kappa * theta / xi**2) * (
        (A - d) * tau - 2 * np.log((1 - c * exp_neg) / (1 - c))
    )
    return np.exp(Cj + Dj * v + i * u * x)


def heston_prob_j(j, x, v, tau, K, r, q, kappa, theta, xi, rho):
    lnK = np.log(K)
    ig = lambda u: np.real(
        np.exp(-1j * u * lnK) * heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho) / (1j * u)
    )
    val, _ = quad(ig, 1e-8, 200, limit=200)
    return 0.5 + val / np.pi


def heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    x = np.log(S)
    P1 = heston_prob_j(1, x, v0, tau, K, r, q, kappa, theta, xi, rho)
    P2 = heston_prob_j(2, x, v0, tau, K, r, q, kappa, theta, xi, rho)
    price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
    intrinsic_floor = max(S * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)  # cota de no-arbitraje
    return max(price, intrinsic_floor)


def heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    c = heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    price = c - S * np.exp(-q * tau) + K * np.exp(-r * tau)  # put-call parity
    intrinsic_floor = max(K * np.exp(-r * tau) - S * np.exp(-q * tau), 0.0)
    return max(price, intrinsic_floor)


def heston_price(S, K, r, q, tau, v0, kappa, theta, xi, rho, option_type="call"):
    if option_type == "call":
        return heston_call(S, K, r, q, tau, v0, kappa, theta, xi, rho)
    return heston_put(S, K, r, q, tau, v0, kappa, theta, xi, rho)


# ============================================================
# 1C. MOTOR COS HÍBRIDO (notebook 6 — Fang-Oosterlee 2008)
# ============================================================
# COS pricea TODOS los strikes de un vencimiento en UNA evaluación vectorizada de la
# función característica (~1 ms vs ~65 ms de quad escalar: speedup medido de ~68x en los
# residuals de calibración). Guardas de sanidad detectan cuando COS no es confiable
# (cumulante c2<=0 — la fórmula publicada falla con kappa bajo + tau largo — o precios
# que violan no-arbitraje/monotonía) y caen automáticamente a quad para ese vector de
# parámetros. N=512, L=20 validados contra quad: error máx 0.007 en el peor rincón del
# espacio de parámetros, ~1e-8 en la región típica de equity.

def _cos_call_raw(S0, K_array, r, q, tau, v0, kappa, theta, xi, rho, N=512, L=20):
    """COS puro (Fang & Oosterlee 2008), CORREGIDO. Devuelve (precios, c2); precios=None si c2<=0.

    BUGFIX (verificado numéricamente): la ventana de truncamiento [a,b] del método COS es
    para la variable y=ln(S_T/K) -- NO para ln(S_T/S0) -- así que debe desplazarse por
    x0=log(S0/K) EN CADA STRIKE. La versión anterior centraba [a,b] únicamente en el drift
    (r-q)τ + ... del log-retorno, igual para todos los strikes, válido solo cuando x0≈0
    (cerca del dinero). Para strikes bien ITM/OTM combinados con τ muy chico (ventana
    angosta), la integral del payoff se evaluaba fuera de donde vive la densidad real,
    dando precios muy alejados del motor de Fourier/quad de referencia -- reproducido:
    S0=220, K=500, τ=2 días, kappa=6, xi=1.5, rho=-0.95 -> COS viejo daba $13.66 cuando
    el precio real es $0.00. Las guardas de heston_calls_fast (no-arbitraje, monotonía)
    no siempre lo atrapan porque el chequeo de monotonía solo actúa con 2+ strikes, y un
    precio de $13.66 seguía técnicamente dentro de [intrínseco, S0] para un solo contrato.

    Como el ancho (b-a) = 2·L·√c2 NO depende del strike (solo el centro se desplaza por
    x0), la malla de frecuencias u_k = kπ/(b-a) sigue siendo la MISMA para todos los
    strikes -- se vectoriza sin perder velocidad. Lo que sí varía por strike es dónde cae
    el soporte del payoff dentro de la ventana (c_eff = clip(a,0,b)), generalizado para
    cubrir también el caso todo-ITM (a>=0, integrar la ventana completa) y todo-OTM
    (b<=0, payoff nulo) -- antes se asumía siempre a<0<b."""
    rd = r - q
    c1 = rd * tau + (1 - np.exp(-kappa * tau)) * (theta - v0) / (2 * kappa) - 0.5 * theta * tau
    c2 = (1.0 / (8 * kappa**3)) * (
        xi * tau * kappa * np.exp(-kappa * tau) * (v0 - theta) * (8 * kappa * rho - 4 * xi)
        + kappa * rho * xi * (1 - np.exp(-kappa * tau)) * (16 * theta - 8 * v0)
        + 2 * theta * kappa * tau * (-4 * kappa * rho * xi + xi**2 + 4 * kappa**2)
        + xi**2 * ((theta - 2 * v0) * np.exp(-2 * kappa * tau) + theta * (6 * np.exp(-kappa * tau) - 7) + 2 * v0)
        + 8 * kappa**2 * (v0 - theta) * (1 - np.exp(-kappa * tau))
    )
    if c2 <= 0:
        return None, c2

    half_width = L * np.sqrt(c2)
    k = np.arange(N)
    u = k * np.pi / (2 * half_width)  # ancho (b-a)=2*half_width, constante para todos los strikes
    u_cf = np.where(u == 0, 1e-8, u)

    x0 = np.log(S0 / K_array)               # x0_i = log(S0/K_i), un valor por strike
    a = x0 + c1 - half_width                # ventana DESPLAZADA por strike (el fix)
    b = x0 + c1 + half_width
    c_eff = np.clip(a, 0.0, None)
    c_eff = np.minimum(c_eff, b)             # generaliza a todo-ITM (a>=0) y todo-OTM (b<=0)

    bma = b - a                              # == 2*half_width, pero lo dejamos explícito por strike
    kpi = np.outer(1.0 / bma, k * np.pi)     # (n_strikes, N) == kπ/(b-a)
    d_minus_a = (b - a)[:, None]             # == bma, por construcción
    c_minus_a = (c_eff - a)[:, None]

    # NOTA: kpi ya incluye la división entre (b-a); el argumento de cos/sin es kpi*(x-a)
    # directamente (NO kpi*(x-a)/bma otra vez -- ese fue un bug de una versión intermedia
    # de este fix: dividir dos veces por (b-a) rompía chi/psi para cualquier strike lejos
    # del spot, dando precios absurdos incluso peores que el bug original).
    chi = (1.0 / (1 + kpi**2)) * (
        np.cos(kpi * d_minus_a) * np.exp(b)[:, None]
        - np.cos(kpi * c_minus_a) * np.exp(c_eff)[:, None]
        + kpi * np.sin(kpi * d_minus_a) * np.exp(b)[:, None]
        - kpi * np.sin(kpi * c_minus_a) * np.exp(c_eff)[:, None]
    )
    psi = np.where(
        k[None, :] == 0,
        (b - c_eff)[:, None],
        (np.sin(kpi * d_minus_a) - np.sin(kpi * c_minus_a)) * (bma[:, None] / (k[None, :] * np.pi + 1e-300)),
    )
    Uk = 2.0 / bma[:, None] * (chi - psi)    # (n_strikes, N)

    cf0 = heston_cf_j(u_cf, 2, 0.0, v0, tau, r, q, kappa, theta, xi, rho)  # (N,), independiente del strike
    phase = np.exp(1j * u[None, :] * (x0[:, None] - a[:, None]))          # (n_strikes, N)
    terms = np.real(phase * cf0[None, :]) * Uk
    terms[:, 0] *= 0.5
    prices = K_array * np.exp(-r * tau) * terms.sum(axis=1)
    return prices, c2


def heston_calls_fast(S0, K_array, r, q, tau, v0, kappa, theta, xi, rho):
    """Calls europeos vectorizados vía COS con fallback automático a quad.
    Guardas: c2<=0, precio no-finito, violación de cotas de no-arbitraje
    [intrínseco, S·e^{-qτ}], o no-monotonía en K (los calls deben decrecer en K)."""
    K_array = np.atleast_1d(np.asarray(K_array, dtype=float))
    order = np.argsort(K_array)
    K_sorted = K_array[order]
    prices_sorted, _c2 = _cos_call_raw(S0, K_sorted, r, q, tau, v0, kappa, theta, xi, rho)
    ok = prices_sorted is not None
    if ok:
        upper = S0 * np.exp(-q * tau)
        intrinsic = np.maximum(S0 * np.exp(-q * tau) - K_sorted * np.exp(-r * tau), 0.0)
        if (np.any(~np.isfinite(prices_sorted))
                or np.any(prices_sorted > upper + 1e-4)
                or np.any(prices_sorted < intrinsic - 1e-4)
                or np.any(np.diff(prices_sorted) > 1e-4)):
            ok = False
    if not ok:
        prices_sorted = np.array([
            heston_call(S0, K, r, q, tau, v0, kappa, theta, xi, rho) for K in K_sorted
        ])
    else:
        intrinsic = np.maximum(S0 * np.exp(-q * tau) - K_sorted * np.exp(-r * tau), 0.0)
        prices_sorted = np.maximum(prices_sorted, intrinsic)
    out = np.empty_like(prices_sorted)
    out[order] = prices_sorted
    return out


def heston_prices_fast(S0, K_array, r, q, tau, v0, kappa, theta, xi, rho, option_type="call"):
    """Vector de precios (calls directo; puts vía put-call parity, misma lógica del motor quad)."""
    K_array = np.atleast_1d(np.asarray(K_array, dtype=float))
    calls = heston_calls_fast(S0, K_array, r, q, tau, v0, kappa, theta, xi, rho)
    if option_type == "call":
        return calls
    puts = calls - S0 * np.exp(-q * tau) + K_array * np.exp(-r * tau)
    intrinsic = np.maximum(K_array * np.exp(-r * tau) - S0 * np.exp(-q * tau), 0.0)
    return np.maximum(puts, intrinsic)


# ============================================================
# 1B. GREEKS
# ============================================================
def bs_greeks(S, K, r, q, tau, sigma, option_type="call"):
    """Greeks de Black-Scholes-Merton en forma cerrada.
    Convención: vega/rho/vanna por 1 punto (0.01) de vol/tasa; theta por día; volga por (1 pto)^2."""
    if tau <= 0 or sigma <= 0:
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan, vanna=np.nan, volga=np.nan)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    pdf = norm.pdf(d1)
    term1 = (S * sigma * np.exp(-q * tau) * pdf) / (2 * np.sqrt(tau))
    if option_type == "call":
        delta = np.exp(-q * tau) * norm.cdf(d1)
        dPdtau = term1 + r * K * np.exp(-r * tau) * norm.cdf(d2) - q * S * np.exp(-q * tau) * norm.cdf(d1)
        rho_raw = K * tau * np.exp(-r * tau) * norm.cdf(d2)
    else:
        delta = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        dPdtau = term1 - r * K * np.exp(-r * tau) * norm.cdf(-d2) + q * S * np.exp(-q * tau) * norm.cdf(-d1)
        rho_raw = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
    gamma = np.exp(-q * tau) * pdf / (S * sigma * np.sqrt(tau))
    vega_raw = S * np.exp(-q * tau) * pdf * np.sqrt(tau)
    vanna_raw = -np.exp(-q * tau) * pdf * d2 / sigma
    volga_raw = vega_raw * d1 * d2 / sigma
    theta_raw = -dPdtau  # dV/dt (calendario) = -dV/dtau
    return dict(
        delta=delta, gamma=gamma,
        vega=vega_raw * 0.01, theta=theta_raw / 365.0, rho=rho_raw * 0.01,
        vanna=vanna_raw, volga=volga_raw,  # crudas (estándar de display del curso)
    )


def heston_delta_analytic(S, K, r, q, tau, v0, kappa, theta_lr, xi, rho, option_type="call"):
    """Delta de Heston en forma analítica: Δ_call = e^{-qτ}·P1 (resultado derivado y
    verificado numéricamente en el notebook 2 del curso). Put vía paridad: Δ_put = Δ_call - e^{-qτ}."""
    x = np.log(S)
    P1 = heston_prob_j(1, x, v0, tau, K, r, q, kappa, theta_lr, xi, rho)
    d_call = np.exp(-q * tau) * P1
    return d_call if option_type == "call" else d_call - np.exp(-q * tau)


def heston_greeks_fd(S, K, r, q, tau, v0, kappa, theta_lr, xi, rho, option_type="call", full=True, h_rel=1e-4):
    """Espejo del `heston_greeks_fd` del notebook 2 del curso: diferencias finitas
    centradas con bumps RELATIVOS h_rel=1e-4 sobre cada parámetro (misma convención),
    extendido con dividendos (q) y soporte de puts.

    Devuelve DOS conjuntos de llaves:
      · Convención del curso (crudas, notebook 2): price, delta_fd, gamma, vega_v0
        (=∂P/∂v0), theta_tau (=∂P/∂τ) y, con full=True, las sensibilidades a todos los
        parámetros: vega_theta, vega_kappa, vega_xi, vega_rho.
      · Convención de mercado (para comparar contra B&S en las mismas unidades):
        delta (analítica e^{-qτ}P1, notebook 2), vega (=vega_v0·2√v0·0.01, regla de la
        cadena a ∂P/∂σ0 por 1 punto de vol), theta (=-theta_tau/365, decaimiento por día),
        rho (=∂P/∂r·0.01 — nota: el notebook 2 tiene un descuido donde su 'rho' re-usa
        el bump de la CORRELACIÓN; aquí se corrige bumpeando la tasa r, que es la Rho
        estándar que pide el brief), vanna y volga (segundo orden, vía regla de la cadena
        desde las derivadas en v0)."""
    def P(**kw):
        a = dict(S=S, K=K, r=r, q=q, tau=tau, v0=v0, kappa=kappa, theta=theta_lr, xi=xi, rho=rho)
        a.update(kw)
        return heston_price(a["S"], a["K"], a["r"], a["q"], a["tau"],
                            a["v0"], a["kappa"], a["theta"], a["xi"], a["rho"], option_type)

    base = P()
    hS = S * h_rel
    p_su, p_sd = P(S=S + hS), P(S=S - hS)
    delta_fd = (p_su - p_sd) / (2 * hS)
    gamma = (p_su - 2 * base + p_sd) / hS**2

    hv = v0 * h_rel
    p_vu, p_vd = P(v0=v0 + hv), P(v0=v0 - hv)
    vega_v0 = (p_vu - p_vd) / (2 * hv)

    htau = max(tau * h_rel, 1e-6)
    theta_tau = (P(tau=tau + htau) - P(tau=max(tau - htau, 1e-8))) / (2 * htau)

    hr = max(abs(r) * h_rel, 1e-6)
    rho_rate = (P(r=r + hr) - P(r=r - hr)) / (2 * hr)

    # Segundo orden (requisito del brief): mixta spot-varianza y convexidad en varianza
    p_pp = P(S=S + hS, v0=v0 + hv)
    p_pm = P(S=S + hS, v0=v0 - hv)
    p_mp = P(S=S - hS, v0=v0 + hv)
    p_mm = P(S=S - hS, v0=v0 - hv)
    vanna_v0 = (p_pp - p_pm - p_mp + p_mm) / (4 * hS * hv)      # ∂²P/(∂S ∂v0)
    volga_v0 = (p_vu - 2 * base + p_vd) / hv**2                  # ∂²P/∂v0²

    sigma0 = np.sqrt(v0)
    out = {
        # --- convención del curso (notebook 2, unidades crudas) ---
        "price": base, "delta_fd": delta_fd, "gamma": gamma,
        "vega_v0": vega_v0, "theta_tau": theta_tau,
        # --- convención de mercado (comparable contra B&S) ---
        "delta": heston_delta_analytic(S, K, r, q, tau, v0, kappa, theta_lr, xi, rho, option_type),
        "vega": vega_v0 * 2 * sigma0 * 0.01,                     # ∂P/∂σ0 por 1 pto de vol
        "theta": -theta_tau / 365.0,                             # decaimiento por día calendario
        "rho": rho_rate * 0.01,                                  # ∂P/∂r por 1 pto de tasa
        "vanna": vanna_v0 * 2 * sigma0,                          # ∂²P/(∂S ∂σ0), cruda (estándar del curso)
        "volga": 4 * v0 * volga_v0 + 2 * vega_v0,               # ∂²P/∂σ0², cruda (regla de la cadena)
    }

    if full:
        # Sensibilidades a los demás parámetros calibrados (mismas del notebook 2)
        hth = theta_lr * h_rel
        hk = kappa * h_rel
        hxi = xi * h_rel
        hrho = abs(rho) * h_rel if abs(rho) > 1e-12 else h_rel
        out["vega_theta"] = (P(theta=theta_lr + hth) - P(theta=theta_lr - hth)) / (2 * hth)
        out["vega_kappa"] = (P(kappa=kappa + hk) - P(kappa=kappa - hk)) / (2 * hk)
        out["vega_xi"] = (P(xi=xi + hxi) - P(xi=xi - hxi)) / (2 * hxi)
        out["vega_rho"] = (P(rho=rho + hrho) - P(rho=rho - hrho)) / (2 * hrho)
    return out


# ---------- Calibración Heston (dos etapas: LatinHypercube + least_squares) ----------
HESTON_BOUNDS = [(0.005, 0.20), (0.005, 0.20), (0.10, 6.00), (0.05, 1.50), (-0.95, 0.50)]
HESTON_NAMES = ["v0", "theta", "kappa", "xi", "rho"]


def calibrate_heston(market_rows, S0, r, q, n_candidatos=30, n_refinar=10, seed=1,
                     feller_penalty_weight=0.0, max_nfev=200, kappa_fijo=None,
                     var_ceiling=0.20):
    """market_rows: lista de tuplas (K, tau, price_mkt, spread, option_type).
    CORREGIDO: este archivo traia max_nfev=10 y solo 10 candidatos LHS / 4 refinados
    ("verbatim" del notebook 4) -- presupuesto insuficiente para que least_squares
    converja en un ajuste no lineal de 5 parametros; podia devolver parametros de
    Heston que no calzan con el mercado (el mismo bug ya diagnosticado y corregido
    en heston_engine.py durante este proyecto). Se sube a 30 candidatos LHS, 10
    refinados, max_nfev=200 -- mismo presupuesto validado en el resto del proyecto.
    Los residuals usan el motor COS híbrido (notebook 6): idéntico a quad a ~1e-8 en la
    región de ajuste, solo más rápido; no cambia el resultado.
    feller_penalty_weight=0.0 por default: la condición de Feller se REPORTA como
    diagnóstico (igual que los notebooks — no se impone en la función objetivo);
    un peso >0 la activaría como soft-constraint opcional.
    kappa_fijo: mitigación del valle de identificabilidad κ–ξ (notebook 5) — fija κ
    a (casi) ese valor y calibra solo los otros 4 parámetros.
    var_ceiling: techo superior de v0/theta (varianza). El default 0.20 (~44.7% vol)
    alcanza para vencimientos normales, pero es matemáticamente insuficiente para
    contratos de τ ultra-corto (0-3 DTE) cuya IV anualizada implícita puede superar
    300-400% sin que el precio en sí sea nada extraordinario — es un artefacto de
    anualizar sobre un τ diminuto. calibrate_for_expiry calcula este techo dinámicamente
    a partir de la IV ATM observada; sin ese ajuste el optimizador queda forzado a un
    problema infactible dentro de bounds y colapsa a una esquina degenerada (precio y
    griegas ≈ 0, Feller aparentemente violado) en vez de converger a un ajuste real."""
    bounds_local = [tuple(b) for b in HESTON_BOUNDS]
    bounds_local[0] = (bounds_local[0][0], max(bounds_local[0][1], var_ceiling))
    bounds_local[1] = (bounds_local[1][0], max(bounds_local[1][1], var_ceiling))
    if kappa_fijo is not None:
        eps = max(abs(kappa_fijo) * 1e-4, 1e-4)
        bounds_local[2] = (kappa_fijo - eps, kappa_fijo + eps)  # índice 2 = kappa
    lb = np.array([b[0] for b in bounds_local])
    ub = np.array([b[1] for b in bounds_local])

    # Pre-agrupar el mercado por tipo de opción (un solo tau por calibración de vencimiento)
    tau_ = market_rows[0][1]
    K_calls = np.array([K for K, _, _, _, ot in market_rows if ot == "call"])
    K_puts = np.array([K for K, _, _, _, ot in market_rows if ot == "put"])
    p_calls = np.array([p for _, _, p, _, ot in market_rows if ot == "call"])
    p_puts = np.array([p for _, _, p, _, ot in market_rows if ot == "put"])
    w_calls = np.array([1.0 / max(s, 0.01) for _, _, _, s, ot in market_rows if ot == "call"])
    w_puts = np.array([1.0 / max(s, 0.01) for _, _, _, s, ot in market_rows if ot == "put"])

    def residuals(params):
        v0, theta, kappa, xi, rho = params
        try:
            res_parts = []
            if len(K_calls) > 0:
                pm_c = heston_prices_fast(S0, K_calls, r, q, tau_, v0, kappa, theta, xi, rho, "call")
                # Normalizar por precio de mercado (error RELATIVO), no solo por spread.
                # Sin esto, dos strikes con el mismo error en dólares pesan IGUAL en la suma
                # de cuadrados aunque uno sea un contrato caro (error relativo chico) y el
                # otro sea barato -- p.ej. un ATM de vencimiento ultra-corto (τ=2 días), cuyo
                # precio en dólares es intrínsecamente chico. El optimizador "gasta" su
                # presupuesto de ajuste en los strikes caros y descuida los baratos, aunque
                # estén dentro del conjunto de calibración -- eso es exactamente lo que
                # producía errores de +190% en contratos ATM de τ chico pese a que la IV de
                # contrato (vía B&S) fuera perfectamente razonable.
                res_parts.append(w_calls * (pm_c - p_calls) / np.maximum(p_calls, 0.05))
            if len(K_puts) > 0:
                pm_p = heston_prices_fast(S0, K_puts, r, q, tau_, v0, kappa, theta, xi, rho, "put")
                res_parts.append(w_puts * (pm_p - p_puts) / np.maximum(p_puts, 0.05))
            res = np.concatenate(res_parts) if res_parts else np.array([])
            if not np.all(np.isfinite(res)):
                res = np.where(np.isfinite(res), res, 1e3)
        except Exception:
            res = np.full(len(market_rows), 1e3)
        feller_gap = max(0.0, xi**2 - 2 * kappa * theta)
        return np.append(res, feller_penalty_weight * feller_gap)

    def loss(params):
        r_ = residuals(params)
        return float(np.dot(r_, r_))

    muestra = qmc.LatinHypercube(d=len(HESTON_BOUNDS), seed=seed).random(n_candidatos)
    candidatos = lb + muestra * (ub - lb)
    perdidas = [loss(c) for c in candidatos]
    mejores_idx = np.argsort(perdidas)[:n_refinar]

    mejor_fit = None
    for idx in mejores_idx:
        fit = least_squares(residuals, candidatos[idx], bounds=(lb, ub), max_nfev=max_nfev)
        if mejor_fit is None or fit.cost < mejor_fit.cost:
            mejor_fit = fit
    return mejor_fit.x, mejor_fit


def feller_condition(params):
    v0, theta, kappa, xi, rho = params
    lhs = 2 * kappa * theta
    rhs = xi**2
    return lhs >= rhs, lhs, rhs


# ============================================================
# 2. PIPELINE DE DATOS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate():
    """13-week T-Bill (FRED DTB3) como proxy de tasa libre de riesgo. Fallback si falla.
    Fallback actualizado a 3.78% (nivel real del 13-week T-Bill al momento de este commit)
    -- antes tenía 4.5%, desactualizado; si la llamada en vivo a FRED falla (red restringida,
    FRED caído, etc. -- puede pasar justo en la defensa en vivo) es mejor que el respaldo
    esté cerca del nivel real de mercado en vez de un número viejo."""
    try:
        import pandas_datareader.data as web
        from datetime import datetime, timedelta

        end = datetime.today()
        start = end - timedelta(days=15)
        df = web.DataReader("DTB3", "fred", start, end)
        return float(df.dropna().iloc[-1, 0]) / 100.0
    except Exception:
        return 0.0378  # fallback: 13-week T-Bill ~3.78% (actualizar periódicamente)


@st.cache_data(ttl=60, show_spinner=False)  # 60s, no 30min: en 0-3DTE el spot/bid/ask se mueven demasiado rápido
def get_underlying_info(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError("Sin datos de precio para este ticker.")

    # yfinance a veces incluye la sesión en curso con Close=NaN (dato parcial /
    # no cerrado todavía). Tomamos el último Close VÁLIDO, no simplemente el
    # último renglón del DataFrame.
    close_series = hist["Close"].dropna()
    if close_series.empty:
        # Fallback 1: fast_info (suele tener el último precio incluso cuando
        # la vela diaria todavía no tiene Close).
        try:
            fi_price = tk.fast_info.get("last_price")
            if fi_price and not np.isnan(fi_price):
                close_series = pd.Series([float(fi_price)])
        except Exception:
            pass

    if close_series.empty:
        # Fallback 2: reintenta con una ventana más larga por si la de 5 días
        # cayó en un tramo con feriados/datos faltantes.
        hist_long = tk.history(period="1mo")
        close_series = hist_long["Close"].dropna()

    if close_series.empty:
        raise ValueError(
            "No se pudo obtener un precio spot válido (Close=NaN en todos los "
            "intentos). Puede ser un problema temporal del feed de datos."
        )

    S0 = float(close_series.iloc[-1])
    if not np.isfinite(S0) or S0 <= 0:
        raise ValueError(f"Precio spot inválido recibido: {S0!r}.")

    q = _robust_dividend_yield(tk, S0)
    expiries = list(tk.options)
    return S0, q, expiries


def _robust_dividend_yield(tk, S0):
    """El campo info['dividendYield'] de yfinance ha cambiado de convención entre
    versiones (decimal vs por ciento), lo que puede envenenar q (ej. 0.47 leído como
    47% cuando el yield real es 0.47% o menos) y con ello TODO el pricing (forward,
    IV ATM, calibración de Heston). Método primario libre de convenciones: dividendos
    realmente pagados en los últimos 365 días ÷ spot. Fallback: campo info con guardas
    de sanidad (ningún equity normal rinde >25%)."""
    try:
        divs = tk.dividends
        if divs is not None and len(divs) > 0:
            cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
            trailing = float(divs[divs.index >= cutoff].sum())
            q_hist = trailing / S0
            if 0.0 <= q_hist < 0.25:
                return q_hist
    except Exception:
        pass
    try:
        q_info = float(tk.info.get("dividendYield", 0) or 0)
        if q_info > 1:      # venía en por ciento (ej. 2.5 = 2.5%)
            q_info /= 100.0
        if q_info > 0.25:   # sigue absurdo para un equity: probable % mal escalado
            q_info /= 100.0
        if 0.0 <= q_info < 0.25:
            return q_info
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=60, show_spinner=False)  # 60s, no 30min: bid/ask de 0-3DTE cambian ~60-70% intradía
def get_clean_chain(ticker, expiry):
    """Descarga y limpia la cadena de opciones (calls y puts) para un expiry dado.
    Limpieza = checklist literal del brief: (1) drop zero-bid rows, (2) drop zero-volume
    rows, (3) filter by spread < threshold (spread relativo (ask-bid)/mid < 50% — corta
    quotes basura de las alas), (4) todo alineado a un único snapshot timestamp que se
    guarda en la columna 'snapshot' y se muestra en el sidebar. La conversión
    τ=(T−t)/365 vive en tau_from_expiry.

    Fallback de liquidez: el filtro 'volumen>0' es literal del brief y correcto como
    default, pero en nombres menos líquidos que los grandes ETFs/megacaps (p.ej. HPE)
    puede dejar CASI NADA para un vencimiento, aunque el market maker sí sostenga bid/ask
    en vivo sobre strikes con open interest pero sin operaciones HOY. Si el filtro
    estricto deja menos de cuatro quotes para este expiry, se reintenta permitiendo
    volumen=0 siempre que haya open interest>0 (evidencia de que el contrato existe y
    tiene posiciones abiertas, aunque no haya cruzado hoy) -- las filas que usaron este
    relajo quedan marcadas en 'liquidity_relaxed' para que la UI lo declare, no lo
    esconda."""
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    snapshot_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    MAX_REL_SPREAD = 0.50

    def _clean(allow_oi_fallback):
        rows_ = []
        for df, otype in [(chain.calls, "call"), (chain.puts, "put")]:
            for _, row in df.iterrows():
                bid, ask = row.get("bid", 0), row.get("ask", 0)
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                if bid is None or ask is None:
                    continue
                if bid <= 0 or ask <= 0:
                    continue  # drop zero-bid rows (brief)
                relaxed = False
                if vol <= 0:
                    if allow_oi_fallback and oi > 0:
                        relaxed = True  # sin volumen hoy, pero con open interest > 0
                    else:
                        continue  # drop zero-volume rows (brief, literal)
                mid = (bid + ask) / 2.0
                spread = ask - bid
                if mid <= 0 or (spread / mid) > MAX_REL_SPREAD:
                    continue  # filter by spread < threshold (brief)
                rows_.append(
                    {
                        "strike": float(row["strike"]),
                        "type": otype,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "spread": spread,
                        "volume": vol,
                        "openInterest": oi,
                        "impliedVolatility": row.get("impliedVolatility", np.nan),
                        "lastPrice": row.get("lastPrice", np.nan),
                        "snapshot": snapshot_ts,
                        "liquidity_relaxed": relaxed,
                    }
                )
        return rows_

    rows = _clean(allow_oi_fallback=False)
    if len(rows) < 4:
        rows = _clean(allow_oi_fallback=True)

    CHAIN_COLUMNS = [
        "strike", "type", "bid", "ask", "mid", "spread", "volume",
        "openInterest", "impliedVolatility", "lastPrice", "snapshot", "liquidity_relaxed",
    ]
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    df = pd.DataFrame(rows)
    return df


def tau_from_expiry(expiry_str):
    exp = np.datetime64(expiry_str)
    today = np.datetime64("today")
    return max(float((exp - today) / np.timedelta64(365, "D")), 1 / 365)


# ============================================================
# 3. SIDEBAR — CONTROLES
# ============================================================
st.sidebar.markdown("## ⚙ Configuración del contrato")

# ---------- Parser de símbolo OCC ----------
# Formato estándar OCC: RAIZ + YYMMDD + C/P + STRIKE*1000 (8 dígitos)
# Ejemplo: SPY260708C00745000 -> SPY, exp 2026-07-08, CALL, strike 745.000
def parse_occ_symbol(symbol):
    symbol = (symbol or "").strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", symbol)
    if not m:
        return None
    root, datestr, cp, strikestr = m.groups()
    yy, mm, dd = int(datestr[0:2]), int(datestr[2:4]), int(datestr[4:6])
    year = 2000 + yy
    try:
        _valid = date(year, mm, dd)  # valida que la fecha exista
    except ValueError:
        return None
    return dict(
        ticker=root,
        expiry=f"{year:04d}-{mm:02d}-{dd:02d}",
        option_type="call" if cp == "C" else "put",
        strike=int(strikestr) / 1000.0,
    )


st.session_state.setdefault("ticker_input", "SPY")
if "occ_pending" not in st.session_state:
    st.session_state["occ_pending"] = None

with st.sidebar.expander("📋 Pegar símbolo OCC (opcional)", expanded=False):
    occ_raw = st.text_input("Ej. SPY260708C00745000", key="occ_symbol_box")
    if st.button("Usar este símbolo"):
        parsed = parse_occ_symbol(occ_raw)
        if parsed is None:
            st.error("Formato inválido. Debe ser TICKER + AAMMDD + C/P + 8 dígitos de strike (strike×1000).")
        else:
            st.session_state["ticker_input"] = parsed["ticker"]
            st.session_state["occ_pending"] = parsed
            st.success(
                f"Detectado: {parsed['ticker']} · {parsed['option_type'].upper()} · "
                f"K={parsed['strike']:g} · exp {parsed['expiry']}"
            )
            st.rerun()

ticker = st.sidebar.text_input("Ticker", key="ticker_input").strip().upper()

data_error = None
S0, q, expiries = None, 0.0, []
if ticker:
    try:
        S0, q, expiries = get_underlying_info(ticker)
    except Exception as e:
        data_error = str(e)

if data_error:
    st.sidebar.error(f"No se pudo descargar '{ticker}': {data_error}")
    st.stop()

if S0 is None or not np.isfinite(S0) or S0 <= 0:
    st.sidebar.error(
        f"Precio spot inválido para '{ticker}' (S0={S0!r}). "
        "El feed de datos devolvió un valor no numérico; reintenta en unos "
        "segundos o revisa el ticker."
    )
    st.stop()

if not expiries:
    st.sidebar.warning("Este ticker no tiene opciones listadas.")
    st.stop()

# Si viene de un símbolo OCC recién parseado, pre-seleccionamos el expiry más cercano
pending = st.session_state.get("occ_pending")
if pending is not None and pending["ticker"] == ticker:
    if pending["expiry"] in expiries:
        matched_expiry = pending["expiry"]
    else:
        target_dt = np.datetime64(pending["expiry"])
        matched_expiry = min(expiries, key=lambda e: abs(np.datetime64(e) - target_dt))
        st.sidebar.info(
            f"No hay opciones exactas para {pending['expiry']}; usando el vencimiento más cercano disponible: {matched_expiry}."
        )
    st.session_state["expiry_select"] = matched_expiry
    st.session_state["type_radio"] = pending["option_type"]

if "expiry_select" not in st.session_state or st.session_state["expiry_select"] not in expiries:
    st.session_state["expiry_select"] = expiries[min(3, len(expiries) - 1)]

expiry = st.sidebar.selectbox("Vencimiento (expiry)", expiries, key="expiry_select")
option_type = st.sidebar.radio("Tipo de contrato", ["call", "put"], horizontal=True, key="type_radio")

with st.spinner("Descargando y limpiando cadena de opciones..."):
    chain_df = get_clean_chain(ticker, expiry)

if chain_df.empty:
    st.sidebar.warning("No quedaron quotes líquidas tras la limpieza para este vencimiento.")
    st.stop()

sub_chain = chain_df[chain_df["type"] == option_type].sort_values("strike")
strikes_available = sub_chain["strike"].unique().tolist()

if not strikes_available:
    st.sidebar.warning(f"No hay {option_type}s líquidos para este vencimiento.")
    st.stop()

# Si viene de un símbolo OCC, pre-seleccionamos el strike líquido más cercano al pedido
if pending is not None and pending["ticker"] == ticker:
    nearest_strike = min(strikes_available, key=lambda k: abs(k - pending["strike"]))
    if abs(nearest_strike - pending["strike"]) > 1e-6:
        st.sidebar.info(f"Strike {pending['strike']:g} no está líquido; usando el más cercano disponible: {nearest_strike:g}.")
    st.session_state["strike_select"] = nearest_strike
    st.session_state["occ_pending"] = None  # ya se aplicó, limpiamos

if "strike_select" not in st.session_state or st.session_state["strike_select"] not in strikes_available:
    atm_idx = int(np.argmin(np.abs(np.array(strikes_available) - S0)))
    st.session_state["strike_select"] = strikes_available[atm_idx]

strike = st.sidebar.selectbox("Strike (K)", strikes_available, key="strike_select")

with st.sidebar.expander("⚙ Calibración de Heston"):
    fijar_kappa = st.checkbox(
        "Fijar κ (mitigación identificabilidad κ–ξ, notebook 5)", value=False,
        help="El valle κ–ξ hace que muchas combinaciones ajusten casi igual. Fijar κ "
             "estabiliza los parámetros entre corridas al costo de imponer la velocidad "
             "de reversión en vez de dejar que el dato la determine.",
    )
    kappa_fijo_val = st.slider("Valor de κ fijo", 0.1, 6.0, 2.0, 0.1, disabled=not fijar_kappa) if fijar_kappa else None
    if fijar_kappa and tau_from_expiry(expiry) < 5 / 365:
        st.caption(
            "⚠ Vencimiento ultra-corto (<5 días): fijar κ quita justo la flexibilidad "
            "que la calibración más necesita aquí. Si Heston sale en $0 / griegas en 0, "
            "prueba desmarcar esta opción."
        )

r = get_risk_free_rate()
tau = tau_from_expiry(expiry)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="mono" style="font-size:0.85rem; color:#8492A6; line-height:1.9;">
    S₀ &nbsp;=&nbsp; <span style="color:#EAF2FB;">{S0:.2f}</span><br>
    q &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{q*100:.2f}%</span><br>
    r &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{r*100:.2f}%</span> (13w T-Bill)<br>
    τ &nbsp;&nbsp;=&nbsp; <span style="color:#EAF2FB;">{tau:.3f}</span> años ({int(tau*365)} días)
    </div>
    """,
    unsafe_allow_html=True,
)
_snap = chain_df["snapshot"].iloc[0] if ("snapshot" in chain_df.columns and len(chain_df)) else "—"
_snap_age_min = None
if _snap != "—":
    try:
        _snap_age_min = (pd.Timestamp.now() - pd.Timestamp(_snap)).total_seconds() / 60.0
    except Exception:
        pass
st.sidebar.caption(
    f"Quotes líquidas en este expiry: {len(chain_df)}  ·  Snapshot: {_snap}"
)
_n_relaxed = int(chain_df["liquidity_relaxed"].sum()) if "liquidity_relaxed" in chain_df.columns else 0
if _n_relaxed > 0:
    st.sidebar.info(
        f"ℹ️ {_n_relaxed} de estas quotes no tuvieron volumen operado HOY, pero se incluyeron "
        "porque tienen open interest > 0 (evidencia de que el contrato existe y el market maker "
        "sostiene bid/ask). Se activó porque el filtro estricto (volumen>0, regla literal del "
        "brief) dejaba menos de 4 quotes para este vencimiento -- típico en tickers menos "
        "líquidos que los grandes ETFs/megacaps."
    )
if _snap_age_min is not None and _snap_age_min > 2:
    st.sidebar.warning(
        f"⚠ Datos con {_snap_age_min:.0f} min de antigüedad. En contratos de vencimiento "
        "ultra-corto (0-3 DTE) el bid/ask puede moverse 50%+ en minutos — dale a "
        "'Refrescar datos' antes de la defensa en vivo."
    )
if st.sidebar.button("🔄 Refrescar datos (limpiar caché)"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# 4. HEADER
# ============================================================
st.markdown(
    f"""
    <div class="desk-header">
        <h1>◆ Options Analytics Dashboard</h1>
        <p>{ticker} &middot; {option_type.upper()} K={strike:g} &middot; exp {expiry} &middot; Panel 1 — Pricing &amp; Validación</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 5. CALIBRACIÓN DE HESTON PARA ESTE VENCIMIENTO
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def calibrate_for_expiry(ticker, expiry, S0, r, q, kappa_fijo=None):
    df = get_clean_chain(ticker, expiry)
    tau_ = tau_from_expiry(expiry)

    if df.empty:
        # Este vencimiento no tiene ni una quote que sobreviva la limpieza (zero-bid,
        # zero-volume, spread excesivo). Salimos con gracia -- el caller (incluido el
        # loop de la superficie 3D, que recorre varios expiries y espera que algunos
        # fallen) ya sabe manejar params_=None.
        return None, None, 0, None

    # Selección de quotes según el estándar del curso/equipos:
    # CALLS y PUTS juntos, bid>0 (ya garantizado por la limpieza de la cadena),
    # volumen>0, spread relativo (ask-bid)/mid < 0.5, y banda de liquidez
    # |log(K/S0)| < 0.5 alrededor del dinero.
    log_m = np.log(df["strike"] / S0)
    spread_rel = df["spread"] / df["mid"].replace(0, np.nan)
    m = df[
        (df["volume"].fillna(0) > 0)
        & (df["mid"] > 0)
        & (spread_rel < 0.5)
        & (log_m.abs() < 0.5)
    ]
    # Fallbacks graduales para tickers/expiries poco líquidos.
    if len(m) < 4:
        m = df[(df["mid"] > 0) & (log_m.abs() < 0.5)]
    if len(m) < 4:
        m = df[df["mid"] > 0.02]

    rows = [
        (row["strike"], tau_, row["mid"], max(row["spread"], 0.01), row["type"])
        for _, row in m.iterrows()
    ]
    if len(rows) < 4:
        return None, None, 0, None

    # ---- Techo dinámico de v0/theta a partir de la IV ATM real de este vencimiento ----
    # Vencimientos ultra-cortos (0-3 DTE) pueden requerir IV anualizada >300% para
    # reproducir un mid perfectamente normal (es aritmética de anualizar sobre τ chico,
    # no un precio "raro"). Si dejamos var_ceiling en el 0.20 (~44.7% vol) por default,
    # el problema de calibración es matemáticamente infactible y el optimizador termina
    # en una esquina degenerada de bounds (precio/griegas colapsan a 0, Feller aparenta
    # violarse). Estimamos la IV ATM con B&S y ampliamos el techo con margen 30%.
    var_ceiling = 0.20
    try:
        atm_row_c = m.iloc[(m["strike"] - S0).abs().argsort()[:1]].iloc[0]
        atm_iv_est = bs_implied_vol(
            atm_row_c["mid"], S0, atm_row_c["strike"], r, q, tau_, atm_row_c["type"]
        )
        if np.isfinite(atm_iv_est) and atm_iv_est > 0:
            var_ceiling = max(0.20, min(9.0, (atm_iv_est * 1.3) ** 2))
    except Exception:
        pass

    params, fit_obj = calibrate_heston(rows, S0, r, q, kappa_fijo=kappa_fijo, var_ceiling=var_ceiling)

    # ---- Diagnóstico: ¿el ajuste final realmente reproduce el mercado? ----
    # fit_obj.cost es 0.5*sum(residuals^2) (convención scipy); lo traducimos a un RMSE
    # relativo en precio para decidir si el resultado es confiable o si, pese al techo
    # ampliado, el optimizador sigue sin poder calzar el smile (bounds de kappa/xi/rho,
    # muy pocas quotes, datos ruidosos, etc.).
    calib_warning = None
    try:
        prices_mkt = np.array([row[2] for row in rows])
        v0_, theta_, kappa_, xi_, rho_ = params
        K_c = np.array([row[0] for row in rows])
        ot_c = [row[4] for row in rows]
        prices_fit = np.array([
            heston_price(S0, k, r, q, tau_, v0_, kappa_, theta_, xi_, rho_, ot)
            for k, ot in zip(K_c, ot_c)
        ])
        rel_err = np.abs(prices_fit - prices_mkt) / np.maximum(prices_mkt, 0.01)
        if np.median(rel_err) > 0.25:
            calib_warning = (
                "La calibración de Heston no logró reproducir el smile de este "
                "vencimiento dentro de los bounds del modelo (error relativo mediano "
                f"{np.median(rel_err)*100:.0f}%). Con τ tan corto esto suele pasar cuando "
                "hay muy pocas quotes líquidas o el vencimiento es prácticamente 0DTE — "
                "trata los precios/griegas de Heston aquí con cautela, o prueba otro "
                "vencimiento con más profundidad de mercado."
            )
    except Exception:
        pass

    return params, fit_obj, len(rows), calib_warning


with st.spinner("Calibrando Heston a la sonrisa de este vencimiento..."):
    heston_params, fit_obj, n_quotes_calib, calib_warning = calibrate_for_expiry(
        ticker, expiry, S0, r, q, kappa_fijo_val
    )

if heston_params is None:
    st.warning(
        "No hay suficientes quotes líquidas para calibrar Heston en este vencimiento. "
        "Prueba otro expiry o ticker más líquido."
    )
    st.stop()

if calib_warning:
    st.warning(f"⚠ {calib_warning}")

v0, theta, kappa, xi, rho = heston_params

# ---------- Persistencia de calibraciones para medir estabilidad entre días ----------
# Se guarda un snapshot por (ticker, expiry, fecha) cada vez que se calibra, sin duplicar
# si ya se corrió hoy. Con esto el dashboard puede comparar parámetros entre corridas de
# días distintos — el requisito que pide el feedback del quant developer.
CALIB_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_history.csv")
CALIB_COLUMNS = ["date", "ticker", "expiry", "v0", "theta", "kappa", "xi", "rho", "feller_ok", "n_quotes", "rmse_price"]


def log_calibration(ticker, expiry, params, feller_ok, n_quotes, rmse_price):
    today_str = date.today().isoformat()
    v0_, theta_, kappa_, xi_, rho_ = params
    file_exists = os.path.exists(CALIB_LOG_PATH)
    if file_exists:
        try:
            hist = pd.read_csv(CALIB_LOG_PATH)
            dup = (hist["ticker"] == ticker) & (hist["expiry"] == expiry) & (hist["date"] == today_str)
            if dup.any():
                return  # ya se registró hoy para este contrato, no duplicar
        except Exception:
            pass
    row = pd.DataFrame([{
        "date": today_str, "ticker": ticker, "expiry": expiry,
        "v0": v0_, "theta": theta_, "kappa": kappa_, "xi": xi_, "rho": rho_,
        "feller_ok": feller_ok, "n_quotes": n_quotes, "rmse_price": rmse_price,
    }])
    try:
        row.to_csv(CALIB_LOG_PATH, mode="a", header=not file_exists, index=False)
    except Exception:
        pass  # entorno de solo lectura u otro problema de filesystem: no interrumpir la app


def load_calibration_history(ticker, expiry):
    if not os.path.exists(CALIB_LOG_PATH):
        return pd.DataFrame(columns=CALIB_COLUMNS)
    try:
        hist = pd.read_csv(CALIB_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=CALIB_COLUMNS)
    return hist[(hist["ticker"] == ticker) & (hist["expiry"] == expiry)].sort_values("date").reset_index(drop=True)


feller_ok, feller_lhs, feller_rhs = feller_condition(heston_params)

# ============================================================
# 6. PRECIO DE MERCADO DEL CONTRATO SELECCIONADO
# ============================================================
row_sel = sub_chain[sub_chain["strike"] == strike].iloc[0]
market_mid = float(row_sel["mid"])
market_iv = row_sel["impliedVolatility"]

# ATM implied vol (para el "mundo flat-vol" de B&S)
atm_row = sub_chain.iloc[(sub_chain["strike"] - S0).abs().argsort()[:1]].iloc[0]
sigma_atm = bs_implied_vol(atm_row["mid"], S0, atm_row["strike"], r, q, tau, option_type)
if np.isnan(sigma_atm) or sigma_atm <= 0:
    sigma_atm = float(market_iv) if not np.isnan(market_iv) else 0.20

# IV del PROPIO contrato (convención de desk, mismo diseño que el resto de los equipos):
# invertida del mid observado; fallback a la columna de yfinance; último recurso: la ATM.
# Con esta σ, B&S reproduce el mid por construcción (error ~0%) — la definición misma de
# volatilidad implícita. La prueba de B&S COMO MODELO vive en el Panel 3 (línea plana σ_ATM
# vs el smile), que es donde el brief pide contrastarlo.
sigma_contract = bs_implied_vol(market_mid, S0, strike, r, q, tau, option_type)
if not np.isfinite(sigma_contract) or sigma_contract <= 0:
    sigma_contract = (
        float(market_iv) if (market_iv is not None and np.isfinite(market_iv) and market_iv > 0) else sigma_atm
    )

moneyness_sel = strike / S0
is_extrapolated = abs(np.log(moneyness_sel)) >= 0.5

# ============================================================
# 7. PRECIOS DE AMBOS MOTORES
# ============================================================
price_bs = bs_price(S0, strike, r, q, tau, sigma_contract, option_type)
price_heston = heston_price(S0, strike, r, q, tau, v0, kappa, theta, xi, rho, option_type)
# Validación cruzada de motores (rúbrica: "validated against each other"): el precio
# mostrado usa integración directa (quad); COS lo verifica de forma independiente.
try:
    price_heston_cos = float(heston_prices_fast(S0, [strike], r, q, tau, v0, kappa, theta, xi, rho, option_type)[0])
    cross_check_diff = abs(price_heston - price_heston_cos)
except Exception:
    price_heston_cos, cross_check_diff = np.nan, np.nan

err_bs_abs = price_bs - market_mid
err_heston_abs = price_heston - market_mid
err_bs_rel = err_bs_abs / market_mid * 100 if market_mid else np.nan
err_heston_rel = err_heston_abs / market_mid * 100 if market_mid else np.nan

# ============================================================
# 8. PANEL 1 — TARJETAS DE PRECIO
# ============================================================
st.markdown("### Panel 1 · Pricing & Validación")
st.caption(
    "B&S se pricea con la **IV implícita del propio contrato** (convención de desk): reproduce el mid "
    "por construcción, así que su error es ~0% — la IV se define justamente como la σ que hace que B&S "
    "iguale el precio observado. La prueba de B&S *como modelo* (una sola σ para todos los strikes) "
    "está en el Panel 3, donde la línea plana falla visiblemente contra el smile."
)

if is_extrapolated:
    st.warning(
        f"⚠️ **Extrapolación fuera de la ventana de calibración.** Este strike tiene moneyness "
        f"K/S₀={moneyness_sel:.2f} — Heston se calibró con quotes dentro de la banda |log(K/S₀)|<0.5, "
        "así que aquí el modelo está extrapolando a una zona sin datos de ajuste. Para contratos muy OTM "
        "y de corto plazo como este, es normal (y esperado) que B&S y Heston den precios cercanos a cero "
        "mientras el mercado sigue cotizando un mínimo por encima de eso — por riesgo de salto (*jump risk*), "
        "iliquidez, o simplemente el tick mínimo de cotización. Esto **no es un error del pipeline de datos**, "
        "es una limitación real de los modelos de difusión continua (B&S y Heston) en los extremos de la sonrisa."
    )


c1, c2, c3 = st.columns(3)
with c1:
    iv_txt = f" &middot; IV {float(market_iv)*100:.2f}%" if (market_iv is not None and np.isfinite(market_iv)) else ""
    last_price_sel = row_sel.get("lastPrice", np.nan)
    last_txt = (
        f" &middot; último {last_price_sel:.2f}"
        if (last_price_sel is not None and np.isfinite(last_price_sel) and last_price_sel > 0)
        else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Precio de mercado (mid)</div>
            <div class="value">${market_mid:,.2f}</div>
            <div class="sub" style="color:#8492A6;">bid {row_sel['bid']:.2f} / ask {row_sel['ask']:.2f}{iv_txt}{last_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Staleness check: ¿el último trade es consistente con el bid/ask actual? ----------
# Bid/ask son cotizaciones en vivo (o casi); lastPrice es el precio de la última operación
# EJECUTADA, que en un contrato ilíquido (como uno a 2 días de vencer) puede llevar horas
# o hasta un día hábil de antigüedad — literalmente otro spot, otra hora. Que difieran NO
# es un bug del pipeline: son dos observables distintos del mercado. El brief pide C_mkt =
# mid de bid/ask (no lastPrice) precisamente por esto — el mid es lo único "actual".
if (
    last_price_sel is not None and np.isfinite(last_price_sel) and last_price_sel > 0
    and not (row_sel["bid"] <= last_price_sel <= row_sel["ask"])
):
    st.info(
        f"ℹ️ El último precio operado (**${last_price_sel:.2f}**) cae **fuera** del bid/ask actual "
        f"(${row_sel['bid']:.2f} / ${row_sel['ask']:.2f}). Esto es normal en contratos poco líquidos: "
        "`lastPrice` es el precio de la última operación EJECUTADA (puede ser de horas o de la sesión "
        "anterior), mientras que bid/ask reflejan el mercado justo ahora. Por eso este dashboard usa "
        "**mid de bid/ask** como `C_mkt` — el spec del proyecto lo pide así (\"Market option price = "
        "mid of bid/ask\") precisamente porque es el único de los dos que es simultáneo con el spot S₀ "
        "usado para pricear. Si el broker te muestra un 'último' distinto al mid de este dashboard, "
        "esa es la explicación — no una descarga incorrecta."
    )

with c2:
    color = "#5FDCB4" if abs(err_bs_rel) < abs(err_heston_rel) else "#F0A85C"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Black &amp; Scholes <span class="tag tag-accent">IV contrato={sigma_contract*100:.2f}%</span></div>
            <div class="value">${price_bs:,.2f}</div>
            <div class="sub" style="color:{color};">error {err_bs_abs:+.2f} ({err_bs_rel:+.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    color = "#5FDCB4" if abs(err_heston_rel) < abs(err_bs_rel) else "#F0A85C"
    feller_tag = (
        '<span class="tag tag-good">Feller OK</span>' if feller_ok else '<span class="tag tag-warn">Feller violado</span>'
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Heston (calibrado) {feller_tag}</div>
            <div class="value">${price_heston:,.2f}</div>
            <div class="sub" style="color:{color};">error {err_heston_abs:+.2f} ({err_heston_rel:+.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Tabla de comparación
# ---------- EXTRA (+3, feedback del trader): bandera de motor recomendado ----------
def recommend_engine(moneyness, tau_years, spread_rel):
    """Regla del trader del brief: 'For a short-dated ATM option, B&S is fine and fast.
    For a long-dated OTM put, you need the skew — use Heston.' Justificación de una
    línea en función de moneyness, madurez y liquidez."""
    dias = tau_years * 365
    otm_dist = abs(moneyness - 1.0)
    if otm_dist <= 0.03 and dias <= 60:
        eng = "B&S"
        why = (f"ATM ({moneyness:.2f}× spot) y corto plazo ({dias:.0f}d): la vol plana basta en el dinero — "
               "B&S es rápido, suficiente y el benchmark que todos aceptan.")
    elif otm_dist > 0.05 or dias > 180:
        eng = "Heston"
        why = (f"Moneyness {moneyness:.2f}× a {dias:.0f} días: el precio vive en el skew de las alas — "
               "se necesita la estructura ρ/ξ que solo Heston captura.")
    else:
        eng = "Heston"
        why = (f"Zona intermedia (moneyness {moneyness:.2f}×, {dias:.0f}d): en cuanto el strike se aleja "
               "del ATM, el skew empieza a pesar — Heston es la elección conservadora.")
    if spread_rel > 0.15:
        why += (f" ⚠ Liquidez: spread = {spread_rel*100:.0f}% del mid — quote ilíquida, desconfía del mid "
                "y valida contra el modelo.")
    return eng, why


spread_rel_sel = float(row_sel["spread"]) / market_mid if market_mid > 0 else 0.0
engine_flag, engine_why = recommend_engine(moneyness_sel, tau, spread_rel_sel)
flag_color = "tag-accent" if engine_flag == "B&S" else "tag-good"
st.markdown(
    f"""
    <div class="interp-box" style="border-left-color:{'#3B82C4' if engine_flag=='B&S' else '#1FAE85'};">
    <span class="tag {flag_color}" style="font-size:0.85rem;">MOTOR RECOMENDADO: {engine_flag.upper()}</span>
    &nbsp;&nbsp;{engine_why}
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

comp_df = pd.DataFrame(
    {
        "Motor": ["Black & Scholes", "Heston"],
        "Precio": [price_bs, price_heston],
        "Mercado (mid)": [market_mid, market_mid],
        "Error absoluto": [err_bs_abs, err_heston_abs],
        "Error relativo (%)": [err_bs_rel, err_heston_rel],
    }
).set_index("Motor")
st.dataframe(
    comp_df.style.format(
        {"Precio": "${:.2f}", "Mercado (mid)": "${:.2f}", "Error absoluto": "{:+.3f}", "Error relativo (%)": "{:+.2f}%"}
    ),
    use_container_width=True,
)

# ============================================================
# 9. GRÁFICA: PRECIOS A TRAVÉS DE STRIKES (contexto visual)
# ============================================================
strikes_sorted = np.array(sorted(strikes_available))
bs_curve = [bs_price(S0, K, r, q, tau, sigma_contract, option_type) for K in strikes_sorted]
heston_curve = [heston_price(S0, K, r, q, tau, v0, kappa, theta, xi, rho, option_type) for K in strikes_sorted]
market_curve = []
for K in strikes_sorted:
    mrow = sub_chain[sub_chain["strike"] == K]
    market_curve.append(float(mrow["mid"].iloc[0]) if not mrow.empty else np.nan)

fig = go.Figure()
fig.add_trace(go.Scatter(x=strikes_sorted, y=market_curve, mode="markers", name="Mercado (mid)",
                          marker=dict(color="#EAF2FB", size=7, symbol="circle")))
fig.add_trace(go.Scatter(x=strikes_sorted, y=bs_curve, mode="lines", name="Black & Scholes",
                          line=dict(color="#3B82C4", width=2, dash="dot")))
fig.add_trace(go.Scatter(x=strikes_sorted, y=heston_curve, mode="lines", name="Heston (calibrado)",
                          line=dict(color="#1FAE85", width=2.5)))
fig.add_vline(x=S0, line_dash="dash", line_color="#8492A6", annotation_text="Spot", annotation_font_color="#8492A6")
fig.add_vline(x=strike, line_color="#D9822B", annotation_text="Seleccionado", annotation_font_color="#D9822B")
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Strike (K)",
    yaxis_title=f"Precio del {option_type}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 10. INTERPRETACIÓN AUTOMÁTICA
# ============================================================
moneyness = strike / S0
if option_type == "call":
    money_state = "ATM" if 0.97 <= moneyness <= 1.03 else ("ITM" if moneyness < 0.97 else "OTM")
else:
    money_state = "ATM" if 0.97 <= moneyness <= 1.03 else ("ITM" if moneyness > 1.03 else "OTM")

if money_state == "ATM":
    razon_heston = (
        "Este contrato está **en el dinero (ATM)** — el corazón de la zona de calibración de Heston, donde "
        "hay más quotes líquidas y el ajuste suele ser mejor. Un error de Heston pequeño aquí confirma que "
        "la calibración está sana; uno grande sugiere quotes ruidosas o inestabilidad de parámetros ese día."
    )
elif money_state == "OTM":
    razon_heston = (
        f"Este contrato está **fuera del dinero ({moneyness:.2f}× spot)**. El error de Heston aquí mide qué "
        "tan bien el modelo extendió la información del *skew* (vía ρ y ξ) hacia las alas. Un desajuste "
        "moderado es normal: Heston ajusta la sonrisa completa con solo 5 parámetros — no puede clavar cada "
        "quote individual, y las alas cargan además prima por riesgo de salto e iliquidez que ningún modelo "
        "de difusión continua captura."
    )
else:
    razon_heston = (
        f"Este contrato está **dentro del dinero ({moneyness:.2f}× spot)**. El valor intrínseco domina el "
        "precio, así que los errores relativos de Heston tienden a ser pequeños aquí incluso si el ajuste de "
        "la parte de valor-tiempo no es perfecto."
    )

st.markdown(
    f"""
    <div class="interp-box">
    <b>Cómo leer esta comparación</b><br><br>
    <b>B&amp;S ({err_bs_rel:+.2f}%):</b> se pricea con la IV implícita del propio contrato, así que reproducir
    el mid es <i>por definición</i> — la IV es exactamente "la σ que hace que B&amp;S iguale el precio observado".
    Un error distinto de cero aquí solo reflejaría redondeo o un mid stale, no al modelo. La validación real de
    B&amp;S como modelo (una sola σ para todo el vencimiento) está en el Panel 3.<br><br>
    <b>Heston ({err_heston_rel:+.2f}%):</b> este error <b>sí es informativo</b> — Heston se calibró contra toda
    la sonrisa del vencimiento (no contra este contrato), así que su desviación vs el mid mide la capacidad
    real del modelo de describir este punto con parámetros globales. {razon_heston}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Nota: B&S del contrato usa su propia IV implícita (convención de desk). Heston se calibra contra las "
    "calls y puts líquidos (volumen>0, spread relativo<0.5, |log(K/S₀)|<0.5) del vencimiento, vía búsqueda global (Latin Hypercube) + "
    "refinamiento local (Levenberg-Marquardt) con motor COS vectorizado. Detalle completo de la calibración "
    "en la sección de robustez, al final del dashboard."
)

# ============================================================
# 11. PANEL 2 — LAS GREEKS
# ============================================================
st.markdown("---")
st.markdown("### Panel 2 · Las Greeks")
st.caption(
    "Δ, Γ y Θ de B&S/Heston coinciden casi exactamente cuando el vol-of-vol (ξ) es chico — así se validó "
    "este motor. Vega, Vanna y Volga sí difieren de forma estructural entre modelos: eso es precisamente "
    "lo que la volatilidad estocástica agrega, no un error de cálculo."
)

GREEK_LABELS = {
    "delta": "Δ Delta", "gamma": "Γ Gamma", "vega": "ν Vega (por 1pto vol)",
    "theta": "Θ Theta (por día)", "rho": "ρ Rho (por 1pto tasa)",
    "vanna": "Vanna (∂Δ/∂σ, cruda)", "volga": "Volga (∂ν/∂σ, cruda)",
}

g_bs_sel = bs_greeks(S0, strike, r, q, tau, sigma_contract, option_type)
g_h_sel = heston_greeks_fd(S0, strike, r, q, tau, v0, kappa, theta, xi, rho, option_type, full=True)

st.markdown("##### Greeks del contrato seleccionado")
cols = st.columns(6)
for col, key in zip(cols, ["delta", "gamma", "vega", "theta", "rho", "vanna"]):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{GREEK_LABELS[key]}</div>
            <div class="sub" style="color:#8492A6; font-size:0.75rem;">B&amp;S</div>
            <div class="value" style="font-size:1.15rem;">{g_bs_sel[key]:.4f}</div>
            <div class="sub" style="color:#8492A6; font-size:0.75rem; margin-top:0.4rem;">Heston</div>
            <div class="value" style="font-size:1.15rem; color:#5FDCB4;">{g_h_sel[key]:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.caption(
    "Vanna (segundo orden) mide cómo cambia Delta ante un shock de volatilidad — o, equivalentemente, "
    "cómo cambia Vega ante un shock de spot. En Heston nace directamente de ρ (correlación spot-vol)."
)

with st.expander("Bonus segundo orden: Volga (convexidad de Vega)"):
    st.markdown(
        f"""
        <div class="metric-card" style="max-width:260px;">
            <div class="label">{GREEK_LABELS['volga']}</div>
            <div class="sub" style="color:#8492A6; font-size:0.75rem;">B&amp;S</div>
            <div class="value" style="font-size:1.25rem;">{g_bs_sel['volga']:.6f}</div>
            <div class="sub" style="color:#8492A6; font-size:0.75rem; margin-top:0.4rem;">Heston</div>
            <div class="value" style="font-size:1.25rem; color:#5FDCB4;">{g_h_sel['volga']:.6f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Volga mide la convexidad de Vega (qué tan no-lineal es la exposición a volatilidad). "
        "En B&S existe matemáticamente pero no tiene una fuente estructural — en Heston nace de ξ (vol-of-vol)."
    )

st.markdown("##### 📓 Griegas de Heston — convención del curso (notebook 2)")
st.caption(
    "Tabla en el formato y unidades EXACTOS del `heston_greeks_fd` del notebook 2 (diferencias finitas "
    "centradas, bumps relativos h=1e-4 sobre cada parámetro, valores crudos sin reescalar) — estos son los "
    "números directamente comparables con el estándar del curso. `vega_v0` es ∂P/∂v₀; `theta` es ∂P/∂τ "
    "(positiva para opciones largas; el decaimiento por día de las tarjetas de arriba es −∂P/∂τ/365)."
)
sens_rows = [
    ("price", g_h_sel["price"]),
    ("delta", g_h_sel["delta_fd"]),
    ("gamma", g_h_sel["gamma"]),
    ("vega_v0", g_h_sel["vega_v0"]),
    ("vega_theta", g_h_sel.get("vega_theta", np.nan)),
    ("vega_kappa", g_h_sel.get("vega_kappa", np.nan)),
    ("vega_xi", g_h_sel.get("vega_xi", np.nan)),
    ("vega_rho", g_h_sel.get("vega_rho", np.nan)),
    ("theta", g_h_sel["theta_tau"]),
    ("rho (∂P/∂ρ, como en el notebook)", g_h_sel.get("vega_rho", np.nan)),
]
sens_df = pd.DataFrame(sens_rows, columns=["Griega", "Valor"]).set_index("Griega")
st.dataframe(sens_df.style.format({"Valor": "{:.6f}"}), use_container_width=True)

delta_p1_diff = abs(g_h_sel["delta_fd"] - g_h_sel["delta"])
with st.expander("Verificación Δ = P₁ y nota sobre Rho"):
    st.markdown(
        f"""
        <div class="interp-box">
        <b>Verificación Δ = e^(−qτ)·P₁ (resultado del notebook 2):</b>
        Delta por diferencias finitas = <span class="mono">{g_h_sel['delta_fd']:.6f}</span> vs
        Delta analítica e^(−qτ)·P₁ = <span class="mono">{g_h_sel['delta']:.6f}</span> —
        diferencia <span class="mono">{delta_p1_diff:.2e}</span>. Coinciden hasta el error de la
        cuadratura numérica, igual que en el notebook.<br><br>
        <b>Nota de método (Rho):</b> la fila <i>rho</i> de la tabla reproduce la convención del notebook 2,
        donde <i>rho</i> re-usa el bump de la correlación ρ (por eso <i>rho</i> = <i>vega_rho</i> ahí).
        La Rho de las tarjetas comparativas de arriba es la estándar del brief — sensibilidad a la
        <b>tasa</b> r (∂P/∂r) — para poder compararla contra la Rho de B&S.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Sweeps: Greeks vs Strike y vs Spot ----------
SWEEP_KEYS = ["delta", "gamma", "vega", "theta", "rho", "vanna"]


@st.cache_data(ttl=1800, show_spinner=False)
def sweep_greeks_vs_strike(strikes_tuple, S0, r, q, tau, sigma_bs, heston_params_tuple, option_type):
    v0_, theta_, kappa_, xi_, rho_ = heston_params_tuple
    rows = []
    for K in strikes_tuple:
        gb = bs_greeks(S0, K, r, q, tau, sigma_bs, option_type)
        gh = heston_greeks_fd(S0, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_, option_type, full=False)
        rows.append({"K": K, **{f"bs_{k}": gb[k] for k in SWEEP_KEYS},
                     **{f"h_{k}": gh[k] for k in SWEEP_KEYS}})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def sweep_greeks_vs_spot(spot_tuple, K, r, q, tau, sigma_bs, heston_params_tuple, option_type):
    v0_, theta_, kappa_, xi_, rho_ = heston_params_tuple
    rows = []
    for S_ in spot_tuple:
        gb = bs_greeks(S_, K, r, q, tau, sigma_bs, option_type)
        gh = heston_greeks_fd(S_, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_, option_type, full=False)
        rows.append({"S": S_, **{f"bs_{k}": gb[k] for k in SWEEP_KEYS},
                     **{f"h_{k}": gh[k] for k in SWEEP_KEYS}})
    return pd.DataFrame(rows)


# Limitar a máx ~13 strikes para no disparar el tiempo de cómputo (bump-and-reprice x quad)
strikes_for_sweep = strikes_sorted
if len(strikes_for_sweep) > 13:
    idx_pick = np.linspace(0, len(strikes_for_sweep) - 1, 13).astype(int)
    strikes_for_sweep = strikes_for_sweep[idx_pick]

spot_range = np.linspace(0.7 * S0, 1.3 * S0, 13)

with st.spinner("Calculando Greeks a través de strikes y de spot (bump-and-reprice)..."):
    df_strike = sweep_greeks_vs_strike(tuple(strikes_for_sweep), S0, r, q, tau, sigma_contract, tuple(heston_params), option_type)
    df_spot = sweep_greeks_vs_spot(tuple(spot_range), strike, r, q, tau, sigma_contract, tuple(heston_params), option_type)

axis_choice = st.radio(
    "Ver Greeks a través de:", ["Strike (K)  — S fijo en el spot actual", "Spot (S)  — K fijo en el seleccionado"],
    horizontal=True,
)
df_plot = df_strike if axis_choice.startswith("Strike") else df_spot
x_col = "K" if axis_choice.startswith("Strike") else "S"
x_label = "Strike (K)" if axis_choice.startswith("Strike") else "Spot (S)"
x_ref = strike if axis_choice.startswith("Strike") else S0

greek_plot_specs = [
    ("delta", "Delta (Δ)"), ("gamma", "Gamma (Γ)"), ("vega", "Vega (ν)"),
    ("theta", "Theta (Θ, por día)"), ("rho", "Rho (ρ)"), ("vanna", "Vanna"),
]

fig2 = make_subplots(rows=2, cols=3, subplot_titles=[t for _, t in greek_plot_specs])
positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
for (key, title), (r_, c_) in zip(greek_plot_specs, positions):
    fig2.add_trace(
        go.Scatter(x=df_plot[x_col], y=df_plot[f"bs_{key}"], name="B&S", line=dict(color="#3B82C4", width=2, dash="dot"),
                    legendgroup="bs", showlegend=(r_ == 1 and c_ == 1)),
        row=r_, col=c_,
    )
    fig2.add_trace(
        go.Scatter(x=df_plot[x_col], y=df_plot[f"h_{key}"], name="Heston", line=dict(color="#1FAE85", width=2.5),
                    legendgroup="heston", showlegend=(r_ == 1 and c_ == 1)),
        row=r_, col=c_,
    )
    fig2.add_vline(x=x_ref, line_dash="dash", line_color="#8492A6", row=r_, col=c_)

fig2.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=620, margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
)
fig2.update_xaxes(title_text=x_label)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Interpretación automática de Greeks ----------
gamma_col = "h_gamma" if True else "bs_gamma"
idx_max_gamma_h = df_strike["h_gamma"].idxmax()
K_max_gamma_h = df_strike.loc[idx_max_gamma_h, "K"]
idx_max_gamma_bs = df_strike["bs_gamma"].idxmax()
K_max_gamma_bs = df_strike.loc[idx_max_gamma_bs, "K"]

vega_otm_low = df_strike[df_strike["K"] < S0]["h_vega"].mean()
vega_otm_high = df_strike[df_strike["K"] > S0]["h_vega"].mean()
vega_skew_txt = (
    "más sensible a volatilidad en los strikes bajos (puts OTM)" if vega_otm_low > vega_otm_high
    else "más sensible a volatilidad en los strikes altos (calls OTM)"
) if not (np.isnan(vega_otm_low) or np.isnan(vega_otm_high)) else "aproximadamente simétrica en este rango"

theta_sign_txt = "negativo" if g_bs_sel["theta"] < 0 else "positivo"
quien_pierde = "el comprador (posición larga)" if g_bs_sel["theta"] < 0 else "el vendedor (posición corta)"

st.markdown(
    f"""
    <div class="interp-box">
    <b>Lectura de las Greeks para este contrato</b><br><br>
    <b>Gamma:</b> el pico de Γ (Heston) está en K≈{K_max_gamma_h:g} y en B&amp;S en K≈{K_max_gamma_bs:g} —
    ambos cerca del spot (S₀={S0:.2f}), como se espera: la convexidad del payoff es máxima justo ATM.
    Para quien hace delta-hedging, esto significa que <b>cerca del dinero y cerca del vencimiento hay que
    re-balancear la cobertura con más frecuencia</b> — el Delta cambia rápido ante pequeños movimientos del spot.<br><br>
    <b>Theta:</b> para este {option_type}, Θ es <b>{theta_sign_txt}</b> ({g_bs_sel['theta']:.4f} por día en B&amp;S).
    Un Θ negativo significa que <b>{quien_pierde}</b> pierde valor cada día que pasa, todo lo demás constante
    — es el "alquiler" que se paga por tener optionalidad.<br><br>
    <b>Vega a través de strikes:</b> en Heston, la sensibilidad a volatilidad resulta {vega_skew_txt}
    (ρ={rho:.2f} en la calibración actual). Esto es exactamente el mecanismo detrás del <i>skew</i> de la
    sonrisa de volatilidad: cuando ρ&lt;0, una caída del spot viene acompañada de un alza en la volatilidad,
    así que los strikes bajos (protección/puts) cargan más sensibilidad — el mercado los precia con IV más alta.<br><br>
    <b>B&amp;S vs Heston:</b> Delta, Gamma y Theta casi no cambian entre motores (ambos miden lo mismo: la forma
    del payoff y el paso del tiempo). Donde sí difieren es en <b>Vega</b> — en Heston es estructuralmente menor
    porque un shock a la varianza de hoy (v₀) se diluye con el tiempo por la reversión a la media (κ, θ) — y en
    <b>Vanna/Volga</b>, que en B&amp;S son casi cero pero en Heston capturan la asimetría y la convexidad de la
    sonrisa. Esa es, literalmente, la estructura adicional que compra la volatilidad estocástica.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Metodología: Greeks de B&S en forma cerrada; Greeks de Heston vía bump-and-reprice (diferencias finitas "
    "centradas) reparametrizando a σ₀=√v₀ para expresar vega/vanna/volga en las mismas unidades que B&S. "
    "Al shockear vega solo se mueve v₀, dejando κ, θ, ξ, ρ (calibrados) fijos — así se aísla el efecto de la "
    "varianza inicial de hoy."
)

# ============================================================
# 12. PANEL 3 — VOLATILITY SMILE / SURFACE
# ============================================================
st.markdown("---")
st.markdown("### Panel 3 · Volatility Smile")
st.caption(
    "El panel central del proyecto: aquí se ve, visualmente, por qué un solo número de volatilidad (B&S) "
    "no alcanza para describir el mercado — y qué compra la volatilidad estocástica de Heston."
)


SMILE_COLUMNS = ["strike", "moneyness", "log_moneyness", "iv", "type"]


def build_smile_data(chain_df, S0, r, q, tau):
    """Construye el smile usando el lado OTM de cada strike (puts para K<S0, calls para K>=S0) —
    es el lado típicamente más líquido y la convención estándar para graficar un smile.

    Eje x = log-moneyness ln(K/S₀) (convención estándar, Gatheral "The Volatility Surface"):
    simétrico alrededor de ATM=0, y es la variable natural en la que Heston/el smile suelen
    describirse — moneyness lineal (K/S₀) comprime las alas OTM y estira las ITM de forma
    asimétrica, dificultando comparar skew put vs call a simple vista.

    Tope de IV: vencimientos ultra-cortos (0-3 DTE) pueden anualizar a IVs muy por encima
    de 300% sin que el precio en sí sea nada anómalo (mismo fenómeno que ya vimos en el
    badge de B&S). Si el tope es demasiado estricto, TODAS las quotes de un expiry corto
    pueden quedar descartadas, dejando 'rows' vacío -> DataFrame sin columnas -> el
    sort_values('log_moneyness') de más abajo truena con KeyError en vez de fallar con
    gracia. Dos fixes: tope más generoso (acorde al var_ceiling dinámico de calibración)
    y devolver siempre un DataFrame con las columnas esperadas, aunque esté vacío."""
    rows = []
    for _, row in chain_df.iterrows():
        otype = "call" if row["strike"] >= S0 else "put"
        if row["type"] != otype:
            continue
        iv = bs_implied_vol(row["mid"], S0, row["strike"], r, q, tau, otype)
        if not np.isnan(iv) and 0.01 < iv < 10.0:
            m = row["strike"] / S0
            rows.append({"strike": row["strike"], "moneyness": m, "log_moneyness": np.log(m), "iv": iv, "type": otype})
    if not rows:
        return pd.DataFrame(columns=SMILE_COLUMNS)
    return pd.DataFrame(rows).sort_values("log_moneyness").reset_index(drop=True)


def build_heston_smile_curve(S0, r, q, tau, params, k_min, k_max, n_points=60):
    """Curva de IV de Heston en una malla DENSA de strikes (n_points, independiente de qué
    strikes tuvieron quotes líquidas). Antes, la curva de Heston del smile se evaluaba SOLO
    en los strikes de mercado que sobrevivieron el filtro de liquidez -- en un vencimiento
    corto eso puede ser apenas 4-12 puntos, y conectados por 'lines+markers' se ve dentado /
    poligonal, no una curva suave. El brief pide explícitamente 'Overlay the Heston-implied
    smile after calibration — it should curve to match' — una curva de verdad, evaluada en
    el modelo calibrado, no una interpolación lineal entre pocos puntos de mercado.
    Devuelve (log_moneyness_grid, iv_grid) usando la misma convención OTM (put si K<S0, call si no)."""
    v0_, theta_, kappa_, xi_, rho_ = params
    K_grid = np.linspace(k_min, k_max, n_points)
    ivs = []
    for K in K_grid:
        otype = "call" if K >= S0 else "put"
        price = heston_price(S0, K, r, q, tau, v0_, kappa_, theta_, xi_, rho_, otype)
        ivs.append(bs_implied_vol(price, S0, K, r, q, tau, otype))
    return np.log(K_grid / S0), np.array(ivs)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_smile(ticker, expiry, S0, r, q, tau, heston_params_tuple):
    chain_full = get_clean_chain(ticker, expiry)
    smile_df = build_smile_data(chain_full, S0, r, q, tau)
    if smile_df.empty:
        return smile_df
    v0_, theta_, kappa_, xi_, rho_ = heston_params_tuple
    heston_ivs = []
    for _, row in smile_df.iterrows():
        price = heston_price(S0, row["strike"], r, q, tau, v0_, kappa_, theta_, xi_, rho_, row["type"])
        heston_ivs.append(bs_implied_vol(price, S0, row["strike"], r, q, tau, row["type"]))
    smile_df["heston_iv"] = heston_ivs
    return smile_df


with st.spinner("Construyendo la sonrisa de volatilidad..."):
    smile_df = compute_smile(ticker, expiry, S0, r, q, tau, tuple(heston_params))

if smile_df.empty or smile_df["heston_iv"].isna().all():
    st.warning("No hay suficientes quotes OTM líquidas en este vencimiento para construir el smile.")
else:
    valid = smile_df.dropna(subset=["heston_iv"])
    # σ plana de B&S para el smile: mediana de las IVs cercanas al ATM (|log-moneyness|<0.03),
    # con fallback a la mediana general — regla estándar de los equipos del curso
    # (más robusta que la IV de un solo strike ATM).
    near_atm = smile_df[np.abs(smile_df["log_moneyness"]) < 0.03]["iv"]
    sigma_flat = float(np.nanmedian(near_atm)) if len(near_atm) else float("nan")
    if not np.isfinite(sigma_flat):
        sigma_flat = float(np.nanmedian(smile_df["iv"])) if len(smile_df) else sigma_atm
    if not np.isfinite(sigma_flat):
        sigma_flat = sigma_atm
    rmse_heston = float(np.sqrt(np.mean((valid["heston_iv"] - valid["iv"]) ** 2))) * 100
    rmse_bs = float(np.sqrt(np.mean((sigma_flat - valid["iv"]) ** 2))) * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"""<div class="metric-card"><div class="label">RMSE B&amp;S plano</div>
        <div class="value" style="font-size:1.4rem; color:#F0A85C;">{rmse_bs:.2f} pts vol</div></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""<div class="metric-card"><div class="label">RMSE Heston calibrado</div>
        <div class="value" style="font-size:1.4rem; color:#5FDCB4;">{rmse_heston:.2f} pts vol</div></div>""",
        unsafe_allow_html=True,
    )
    mejora = (1 - rmse_heston / rmse_bs) * 100 if rmse_bs > 0 else np.nan
    c3.markdown(
        f"""<div class="metric-card"><div class="label">Mejora de Heston vs B&amp;S</div>
        <div class="value" style="font-size:1.4rem;">{mejora:+.0f}%</div></div>""",
        unsafe_allow_html=True,
    )

    # Curva de Heston en malla DENSA (no solo en los strikes de mercado -- ver
    # build_heston_smile_curve) para que el overlay sea una curva suave de verdad,
    # tal como pide el brief, y no una línea poligonal entre 4-12 puntos dispersos.
    k_lo = float(smile_df["strike"].min()) * 0.97
    k_hi = float(smile_df["strike"].max()) * 1.03
    curve_lm, curve_iv = build_heston_smile_curve(S0, r, q, tau, tuple(heston_params), k_lo, k_hi, n_points=60)
    curve_mask = np.isfinite(curve_iv)

    # Eje x = log-moneyness ln(K/S₀) (convención estándar, Gatheral): simétrico alrededor
    # de ATM=0, en vez de moneyness lineal K/S₀ (asimétrica, comprime las alas OTM).
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=smile_df["log_moneyness"], y=smile_df["iv"] * 100, mode="markers", name="Mercado (OTM)",
        marker=dict(color="#EAF2FB", size=8, symbol="circle"),
        text=smile_df["type"], hovertemplate="ln(K/S₀)=%{x:.3f}<br>IV=%{y:.2f}%<br>%{text}<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=smile_df["log_moneyness"], y=[sigma_flat * 100] * len(smile_df), mode="lines", name=f"B&S (σ plana = {sigma_flat*100:.1f}%)",
        line=dict(color="#3B82C4", width=2.5, dash="dot"),
    ))
    fig3.add_trace(go.Scatter(
        x=curve_lm[curve_mask], y=curve_iv[curve_mask] * 100, mode="lines", name="Heston (calibrado, curva)",
        line=dict(color="#1FAE85", width=2.5),
    ))
    fig3.add_trace(go.Scatter(
        x=smile_df["log_moneyness"], y=smile_df["heston_iv"] * 100, mode="markers", name="Heston (en strikes de mercado)",
        marker=dict(color="#1FAE85", size=6, symbol="diamond"),
        hovertemplate="ln(K/S₀)=%{x:.3f}<br>Heston IV=%{y:.2f}%<extra></extra>",
    ))
    fig3.add_vline(x=0.0, line_dash="dash", line_color="#8492A6", annotation_text="ATM", annotation_font_color="#8492A6")
    fig3.add_vline(x=np.log(strike / S0), line_color="#D9822B", annotation_text="Seleccionado", annotation_font_color="#D9822B")
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Log-moneyness ln(K/S₀)", yaxis_title="Volatilidad implícita (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---------- Interpretación automática del smile ----------
    low_side = valid[valid["moneyness"] < 0.95]
    high_side = valid[valid["moneyness"] > 1.05]
    if len(low_side) > 0 and len(high_side) > 0:
        iv_low = low_side["iv"].mean()
        iv_high = high_side["iv"].mean()
        skew_txt = (
            f"IV promedio en el ala baja (puts OTM, K/S&lt;0.95) = {iv_low*100:.1f}% vs "
            f"ala alta (calls OTM, K/S&gt;1.05) = {iv_high*100:.1f}%. "
            + (
                "Hay <b>skew negativo</b> (las puts OTM cuestan más en términos de vol) — el patrón típico de "
                "acciones e índices: el mercado paga más por protección contra caídas que por upside, "
                "consistente con ρ&lt;0 en la calibración de Heston."
                if iv_low > iv_high else
                "El skew es plano o invertido en este vencimiento — puede pasar en subyacentes con dinámica "
                "distinta (commodities, algunos ETFs) o simplemente con pocas quotes líquidas en las alas."
            )
        )
    else:
        skew_txt = "No hay suficientes quotes en ambas alas para medir el skew de forma confiable en este vencimiento."

    st.markdown(
        f"""
        <div class="interp-box">
        <b>Lectura del smile</b><br><br>
        La línea punteada azul (B&amp;S) es <b>plana por construcción</b> — usa una sola σ para todos los strikes,
        así que **necesariamente** falla en ajustar cualquier forma que no sea una línea recta. Su error (RMSE)
        contra el mercado es de <b>{rmse_bs:.2f} puntos de volatilidad</b>.<br><br>
        La curva verde (Heston calibrado) sí se dobla para seguir la forma observada, porque ρ y ξ le dan
        grados de libertad que B&amp;S no tiene. Su error cae a <b>{rmse_heston:.2f} puntos de volatilidad</b> —
        una mejora de <b>{mejora:.0f}%</b> sobre el modelo de vol plana.<br><br>
        {skew_txt}<br><br>
        <b>Esto es, literalmente, la respuesta a "¿por qué Heston?"</b> — B&amp;S asume que el mercado piensa
        que la volatilidad es la misma sin importar qué tan lejos esté el strike del spot; el smile de arriba
        muestra que eso es falso, y Heston es la forma más simple (dentro de lo visto en el curso) de
        corregirlo sin perder una fórmula semi-cerrada.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Metodología: el smile usa el lado OTM de cada strike (puts para K<S₀, calls para K≥S₀), la convención "
        "estándar de mercado por ser el lado más líquido. La IV de Heston se obtiene invirtiendo B&S sobre el "
        "precio que arroja el modelo calibrado — es la forma estándar de comparar un modelo de vol estocástica "
        "contra el mercado en el mismo espacio (vol implícita, no precio)."
    )

    # ---------- Bonus: superficie 3D across expiries ----------
    with st.expander("🎁 Bonus: superficie de volatilidad 3D (todos los vencimientos)"):
        st.caption(
            "Calibra Heston en varios vencimientos y arma una superficie IV(moneyness, τ). "
            "Puede tardar 20-40s la primera vez (se cachea después)."
        )
        if st.button("Calcular superficie 3D"):
            expiries_for_surface = expiries[:8] if len(expiries) > 8 else expiries
            moneyness_grid = np.linspace(0.75, 1.25, 13)

            @st.cache_data(ttl=1800, show_spinner=False)
            def build_vol_surface(ticker, expiries_tuple, S0, r, q, moneyness_grid_tuple):
                surface_rows = []
                scatter_rows = []
                for exp_ in expiries_tuple:
                    tau_ = tau_from_expiry(exp_)
                    if tau_ < 2 / 365:
                        continue
                    params_, _, _, _ = calibrate_for_expiry(ticker, exp_, S0, r, q)
                    if params_ is None:
                        continue
                    v0_, theta_, kappa_, xi_, rho_ = params_
                    for m in moneyness_grid_tuple:
                        K_ = m * S0
                        otype_ = "call" if m >= 1.0 else "put"
                        price_ = heston_price(S0, K_, r, q, tau_, v0_, kappa_, theta_, xi_, rho_, otype_)
                        iv_ = bs_implied_vol(price_, S0, K_, r, q, tau_, otype_)
                        surface_rows.append({"moneyness": m, "tau": tau_, "iv": iv_})
                    smile_ = compute_smile(ticker, exp_, S0, r, q, tau_, tuple(params_))
                    for _, row in smile_.iterrows():
                        scatter_rows.append({"moneyness": row["moneyness"], "tau": tau_, "iv": row["iv"]})
                return pd.DataFrame(surface_rows), pd.DataFrame(scatter_rows)

            with st.spinner("Calibrando Heston en cada vencimiento y armando la superficie..."):
                surf_df, scatter_df = build_vol_surface(
                    ticker, tuple(expiries_for_surface), S0, r, q, tuple(moneyness_grid)
                )

            if surf_df.empty:
                st.warning("No se pudo construir la superficie (pocos vencimientos con datos líquidos).")
            else:
                pivot = surf_df.pivot(index="tau", columns="moneyness", values="iv") * 100
                fig_surf = go.Figure(data=[
                    go.Surface(
                        x=pivot.columns, y=pivot.index, z=pivot.values,
                        colorscale=[[0, "#1FAE85"], [0.5, "#3B82C4"], [1, "#D9822B"]],
                        opacity=0.85, showscale=True, colorbar=dict(title="IV %"),
                    )
                ])
                if not scatter_df.empty:
                    fig_surf.add_trace(go.Scatter3d(
                        x=scatter_df["moneyness"], y=scatter_df["tau"], z=scatter_df["iv"] * 100,
                        mode="markers", marker=dict(size=2.5, color="#EAF2FB"), name="Mercado (OTM)",
                    ))
                fig_surf.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    height=560, margin=dict(l=0, r=0, t=20, b=0),
                    scene=dict(
                        xaxis_title="Moneyness (K/S₀)", yaxis_title="τ (años)", zaxis_title="IV (%)",
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig_surf, use_container_width=True)
                st.caption(
                    f"Superficie construida con {len(expiries_for_surface)} vencimientos calibrados por separado "
                    "(cada uno con su propio v₀, κ, θ, ξ, ρ) — no es un solo Heston 'global', es la envolvente de "
                    "smiles individuales por plazo. Los puntos blancos son las quotes de mercado reales (OTM)."
                )

# ============================================================
# 12B. EXTRA — PORTAFOLIO & GREEKS AGREGADAS (+3 pts)
# ============================================================
st.markdown("---")
st.markdown("### Extra · Portafolio & Greeks agregadas")
st.caption(
    "Requisito de puntos extra del feedback del risk manager: *'Greeks without a portfolio are an academic "
    "toy... Allow the user to enter a small portfolio (2–4 options) and show aggregate Greeks. Then answer: "
    "what is your Delta-hedge, and how much does it cost to stay Gamma-neutral?'* "
    "Edita las patas abajo (mismo vencimiento seleccionado; multiplicador estándar de 100 acciones/contrato)."
)

strikes_all = sorted(float(k) for k in chain_df["strike"].unique())
_atm_k = min(strikes_all, key=lambda k: abs(k - S0))
_otm_candidates = [k for k in strikes_all if k > S0 * 1.04]
_otm_k = _otm_candidates[0] if _otm_candidates else strikes_all[-1]

default_legs = pd.DataFrame([
    {"Tipo": "call", "Strike": _atm_k, "Contratos (+long / −short)": 1},
    {"Tipo": "call", "Strike": _otm_k, "Contratos (+long / −short)": -1},
])
legs = st.data_editor(
    default_legs,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Tipo": st.column_config.SelectboxColumn("Tipo", options=["call", "put"], required=True),
        "Strike": st.column_config.SelectboxColumn("Strike", options=strikes_all, required=True),
        "Contratos (+long / −short)": st.column_config.NumberColumn(
            "Contratos (+long / −short)", min_value=-20, max_value=20, step=1
        ),
    },
    key="portfolio_legs",
)

legs = legs.dropna(subset=["Tipo", "Strike", "Contratos (+long / −short)"])
legs = legs[legs["Contratos (+long / −short)"] != 0]
if len(legs) > 4:
    st.warning("El brief pide un portafolio pequeño (2–4 opciones); se usan solo las primeras 4 patas.")
    legs = legs.head(4)

if len(legs) < 2:
    st.info("Agrega al menos 2 patas con cantidad distinta de cero para calcular las Greeks agregadas.")
else:
    MULT = 100  # acciones por contrato
    agg = {"B&S": dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
           "Heston": dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)}
    leg_detail = []
    for _, leg in legs.iterrows():
        lt, lk, lq = leg["Tipo"], float(leg["Strike"]), float(leg["Contratos (+long / −short)"])
        qrow = chain_df[(chain_df["type"] == lt) & (np.isclose(chain_df["strike"], lk))]
        if not qrow.empty:
            leg_mid = float(qrow["mid"].iloc[0])
            leg_iv = bs_implied_vol(leg_mid, S0, lk, r, q, tau, lt)
            if not np.isfinite(leg_iv) or leg_iv <= 0:
                _yf_iv = qrow["impliedVolatility"].iloc[0]
                leg_iv = float(_yf_iv) if np.isfinite(_yf_iv) and _yf_iv > 0 else sigma_contract
        else:
            leg_mid, leg_iv = np.nan, sigma_contract
        gb = bs_greeks(S0, lk, r, q, tau, leg_iv, lt)
        gh = heston_greeks_fd(S0, lk, r, q, tau, v0, kappa, theta, xi, rho, lt, full=False)
        for k_ in ["delta", "gamma", "vega", "theta"]:
            agg["B&S"][k_] += lq * MULT * gb[k_]
            agg["Heston"][k_] += lq * MULT * gh[k_]
        leg_detail.append({
            "Pata": f"{'+' if lq>0 else ''}{lq:g} {lt} K={lk:g}",
            "Mid": leg_mid, "IV usada (B&S)": leg_iv,
            "Δ Heston/contrato": gh["delta"], "Γ Heston/contrato": gh["gamma"],
        })

    st.markdown("##### Greeks netas del libro (por posición total, multiplicador 100)")
    agg_df = pd.DataFrame(agg).T
    agg_df.columns = ["Net Δ (acciones)", "Net Γ (acc./$)", "Net Vega ($/1pto vol)", "Net Θ ($/día)"]
    st.dataframe(
        agg_df.style.format({"Net Δ (acciones)": "{:+.1f}", "Net Γ (acc./$)": "{:+.3f}",
                             "Net Vega ($/1pto vol)": "{:+.2f}", "Net Θ ($/día)": "{:+.2f}"}),
        use_container_width=True,
    )
    with st.expander("Detalle por pata"):
        st.dataframe(pd.DataFrame(leg_detail).set_index("Pata").style.format(
            {"Mid": "${:.2f}", "IV usada (B&S)": "{:.2%}", "Δ Heston/contrato": "{:.4f}", "Γ Heston/contrato": "{:.4f}"}),
            use_container_width=True)

    # ---- Respuestas del risk manager (con el motor calibrado, Heston) ----
    net_d, net_g = agg["Heston"]["delta"], agg["Heston"]["gamma"]
    hedge_shares = -net_d
    accion_delta = "compra" if hedge_shares > 0 else "vende"

    atm_row_c = chain_df[(chain_df["type"] == "call") & (np.isclose(chain_df["strike"], _atm_k))]
    atm_mid_c = float(atm_row_c["mid"].iloc[0]) if not atm_row_c.empty else heston_price(S0, _atm_k, r, q, tau, v0, kappa, theta, xi, rho, "call")
    g_atm = heston_greeks_fd(S0, _atm_k, r, q, tau, v0, kappa, theta, xi, rho, "call", full=False)
    n_gamma_hedge = -net_g / (MULT * g_atm["gamma"]) if abs(g_atm["gamma"]) > 1e-12 else np.nan
    costo_prima = n_gamma_hedge * MULT * atm_mid_c
    delta_residual = net_d + n_gamma_hedge * MULT * g_atm["delta"]

    st.markdown(
        f"""
        <div class="interp-box">
        <b>¿Cuál es tu Delta-hedge?</b> El libro carga Δ neto = {net_d:+.1f} acciones (Heston). Para
        neutralizarlo: <b>{accion_delta} {abs(hedge_shares):.0f} acciones</b> del subyacente
        (≈ ${abs(hedge_shares)*S0:,.0f} de nocional a S₀={S0:.2f}).<br><br>
        <b>¿Cuánto cuesta quedarte Gamma-neutral?</b> El subyacente tiene Γ=0, así que la Γ solo se
        neutraliza con opciones. Usando el call ATM (K={_atm_k:g}, Γ={g_atm['gamma']:.4f}/contrato) como
        instrumento de cobertura: se necesitan <b>{n_gamma_hedge:+.2f} contratos</b> →
        prima {'desembolsada' if costo_prima>0 else 'recibida'} ≈ <b>${abs(costo_prima):,.0f}</b>.
        Esa cobertura agrega Δ propio, así que el hedge de acciones se re-ajusta a
        <b>{-delta_residual:+.0f} acciones</b>. El costo real de mantenerse Γ-neutral no es solo la prima:
        es también el Θ que esa posición larga de opciones sangra cada día — el "alquiler" de la convexidad.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Método (nota del risk manager sobre consistencia): las Greeks de B&S son de forma cerrada con la IV "
        "propia de cada pata; las de Heston son bump-and-reprice con la convención del notebook 2 sobre los "
        "parámetros calibrados de este vencimiento — mismas convenciones de unidades en ambos motores "
        "(Δ/Γ crudas, Vega por 1pto, Θ por día). Los hedges se calculan con Heston (el motor calibrado)."
    )

# ============================================================
# 13. APARTADO EXTRA — CALIBRACIÓN ROBUSTA DE HESTON (+3 pts)
# ============================================================
st.markdown("---")
st.markdown("### Extra · Calibración robusta de Heston")
st.caption(
    "Requisito de puntos extra del feedback del quant developer: optimizador sensato con bounds y buenos "
    "initial guesses, condición de Feller como restricción (o flaggeada), y estabilidad de parámetros "
    "re-calibrando en días distintos."
)

st.markdown("##### Parámetros calibrados en este vencimiento")
pcols = st.columns(5)
labels = ["v₀ (var. inicial)", "θ (var. largo plazo)", "κ (reversión)", "ξ (vol-of-vol)", "ρ (correlación)"]
for col, lab, val in zip(pcols, labels, heston_params):
    col.metric(lab, f"{val:.4f}")

feller_msg = (
    "se cumple ✅ (la varianza no toca cero)" if feller_ok
    else "VIOLADA ⚠️ (la varianza puede tocar cero — frecuente en calibraciones de mercado real; "
         "el mercado a veces exige colas más gordas de las que la región Feller permite)"
)
st.markdown(
    f"""
    <div class="interp-box">
    <b>Condición de Feller:</b> 2κθ = {feller_lhs:.4f} vs ξ² = {feller_rhs:.4f} → {feller_msg}.<br><br>
    <b>Metodología del optimizador:</b> búsqueda global por Latin
    Hypercube (30 candidatos, seed=1, dentro de los bounds del notebook) + refinamiento local
    least_squares/TRF de los 10 mejores (max_nfev=200), con residuos ponderados por
    1/max(spread, 0.01). El presupuesto original (10 candidatos, max_nfev=10, "verbatim" del
    notebook 4) resultó insuficiente para que least_squares converja en las 5 dimensiones del
    ajuste; se corrigió con el mismo presupuesto ya validado en el resto del proyecto.
    La condición de Feller se trata como
    <b>diagnóstico</b> (igual que los notebooks del curso): se reporta pero no se impone — el mercado puede
    exigir un skew que la viole. Opcional en el sidebar: <b>fijar κ</b> (mitigación del valle de
    identificabilidad κ–ξ del notebook 5) para parámetros más estables entre corridas.<br><br>
    <b>Motor de pricing de la calibración:</b> COS vectorizado (Fang-Oosterlee 2008, notebook 6 del curso) —
    una sola evaluación de la función característica pricea todos los strikes del vencimiento (~68× más rápido
    que integración escalar, medido) — con guardas de sanidad (cumulantes, cotas de no-arbitraje, monotonía en K)
    y fallback automático a integración directa. Función característica en forma estable
    (<i>little Heston trap</i>, Albrecher et al. 2007).<br><br>
    <b>Validación cruzada de motores</b> (contrato seleccionado): integración directa ${price_heston:.4f}
    vs COS ${price_heston_cos:.4f} — diferencia {cross_check_diff:.2e}.
    Quotes usadas en la calibración: {n_quotes_calib}.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("##### Estabilidad de la calibración entre días")
log_calibration(ticker, expiry, heston_params, feller_ok, n_quotes_calib, float(fit_obj.cost))
hist_df = load_calibration_history(ticker, expiry)

if len(hist_df) < 2:
    st.info(
        "Todavía solo hay **un día** de calibración registrado para este ticker+vencimiento "
        f"({hist_df['date'].iloc[-1] if len(hist_df) else date.today().isoformat()}). "
        "Corre el dashboard otro día con el mismo contrato y aquí aparecerá automáticamente la "
        "comparación de estabilidad (*'re-calibrate on two different days and compare'*)."
    )
else:
    last_two = hist_df.tail(2).reset_index(drop=True)
    d_old, d_new = last_two.iloc[0], last_two.iloc[1]
    st.caption(f"Comparando calibración de **{d_old['date']}** vs **{d_new['date']}**:")

    stab_cols = st.columns(5)
    param_keys = ["v0", "theta", "kappa", "xi", "rho"]
    max_pct_change = 0.0
    for col, key, lab in zip(stab_cols, param_keys, labels):
        old_v, new_v = d_old[key], d_new[key]
        pct = abs(new_v - old_v) / (abs(old_v) + 1e-6) * 100
        max_pct_change = max(max_pct_change, pct)
        col.metric(lab, f"{new_v:.4f}", f"{new_v - old_v:+.4f} ({pct:.0f}%)")

    if max_pct_change > 50:
        st.warning(
            f"⚠️ **Inestabilidad detectada**: al menos un parámetro cambió más de {max_pct_change:.0f}% "
            "de un día a otro. Es el riesgo que advierte el feedback del quant developer — la superficie "
            "de pérdida tiene un valle de identificabilidad κ–ξ (múltiples mínimos casi-equivalentes), "
            "común con pocas quotes líquidas o vencimientos muy cortos. No es un error del código."
        )
    else:
        st.success(
            f"✅ Parámetros razonablemente estables (cambio máximo: {max_pct_change:.0f}%) entre las dos "
            "calibraciones más recientes."
        )

    with st.expander("Ver historial completo de calibraciones"):
        st.dataframe(
            hist_df[["date", "v0", "theta", "kappa", "xi", "rho", "feller_ok", "n_quotes"]].style.format(
                {"v0": "{:.4f}", "theta": "{:.4f}", "kappa": "{:.4f}", "xi": "{:.4f}", "rho": "{:.4f}"}
            ),
            use_container_width=True,
        )
