"""
Data pipeline - Options Analytics Dashboard
Ticker dinamico (ingresado por el usuario en Streamlit) -> snapshot limpio y
estandarizado listo para alimentar B&S y Heston.

Requiere: yfinance, pandas-datareader, pandas, numpy
(los imports de red son lazy: este modulo se puede importar y probar
 sin tener yfinance/pandas-datareader instalados)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone


def get_underlying_data(ticker: str) -> dict:
    """Valida el ticker y regresa spot, dividend yield y expiries disponibles."""
    import yfinance as yf
    tk = yf.Ticker(ticker)

    expiries = tk.options
    if not expiries:
        raise ValueError(
            f"'{ticker}' no tiene opciones listadas en yfinance. "
            "Verifica el simbolo o prueba con otro ticker."
        )

    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"'{ticker}' no devolvio precio spot (posible ticker invalido).")
    spot = float(hist["Close"].iloc[-1])

    info = tk.info or {}
    q_raw = info.get("dividendYield", 0.0) or 0.0
    q = q_raw / 100.0 if q_raw > 1.0 else q_raw

    # Tope de cordura: ningun equity paga >25% de dividend yield. El campo
    # dividendYield de yfinance ha regresado valores ambiguos entre versiones
    # (a veces %, a veces decimal) -- si algo se malinterpreta, este tope evita
    # que un solo input corrupto envenene el forward, la IV y la calibracion
    # completa de Heston (visto en un caso real: 0.47 leido como q=47%).
    if q > 0.25:
        print(f"[warn] dividendYield sospechoso para {ticker}: {q_raw} -> q={q:.2%}. "
              f"Se capa a 25%. Verifica manualmente contra la fuente real de dividendos.")
        q = 0.25

    return {
        "ticker": ticker,
        "spot": spot,
        "q": q,
        "expiries": list(expiries),
        "asof": datetime.now(timezone.utc),
    }


def get_risk_free_rate(fallback: float = 0.05) -> float:
    """Ultimo dato de DTB3 (T-Bill 13 semanas) de FRED, en decimal (no %)."""
    try:
        import pandas_datareader.data as web
        data = web.DataReader("DTB3", "fred")
        r = float(data.dropna().iloc[-1, 0]) / 100.0
        return r
    except Exception as e:
        print(f"[warn] No se pudo obtener DTB3 de FRED ({e}); usando fallback r={fallback:.2%}")
        return fallback


def get_option_chain_raw(ticker: str, expiry: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


REQUIRED_COLS = ["contractSymbol", "strike", "bid", "ask", "lastPrice",
                  "volume", "openInterest", "impliedVolatility"]


def clean_option_chain(df: pd.DataFrame, spot: float, side: str,
                        spread_max_rel: float = 0.20,
                        moneyness_bounds: tuple = (0.7, 1.3),
                        min_volume: int = 1) -> pd.DataFrame:
    """Reglas de limpieza del mandato: bid/ask>0, volume>=min_volume,
    spread relativo <= spread_max_rel, moneyness dentro de bounds."""
    d = df.copy()
    missing = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing:
        raise KeyError(f"Faltan columnas esperadas de yfinance: {missing}")

    n0 = len(d)
    d = d[(d["bid"] > 0) & (d["ask"] > 0)]
    d = d[d["volume"].fillna(0) >= min_volume]

    d["mid"] = (d["bid"] + d["ask"]) / 2.0
    d["spread_rel"] = (d["ask"] - d["bid"]) / d["mid"]
    d = d[d["spread_rel"] <= spread_max_rel]

    d["moneyness"] = d["strike"] / spot
    lo, hi = moneyness_bounds
    d = d[(d["moneyness"] >= lo) & (d["moneyness"] <= hi)]

    d["type"] = side
    d.attrs["n_dropped"] = n0 - len(d)
    d.attrs["n_kept"] = len(d)
    return d.reset_index(drop=True)


SCHEMA = ["type", "strike", "moneyness", "bid", "ask", "mid", "Cmkt",
          "iv_mkt", "spot", "tau", "r", "q", "spread_rel",
          "openInterest", "volume", "snapshot_ts", "contractSymbol"]


def build_snapshot(ticker: str, expiry: str, r: float = None,
                    spread_max_rel: float = 0.20,
                    moneyness_bounds: tuple = (0.7, 1.3)) -> pd.DataFrame:
    """Pipeline completo: ticker + expiry -> DataFrame limpio, un solo snapshot."""
    under = get_underlying_data(ticker)
    spot, q = under["spot"], under["q"]

    if expiry not in under["expiries"]:
        raise ValueError(
            f"Expiry '{expiry}' no esta disponible para {ticker}. "
            f"Expiries validas: {under['expiries']}"
        )

    if r is None:
        r = get_risk_free_rate()

    calls_raw, puts_raw = get_option_chain_raw(ticker, expiry)
    calls = clean_option_chain(calls_raw, spot, "call", spread_max_rel, moneyness_bounds)
    puts = clean_option_chain(puts_raw, spot, "put", spread_max_rel, moneyness_bounds)

    if calls.empty and puts.empty:
        raise ValueError(
            f"Despues de limpiar, no quedo ningun contrato liquido para "
            f"{ticker} {expiry}. Relaja spread_max_rel o moneyness_bounds."
        )

    out = pd.concat([calls, puts], ignore_index=True)

    snapshot_ts = under["asof"]
    expiry_ts = pd.Timestamp(expiry, tz="UTC")
    tau = (expiry_ts - snapshot_ts).total_seconds() / (365.0 * 24 * 3600)
    if tau <= 0:
        raise ValueError(f"Expiry '{expiry}' ya vencio (tau={tau:.4f}).")

    out["spot"] = spot
    out["q"] = q
    out["r"] = r
    out["tau"] = tau
    out["snapshot_ts"] = snapshot_ts
    out["iv_mkt"] = out["impliedVolatility"]
    out["Cmkt"] = out["mid"]

    return out[SCHEMA].sort_values(["type", "strike"]).reset_index(drop=True)
