"""
Heston + Black-Scholes engines - Options Analytics Dashboard
Consolida y corrige lo revisado en los notebooks de clase:
  - heston_cf_j: forma estable "Little Trap" (Albrecher et al. 2007), con q.
  - cos_price_vector: metodo COS (Fang-Oosterlee 2008), VECTORIZADO sobre
    un array de strikes de la misma madurez -- una sola pasada de numpy en
    vez de una integral adaptativa por strike. Este es el fix de eficiencia
    para que la calibracion (cientos de evaluaciones) no tarde minutos.
  - heston_call_direct: integracion de Fourier adaptativa (mas lenta), se
    conserva solo como cross-check de validacion, no para uso en produccion.
  - bs_price / bs_greeks: Black-Scholes cerrado, con dividend yield q.
  - heston_greeks_fd: diferencias finitas centradas, motor generico
    (bug de Rho financiera del notebook anterior corregido aqui).
  - calibrate_heston: dos etapas (Latin Hypercube + least_squares), usando
    el pricer vectorizado. Bug de NB4 (theta/kappa invertidos en loss())
    corregido: aqui loss/residuals llaman con el mismo orden posicional
    que la firma real de las funciones de pricing.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.stats import qmc


# ===========================================================================
# 1. Black-Scholes (con dividend yield q)
# ===========================================================================
def bs_price(S, K, r, q, tau, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    call = S*np.exp(-q*tau)*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return call, d1, d2


def bs_implied_vol(price, S, K, r, q, tau, tol=1e-7):
    intrinsic = max(S*np.exp(-q*tau) - K*np.exp(-r*tau), 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        diff = bs_price(S, K, r, q, tau, mid)[0] - price
        if abs(diff) < tol:
            return mid
        lo, hi = (mid, hi) if diff < 0 else (lo, mid)
    return mid


def bs_greeks(S, K, r, q, tau, sigma):
    """Griegas analiticas de B&S (call). Todas de forma cerrada."""
    price, d1, d2 = bs_price(S, K, r, q, tau, sigma)
    nd1 = norm.pdf(d1)
    disc_q = np.exp(-q*tau)
    disc_r = np.exp(-r*tau)

    delta = disc_q * norm.cdf(d1)
    gamma = disc_q * nd1 / (S*sigma*np.sqrt(tau))
    vega = S*disc_q*nd1*np.sqrt(tau)                      # por 1.0 de sigma (no /100)
    theta = (-S*disc_q*nd1*sigma/(2*np.sqrt(tau))
             - r*K*disc_r*norm.cdf(d2)
             + q*S*disc_q*norm.cdf(d1))                     # por anio (no /365)
    rho = K*tau*disc_r*norm.cdf(d2)                          # por 1.0 de r
    vanna = -disc_q*nd1*d2/sigma                             # dDelta/dVol = dVega/dS
    volga = vega*d1*d2/sigma                                 # dVega/dVol

    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho, "vanna": vanna, "volga": volga}


# ===========================================================================
# 2. Funcion caracteristica de Heston (Little Trap), con q
# ===========================================================================
def heston_cf_j(u, j, x, v, tau, r, q, kappa, theta, xi, rho):
    i = 1j
    uj = 0.5 if j == 1 else -0.5
    bj = kappa - rho*xi if j == 1 else kappa
    dj = np.sqrt((rho*xi*i*u - bj)**2 + xi**2*(u**2 - 2*uj*i*u))
    g2j = (bj - rho*xi*i*u + dj) / (bj - rho*xi*i*u - dj)
    Dj = ((bj - rho*xi*i*u + dj)/xi**2) * ((1-np.exp(dj*tau))/(1-g2j*np.exp(dj*tau)))
    drift = (r - q)          # dividend yield resta del drift risk-neutral
    Cj = drift*i*u*tau + (kappa*theta/xi**2)*(
        (bj-rho*xi*i*u+dj)*tau - 2*np.log((1-g2j*np.exp(dj*tau))/(1-g2j)))
    return np.exp(Cj + Dj*v + i*u*x)


def heston_call_direct(S, K, r, q, tau, v0, kappa, theta, xi, rho):
    """Integracion de Fourier adaptativa (quad). Lenta -- solo para cross-check."""
    x = np.log(S)

    def prob_j(j):
        lnK = np.log(K)
        integrand = lambda u: np.real(
            np.exp(-1j*u*lnK) * heston_cf_j(u, j, x, v0, tau, r, q, kappa, theta, xi, rho) / (1j*u))
        val, _ = quad(integrand, 1e-8, 200, limit=200)
        return 0.5 + val/np.pi

    P1, P2 = prob_j(1), prob_j(2)
    price = S*np.exp(-q*tau)*P1 - K*np.exp(-r*tau)*P2
    intrinsic = max(S*np.exp(-q*tau) - K*np.exp(-r*tau), 0.0)
    return max(price, intrinsic)


# ===========================================================================
# 3. COS vectorizado: precio de un ARRAY de strikes de la misma madurez
#    en una sola operacion matricial (Fang & Oosterlee, 2008)
# ===========================================================================
def cos_price_vector(S0, Ks, r, q, tau, v0, kappa, theta, xi, rho, N=256, L=12):
    """
    Precios de call europeo para un array de strikes `Ks` (misma tau),
    vectorizado: los nodos u_k y coeficientes U_k se calculan UNA vez,
    y phi(u,0) tambien se evalua UNA vez (no depende del strike, solo
    x0=ln(S0/K) lo hace, y eso entra como una fase exp(i*u*x0) barata).
    """
    Ks = np.atleast_1d(np.asarray(Ks, dtype=float))
    drift = r - q

    c1 = drift*tau + (1-np.exp(-kappa*tau))*(theta-v0)/(2*kappa) - 0.5*theta*tau
    c2 = (1.0/(8*kappa**3))*(
        xi*tau*kappa*np.exp(-kappa*tau)*(v0-theta)*(8*kappa*rho-4*xi)
        + kappa*rho*xi*(1-np.exp(-kappa*tau))*(16*theta-8*v0)
        + 2*theta*kappa*tau*(-4*kappa*rho*xi+xi**2+4*kappa**2)
        + xi**2*((theta-2*v0)*np.exp(-2*kappa*tau)+theta*(6*np.exp(-kappa*tau)-7)+2*v0)
        + 8*kappa**2*(v0-theta)*(1-np.exp(-kappa*tau)))
    a = c1 - L*np.sqrt(abs(c2))
    b = c1 + L*np.sqrt(abs(c2))

    k = np.arange(N)
    u = k*np.pi/(b-a)
    u_cf = np.where(u == 0, 1e-8, u)

    # Coeficientes del payoff de call, U_k (chi/psi sobre el dominio [0,b])
    kp = k*np.pi/(b-a)
    d, c = b, 0.0
    chi = (1/(1+kp**2)) * (np.cos(kp*(d-a))*np.exp(d) - np.cos(kp*(c-a))*np.exp(c)
                           + kp*np.sin(kp*(d-a))*np.exp(d) - kp*np.sin(kp*(c-a))*np.exp(c))
    psi = np.where(k == 0, d-c, (np.sin(kp*(d-a))-np.sin(kp*(c-a)))/np.where(kp == 0, 1, kp))
    Uk = 2/(b-a)*(chi-psi)

    # phi(u,0): NO depende del strike -- se evalua una sola vez por vencimiento
    phi0 = heston_cf_j(u_cf, 2, 0.0, v0, tau, r, q, kappa, theta, xi, rho)
    phase_a = np.exp(-1j*u*a)
    base_terms = phi0 * phase_a * Uk          # shape (N,)
    base_terms[0] *= 0.5

    # Fase especifica de cada strike: exp(i*u*x0_k), x0_k = ln(S0/K_k)
    x0 = np.log(S0/Ks)                        # shape (n_strikes,)
    phase_k = np.exp(1j*np.outer(x0, u))      # shape (n_strikes, N)

    terms = np.real(phase_k * base_terms[None, :])   # (n_strikes, N)
    precios = Ks*np.exp(-r*tau) * terms.sum(axis=1)

    # Piso de no-arbitraje: el precio de un call nunca puede ser menor que su
    # intrinseco descontado. La truncacion numerica del COS puede, en strikes
    # extremos o parametros de calibracion fuera de rango normal, dar un
    # precio ligeramente por debajo de eso (o incluso negativo) -- se acota.
    intrinsic = np.maximum(S0*np.exp(-q*tau) - Ks*np.exp(-r*tau), 0.0)
    precios = np.maximum(precios, intrinsic)
    return precios


def cos_price_single(S0, K, r, q, tau, v0, kappa, theta, xi, rho, N=256, L=12):
    return float(cos_price_vector(S0, [K], r, q, tau, v0, kappa, theta, xi, rho, N, L)[0])


# ===========================================================================
# 4. Feller
# ===========================================================================
def feller_condition(kappa, theta, xi):
    lhs = 2*kappa*theta
    rhs = xi**2
    return {"cumple": lhs > rhs, "lhs": lhs, "rhs": rhs}


# ===========================================================================
# 5. Greeks de Heston por diferencias finitas (motor generico)
#    Corrige el bug de la vez pasada: Rho financiera = d/dr, no d/drho.
# ===========================================================================
def heston_greeks_fd(S0, K, r, q, tau, v0, kappa, theta, xi, rho, h_rel=1e-4):
    def precio(S=S0, K_=K, r_=r, q_=q, tau_=tau, v0_=v0, kappa_=kappa,
               theta_=theta, xi_=xi, rho_=rho):
        return cos_price_single(S, K_, r_, q_, tau_, v0_, kappa_, theta_, xi_, rho_)

    def central(bump_name, x0, h):
        kwargs_up = {bump_name: x0+h}
        kwargs_dn = {bump_name: x0-h}
        return (precio(**kwargs_up) - precio(**kwargs_dn)) / (2*h)

    base = precio()
    hS = S0*h_rel
    delta = (precio(S=S0+hS) - precio(S=S0-hS)) / (2*hS)
    gamma = (precio(S=S0+hS) - 2*base + precio(S=S0-hS)) / (hS**2)

    hv0 = max(v0*h_rel, 1e-6)
    vega_v0 = central("v0_", v0, hv0)           # proxy de Vega: sensibilidad a v0

    # Vanna = d(Delta)/d(v0)  (mixed partial, sensibilidad cruzada spot-vol)
    hS2, hv02 = S0*h_rel, max(v0*h_rel, 1e-6)
    delta_up = (precio(S=S0+hS2, v0_=v0+hv02) - precio(S=S0-hS2, v0_=v0+hv02)) / (2*hS2)
    delta_dn = (precio(S=S0+hS2, v0_=v0-hv02) - precio(S=S0-hS2, v0_=v0-hv02)) / (2*hS2)
    vanna = (delta_up - delta_dn) / (2*hv02)

    # Volga = d(Vega)/d(v0)  (segunda derivada respecto a v0)
    volga = (precio(v0_=v0+hv02) - 2*base + precio(v0_=v0-hv02)) / (hv02**2)

    htau = max(tau*h_rel, 1e-6)
    theta_g = -(precio(tau_=tau+htau) - precio(tau_=tau-htau)) / (2*htau)  # theta = -dC/dtau

    hr = max(abs(r)*h_rel, 1e-5)                # bump absoluto: r puede ser ~0
    rho_g = (precio(r_=r+hr) - precio(r_=r-hr)) / (2*hr)   # Rho FINANCIERA: d/dr

    hxi = xi*h_rel
    sens_xi = central("xi_", xi, hxi)
    htheta = theta*h_rel
    sens_theta = central("theta_", theta, htheta)
    hkappa = kappa*h_rel
    sens_kappa = central("kappa_", kappa, hkappa)
    hrho_corr = max(abs(rho)*h_rel, 1e-5)
    sens_rho_corr = central("rho_", rho, hrho_corr)

    return {"price": base, "delta": delta, "gamma": gamma, "vega_v0": vega_v0,
            "vanna": vanna, "volga": volga, "theta": theta_g, "rho": rho_g,
            "sens_kappa": sens_kappa, "sens_theta": sens_theta,
            "sens_xi": sens_xi, "sens_rho": sens_rho_corr}


# ===========================================================================
# 6. Calibracion: dos etapas (Latin Hypercube global + least_squares local)
#    Bug de NB4 corregido: aqui loss()/residuals() llaman a
#    cos_price_vector/heston_call_direct con el MISMO orden posicional que
#    su firma real (v0, kappa, theta, xi, rho) -- nunca invertidos.
# ===========================================================================
NOMBRES_HESTON = ["v0", "theta", "kappa", "xi", "rho"]
BOUNDS_HESTON = [
    (0.005, 0.20),   # v0
    (0.005, 0.20),   # theta
    (0.10,  6.00),   # kappa
    (0.05,  1.50),   # xi
    (-0.95, 0.50),   # rho  (asimetrico: equities tienen skew negativo)
]


def _pack(params):
    v0, theta, kappa, xi, rho = params
    return v0, theta, kappa, xi, rho


def loss_vectorized(params, S0, r, q, quotes_by_tau):
    """quotes_by_tau: dict {tau: (strikes_array, prices_mkt_array, spreads_array)}"""
    v0, theta, kappa, xi, rho = _pack(params)
    total = 0.0
    for tau, (Ks, mkt, spreads) in quotes_by_tau.items():
        try:
            with np.errstate(all="ignore"):
                mdl = cos_price_vector(S0, Ks, r, q, tau, v0, kappa, theta, xi, rho)
        except Exception:
            return 1e10
        if not np.all(np.isfinite(mdl)):
            return 1e10
        w = 1.0 / np.maximum(spreads, 0.01)**2
        total += np.sum(w*(mdl-mkt)**2)
    return total


def residuals_vectorized(params, S0, r, q, quotes_by_tau):
    v0, theta, kappa, xi, rho = _pack(params)
    res = []
    for tau, (Ks, mkt, spreads) in quotes_by_tau.items():
        try:
            with np.errstate(all="ignore"):
                mdl = cos_price_vector(S0, Ks, r, q, tau, v0, kappa, theta, xi, rho)
        except Exception:
            res.extend([1e3]*len(Ks))
            continue
        if not np.all(np.isfinite(mdl)):
            res.extend([1e3]*len(Ks))
            continue
        w = 1.0 / np.maximum(spreads, 0.01)
        res.extend((w*(mdl-mkt)).tolist())
    return res


def calibrate_heston(S0, r, q, quotes_by_tau, bounds=None, n_candidatos=30,
                      n_refinar=10, seed=1, fix_kappa=None):
    """
    Dos etapas: barrido Latin-Hypercube (global, barato) + least_squares
    (local, TRF) sobre los mejores candidatos.
    fix_kappa: si se da un valor, se fija kappa (bounds casi degenerados)
    para mitigar el valle de identificabilidad kappa-xi (ver notebook 5).
    """
    bounds = bounds or BOUNDS_HESTON
    if fix_kappa is not None:
        bounds = list(bounds)
        bounds[2] = (fix_kappa - 1e-4, fix_kappa + 1e-4)

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    muestra = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n_candidatos)
    candidatos = lb + muestra*(ub-lb)
    perdidas = [loss_vectorized(c, S0, r, q, quotes_by_tau) for c in candidatos]
    mejores_idx = np.argsort(perdidas)[:n_refinar]

    mejor_fit = None
    for idx in mejores_idx:
        fit = least_squares(residuals_vectorized, candidatos[idx], bounds=(lb, ub),
                             args=(S0, r, q, quotes_by_tau), max_nfev=200)
        if mejor_fit is None or fit.cost < mejor_fit.cost:
            mejor_fit = fit

    params = dict(zip(NOMBRES_HESTON, mejor_fit.x))
    feller = feller_condition(params["kappa"], params["theta"], params["xi"])
    return {"params": params, "fit": mejor_fit, "feller": feller,
            "loss_final": 2*mejor_fit.cost}
