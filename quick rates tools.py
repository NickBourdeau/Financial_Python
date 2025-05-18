# Bond & Portfolio Quick-Stats Toolkit
# =====================================
# Simplified inputs: functions take face, coupon rate, maturity, etc.,
# so you don't need to build cashflow arrays or DataFrames yourself.
# Sections:
#  A) Individual Bond Summary
#  B) Portfolio Summary
#  C) Yield Curve Summary
#  D) Other Helpers
#  E) Trader Quick Calculations
#  F) Hedging Calculators

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.optimize import newton
from scipy.stats import norm
from sklearn.decomposition import PCA

# ====================================================================
# A) Individual Bond Summary
# ====================================================================

def bond_summary(
    face: float,             # face/par value of the bond
    coupon_rate: float,      # annual coupon rate (decimal), e.g. 0.05 for 5%
    maturity: float,         # time to maturity in years, e.g. 5.0
    freq: int,               # coupon payments per year, e.g. 2 for semiannual
    price: float,            # dirty price (including accrued)
    ytm: Optional[float] = None,          # if None, will be solved
    zero_curve: Optional[pd.Series] = None, # for Z-spread/OAS
    key_tenors: List[float] = [2,5,10]    # tenors for key-rate durations
) -> pd.DataFrame:
    """
    Returns summary of bond metrics:
      - Clean price & accrued interest
      - YTM
      - Macaulay & modified duration
      - Convexity
      - DV01
      - Key-rate durations (placeholders)
      - PnL for ±1 bp (duration/convexity approx)
      - Z-spread/OAS (if zero_curve provided)
    """
    # Generate cashflows and times
    n = int(maturity * freq)
    times = np.arange(1, n+1) / freq
    coupon = coupon_rate * face / freq
    cashflows = np.full(n, coupon)
    cashflows[-1] += face  # add principal at maturity

    # Accrued interest & clean price
    fraction = (1 - (times[0] % 1))
    accrued = coupon * fraction
    clean_price = price - accrued

    # Solve YTM if not given
    if ytm is None:
        def f(y):
            return np.sum(cashflows * np.exp(-y*times)) - price
        ytm = newton(f, x0=coupon_rate)

    # Durations and convexity
    dfs = np.exp(-ytm * times)
    pv = cashflows * dfs
    macaulay = np.sum(times * pv) / np.sum(pv)
    modified = macaulay / (1 + ytm / freq)
    convexity = np.sum(pv * times**2) / np.sum(pv)
    dv01 = modified * price * 1e-4

    # Key-rate durations (stub)
    kr = {f'KR_{t}y': np.nan for t in key_tenors}

    # PnL for ±1 bp
    pnl_up = -dv01 + 0.5 * convexity * price * (1e-4)**2
    pnl_down = dv01 + 0.5 * convexity * price * (1e-4)**2

    # Z-spread placeholder
    zspread = None

    data = {
        'CleanPrice': clean_price,
        'Accrued': accrued,
        'YTM': ytm,
        'MacaulayDur': macaulay,
        'ModDur': modified,
        'Convexity': convexity,
        'DV01': dv01,
        **kr,
        'PnL_up1bp': pnl_up,
        'PnL_down1bp': pnl_down,
        'ZSpread': zspread
    }
    return pd.DataFrame([data])

# ====================================================================
# B) Portfolio Summary
# ====================================================================

def portfolio_summary(
    positions: pd.DataFrame,           # columns ['Ticker','Qty']
    specs: Dict[str, Dict],            # per-ticker bond_summary inputs
    zero_curve: Optional[pd.Series] = None,
    key_tenors: List[float] = [2,5,10],
    shock_bp: float = 1.0              # basis points
) -> pd.DataFrame:
    """
    Aggregate bond_summary across positions to compute:
      - Total DV01, portfolio duration, convexity
      - PnL for ±shock_bp
    """
    stats = []
    for _, row in positions.iterrows():
        tkr, qty = row['Ticker'], row['Qty']
        s = specs[tkr]
        ind = bond_summary(s['face'], s['coupon_rate'], s['maturity'],
                           s['freq'], s['price'], s.get('ytm'),
                           zero_curve, key_tenors)
        ind['Qty'] = qty
        stats.append(ind)
    df = pd.concat(stats, ignore_index=True)
    df['Pos_DV01'] = df['DV01'] * df['Qty']
    total_dv01 = df['Pos_DV01'].sum()
    w = df['Qty'] * df['CleanPrice']
    port_dur = np.sum(df['ModDur'] * w) / w.sum()
    port_conv = np.sum(df['Convexity'] * w) / w.sum()
    port_pnl = total_dv01 * shock_bp * 1e-4
    return pd.DataFrame([{
        'TotalDV01': total_dv01,
        'PortDur': port_dur,
        'PortConv': port_conv,
        f'PnL_{shock_bp}bp': port_pnl
    }])

# ====================================================================
# C) Yield Curve Summary
# ====================================================================

def curve_summary(
    zero_df: pd.DataFrame,             # index=Date, cols=tenors
    date: pd.Timestamp,
    shock_bp: float = 1.0,
    interp_tenors: Optional[List[float]] = None
) -> Dict:
    """
    On a given date, return:
      - Min/Max/Mean/Std of curve
      - PCA scores (first 3)
      - Forward rates between tenors
      - Shocked up/down curves
      - Optional interpolation
    """
    series = zero_df.loc[date]
    stats = pd.Series({
        'Min': series.min(),
        'Max': series.max(),
        'Mean': series.mean(),
        'Std': series.std()
    })
    pca = PCA(n_components=3).fit(zero_df.values)
    idx = zero_df.index.get_loc(date)
    pca_scores = pca.transform(zero_df.values)[idx]
    ten = zero_df.columns.values
    fwd = {
        f'F_{ten[i]}x{ten[i+1]}':
        ((1+series[ten[i+1]])**ten[i+1] / (1+series[ten[i]])**ten[i]) - 1
        for i in range(len(ten)-1)
    }
    up = series + shock_bp*1e-4
    down = series - shock_bp*1e-4
    interp = None
    if interp_tenors:
        interp = pd.Series(
            np.interp(interp_tenors, ten, series.values),
            index=interp_tenors
        )
    return {
        'stats': stats,
        'pca_scores': pca_scores,
        'forwards': pd.Series(fwd),
        'base': series,
        'up': up,
        'down': down,
        'interp': interp
    }

# ====================================================================
# D) Other Helpers
# ====================================================================

def forward_rate(z_t: float, z_u: float, t: float, u: float) -> float:
    return ((1+z_u)**u / (1+z_t)**t) - 1

def zero_interpolate(zero_df: pd.DataFrame, date: pd.Timestamp, tenor: float) -> float:
    series = zero_df.loc[date]
    return float(np.interp(tenor, series.index.values, series.values))

def swap_rate(zero_curve: pd.Series) -> pd.Series:
    dfs = np.exp(-zero_curve * zero_curve.index)
    return pd.Series({
        T: (1 - dfs[T]) / (dfs.loc[:T].sum() * (zero_curve.index[1] - zero_curve.index[0]))
        for T in zero_curve.index
    })

def fra_rate(zero_curve: pd.Series, start: float, end: float) -> float:
    dfs = np.exp(-zero_curve * zero_curve.index)
    return (dfs[start] / dfs[end] - 1) / (end - start)

def black_scholes(F:float, K:float, sigma:float, T:float, P0T:float, option:str='call') -> float:
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option=='call':
        return P0T*(F*norm.cdf(d1) - K*norm.cdf(d2))
    else:
        return P0T*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

def bs_greeks(F:float, K:float, sigma:float, T:float, P0T:float) -> Dict:
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)
    return {
        'Delta': P0T*norm.cdf(d1),
        'Gamma': P0T*pdf/(F*sigma*np.sqrt(T)),
        'Vega': P0T*F*pdf*np.sqrt(T),
        'Theta': -P0T*F*pdf*sigma/(2*np.sqrt(T)),
        'Rho': T*P0T*(F*norm.cdf(d1) - K*norm.cdf(d2))
    }

def implied_vol(F:float, K:float, price:float, T:float, P0T:float) -> float:
    def f(vol): return black_scholes(F,K,vol,T,P0T) - price
    return newton(f, x0=0.2)

# ====================================================================
# E) Trader Quick Calculations
# ====================================================================

def carry_and_rolldown(coupon_rate:float, face:float, dt:float,
                       z_t:float, z_tdt:float) -> Dict:
    """
    Compute carry (coupon*dt*face) and rolldown ((z_t - z_tdt)*dt*face)
    """
    carry    = coupon_rate * face * dt
    rolldown = (z_t - z_tdt) * dt * face
    return {'carry': carry, 'rolldown': rolldown}

def breakeven_inflation(nominal_yield:float, real_yield:float) -> float:
    return (1+nominal_yield)/(1+real_yield) - 1

def swap_spread(swap_rate:float, treas_swap_rate:float) -> float:
    return swap_rate - treas_swap_rate

def price_from_dy(price:float, duration:float, dy:float) -> float:
    return price * (1 - duration * dy)

def dy_for_dprice(dprice:float, price:float, duration:float) -> float:
    return -dprice / (price * duration)

def carry_adjusted_yield(price:float, days:float) -> float:
    return ((100 - price)/price) * (360/days)

def generate_cashflows(settle:pd.Timestamp, maturity:pd.Timestamp,
                       coupon_rate:float, freq:int, face:float=100) -> pd.DataFrame:
    periods = int((maturity.year - settle.year)*freq +
                  (maturity.month - settle.month)/12*freq)
    dates = pd.date_range(start=settle, periods=periods+1,
                          freq=pd.DateOffset(months=int(12/freq)))
    cfs = [coupon_rate/freq*face]*(periods) + [coupon_rate/freq*face+face]
    return pd.DataFrame({'Date': dates, 'CF': cfs})

def bucket_dv01(key_rate_durs: pd.Series, buckets:Dict[str,List[float]], dv01:float) -> pd.Series:
    return pd.Series({
        name: dv01 * key_rate_durs.reindex(tenors).sum()
        for name, tenors in buckets.items()
    })

def spread_contribution(positions:pd.DataFrame, oas_changes:pd.Series,
                        dv01s:pd.Series) -> pd.Series:
    contrib = oas_changes * dv01s * positions['Qty']
    return contrib.sort_values(ascending=False)

def implied_repo(F:float, P:float, days:float) -> float:
    return -(F - P)/P * (360/days)

def vwap_tracker(prices:pd.Series, volumes:pd.Series) -> float:
    return (prices * volumes).sum() / volumes.sum()

# ====================================================================
# F) Hedging Calculators
# ====================================================================

def duration_hedge_notional(
    target_exposure: float,  # total DV01 to offset
    hedge_dv01: float        # DV01 per unit of hedge instrument
) -> float:
    """
    Compute notional units of hedge required to neutralize duration risk.
    """
    return - target_exposure / hedge_dv01

def convexity_hedge_notional(
    target_convexity: float, # portfolio convexity to offset
    hedge_convexity: float   # convexity per unit of hedge instrument
) -> float:
    return - target_convexity / hedge_convexity

def pca_factor_hedge(
    exposures: pd.Series,      # factor exposures of portfolio
    hedge_betas: pd.DataFrame   # betas: rows=factor names, cols=hedge instrument names
) -> pd.Series:
    # Solve hedge_betas.values @ weights = - exposures.values
    weights = pd.Series(
        np.linalg.pinv(hedge_betas.values) @ (- exposures.values),
        index=hedge_betas.columns
    )
    return weights

def regression_hedge_ratios(
    portfolio_returns: pd.Series,  # portfolio return series
    hedge_returns: pd.DataFrame    # hedge instrument return series
) -> pd.Series:
    import statsmodels.api as sm
    X = sm.add_constant(hedge_returns)
    model = sm.OLS(portfolio_returns, X).fit()
    return model.params.drop('const')

# End of Toolkit
