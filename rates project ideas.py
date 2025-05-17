# Multi-Desk Project Toolkit (Fully Commented)
# ================================================
# This Python file provides scaffolded functions for various Rates-Trading
# (market-making) and Corporate-Credit intern projects. Every function and
# code block is annotated with detailed comments explaining inputs,
# processing steps, and outputs. Use this as a reference or starting point
# for any data-driven desk assignment.

# ====================================================================
# Section 1: Yield Curve Bootstrapping & Visualization
# ====================================================================

import pandas as pd            # pandas for DataFrames
import numpy as np             # numpy for numerical operations
import matplotlib.pyplot as plt  # matplotlib for plotting
from typing import List        # type hints for lists

# Function: fit_zero_curve
# -------------------------
# Purpose:
#   Compute zero-coupon rates at specified maturities by bootstrapping
#   from par yields for each date.
# Inputs:
#   par_df: DataFrame with columns --
#     'Date' (datetime-like), 'Ticker' (string), 'ParYield' (float decimal)
#   maturities: list of floats (years) corresponding to each Ticker
# Returns:
#   zero_df: DataFrame indexed by Date, columns equal maturities,
#            values are zero rates (floats)
def fit_zero_curve(
    par_df: pd.DataFrame,
    maturities: List[float]
) -> pd.DataFrame:
    # 1) Copy input to avoid side effects
    df = par_df.copy()
    # 2) Ensure 'Date' column is proper datetime with timezone
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # 3) Remove rows missing ParYield, since bootstrapping needs yields
    df = df.dropna(subset=['ParYield'])
    # 4) Pivot to wide: rows=dates, cols=tickers, cells=par yields
    pivot = df.pivot(index='Date', columns='Ticker', values='ParYield')
    # 5) Prepare container for zero rates arrays
    zeros_list = []
    # 6) Loop through each date to run bootstrap
    for date, row in pivot.iterrows():
        # a) Extract par yields as numpy array
        par_yields = row.values  # shape = (n_tickers,)
        # b) Call external bootstrap function (not shown here)
        zeros = bootstrap_zero(par_yields, maturities)  # returns np.ndarray
        # c) Append zeros array to list
        zeros_list.append(zeros)
    # 7) Stack list into 2D array (n_dates x n_maturities)
    zeros_matrix = np.vstack(zeros_list)
    # 8) Create DataFrame: index=dates, columns=maturities
    zero_df = pd.DataFrame(
        data=zeros_matrix,
        index=pivot.index,
        columns=maturities
    )
    # 9) Return full zero-coupon rate DataFrame
    return zero_df

# Function: plot_zero_curve
# -------------------------
# Purpose:
#   Visualize zero-coupon curves for selected dates
# Inputs:
#   zero_df: DataFrame output from fit_zero_curve
#   dates: Optional list of Timestamps; defaults to last 3 dates
# Outputs:
#   Displays a matplotlib line plot

def plot_zero_curve(
    zero_df: pd.DataFrame,
    dates: List[pd.Timestamp] = None
) -> None:
    # 1) Default selection: last 3 dates if none provided
    if dates is None:
        dates = list(zero_df.index[-3:])
    # 2) Initialize figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # 3) Plot each date's zero curve
    for dt in dates:
        rates = zero_df.loc[dt].values  # zero rates vector
        ax.plot(
            zero_df.columns,            # maturities on x-axis
            rates,                      # rates on y-axis
            label=str(dt.date())        # label for legend
        )
    # 4) Label axes and title
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Zero Rate (decimal)')
    ax.set_title('Zero-Coupon Yield Curves')
    # 5) Show legend and plot
    ax.legend(); plt.show()

# ====================================================================
# Section 2: Intraday P&L Attribution
# ====================================================================

import pandas as pd
import numpy as np

# Function: clean_trades
# ----------------------
# Purpose:
#   Standardize and sort trade blotter
# Inputs:
#   trades_df: DataFrame with ['Time','Ticker','Qty','Price']
# Returns:
#   Cleaned DataFrame sorted by time

def clean_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    # Copy data
    df = trades_df.copy()
    # Parse 'Time' as datetime with UTC
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    # Drop exact duplicate rows
    df = df.drop_duplicates()
    # Sort by time and reset index
    df = df.sort_values('Time').reset_index(drop=True)
    return df

# Function: clean_quotes
# ----------------------
# Purpose:
#   Fill missing bid/ask quotes in a time series
# Inputs:
#   quotes_df: DataFrame with ['Time','Ticker','Bid','Ask']
# Returns:
#   Forward/backward-filled quotes per ticker
def clean_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    # Copy data
    q = quotes_df.copy()
    # Parse time
    q['Time'] = pd.to_datetime(q['Time'], utc=True)
    # Sort by ticker and time
    q = q.sort_values(['Ticker', 'Time'])
    # Fill missing Bid/Ask within each ticker
    q[['Bid','Ask']] = q.groupby('Ticker')[['Bid','Ask']].ffill().bfill()
    return q

# Function: pnl_attribution
# -------------------------
# Purpose:
#   Merge trades and quotes to calculate PnL components
# Inputs:
#   trades: cleaned trades DataFrame
#   quotes: cleaned quotes DataFrame
#   zero_df: zero rates for sensitivity (if needed)
# Returns:
#   DataFrame with PnL, with placeholders for sub-components
def pnl_attribution(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    zero_df: pd.DataFrame
) -> pd.DataFrame:
    # Merge-asof: assign most recent quote to each trade
    merged = pd.merge_asof(
        trades.sort_values('Time'),
        quotes.sort_values('Time'),
        by='Ticker',
        left_on='Time',
        right_on='Time',
        direction='backward'
    )
    # Compute mid price = (Bid+Ask)/2
    merged['Mid'] = 0.5 * (merged['Bid'] + merged['Ask'])
    # Compute mark-to-market: Qty * Mid price
    merged['MTM'] = merged['Qty'] * merged['Mid']
    # Initialize placeholders for PnL attribution
    merged['CurveMove'] = np.nan
    merged['SpreadCapture'] = np.nan
    # Return key columns\    
    return merged[['Time','Ticker','Qty','Price','Mid','MTM','CurveMove','SpreadCapture']]

# ====================================================================
# Section 3: Volatility Surface & ATM Vol Forecasting
# ====================================================================

import pandas as pd
import numpy as np
from arch import arch_model

# Function: clean_options
# -----------------------
# Purpose:
#   Prepare option quotes for surface modeling
# Inputs:
#   opt_df: DataFrame ['Date','Expiry','Strike','MidVol']
# Returns:
#   Filtered DataFrame with valid expiries and vols
def clean_options(opt_df: pd.DataFrame) -> pd.DataFrame:
    # Copy and parse
    df = opt_df.copy()
    df['Date']   = pd.to_datetime(df['Date'], utc=True)
    df['Expiry'] = pd.to_datetime(df['Expiry'], utc=True)
    # Drop missing vol rows
    df = df.dropna(subset=['MidVol'])
    # Keep only expiries after quote date
    df = df[df['Expiry'] > df['Date']]
    return df

# Function: fit_atm_garch
# -----------------------
# Purpose:
#   Fit a GARCH(1,1) on ATM vol series to forecast volatility
# Inputs:
#   vol_series: Series indexed by date, values = ATM vol (decimal)
# Returns:
#   arch_model result object def fit_atm_garch(vol_series: pd.Series):
    # Convert to percentage for arch library
    series_pct = vol_series * 100
    # Initialize model
    am = arch_model(series_pct, vol='Garch', p=1, q=1, mean='Zero')
    # Fit quietly
    res = am.fit(disp='off')
    return res

# ====================================================================
# Section 4: Credit Spread Screener & Alerts
# ====================================================================

import pandas as pd
import numpy as np

# Function: compute_oas
# ---------------------
# Purpose:
#   Compute Option-Adjusted Spread (OAS) relative to treasury zero curve
# Inputs:
#   bonds_df: ['Date','Ticker','BidYld','AskYld']
#   zero_df: treasury zero rates DataFrame
# Returns:
#   DataFrame ['Date','Ticker','OAS']
def compute_oas(bonds_df: pd.DataFrame, zero_df: pd.DataFrame) -> pd.DataFrame:
    # Copy and parse
    df = bonds_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # Mid yield
    df['MidYld'] = 0.5 * (df['BidYld'] + df['AskYld'])
    # Flatten zero_df for merge: columns ['Date','Ticker','Zero']
    zero = zero_df.stack().rename('Zero').reset_index()
    # Merge on date & ticker
    merged = df.merge(zero, on=['Date','Ticker'], how='left')
    # Calculate OAS
    merged['OAS'] = merged['MidYld'] - merged['Zero']
    return merged[['Date','Ticker','OAS']]

# Function: flag_spreads
# ----------------------
# Purpose:
#   Identify cheap vs rich bonds based on rolling OAS percentiles
# Inputs:
#   spreads_df: ['Date','Ticker','OAS']
#   window: lookback window length
#   low_pct/high_pct: percentile thresholds
# Returns:
#   spreads_df with 'Flag' column
def flag_spreads(spreads_df: pd.DataFrame, window: int=100, low_pct: float=0.05, high_pct: float=0.95) -> pd.DataFrame:
    df = spreads_df.copy().sort_values(['Ticker','Date'])
    # Rolling low percentile
    df['Lo'] = df.groupby('Ticker')['OAS'].transform(lambda x: x.rolling(window).quantile(low_pct))
    # Rolling high percentile
    df['Hi'] = df.groupby('Ticker')['OAS'].transform(lambda x: x.rolling(window).quantile(high_pct))
    # Flag cheap/rich
    df['Flag'] = np.where(df['OAS'] < df['Lo'], 'Rich', np.where(df['OAS'] > df['Hi'], 'Cheap', None))
    return df

# ====================================================================
# Section 5: Merton Model Default Probability
# ====================================================================

import numpy as np
from scipy.stats import norm

# Function: merton_pd
# -------------------
# Purpose:
#   Estimate probability of default over horizon T
# Inputs:
#   equity_price: market value of equity
#   equity_vol: annual vol (decimal)\#   debt_face: face value of debt\#   T: time horizon (years)
# Returns:
#   pd: probability default

def merton_pd(equity_price: float, equity_vol: float, debt_face: float, T: float=1.0) -> float:
    # Estimate asset value V0
    V0 = equity_price + debt_face
    # Implied asset vol
    sigma = equity_vol * (equity_price / V0)
    # Compute d2
    d2 = (np.log(V0/debt_face) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    # PD = N(-d2)
    return norm.cdf(-d2)

# ====================================================================
# Section 6: PCA on Yield Curves
# ====================================================================

import numpy as np
from sklearn.decomposition import PCA

# Function: pca_yield_curve
# -------------------------
# Purpose:
#   Reduce zero-curve dimensionality via PCA
# Inputs:
#   zero_df: DataFrame index=Date cols=maturities
#   n_components: number of PCs
# Returns:
#   pca: fitted PCA object, scores: array (n_dates,n_components)
def pca_yield_curve(zero_df: pd.DataFrame, n_components: int=3):
    pca = PCA(n_components=n_components)              # init PCA
    scores = pca.fit_transform(zero_df.values)        # fit & transform
    return pca, scores

# ====================================================================
# Section 7: Regression Forecasting (OLS, Ridge, Lasso)
# ====================================================================

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Function: regression_forecast
# ------------------------------
# Purpose:
#   Fit and evaluate linear models
# Inputs:
#   X_train, y_train, X_test, y_test, model_type
# Returns:
#   dict with 'mse','r2','preds'
def regression_forecast(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model_type: str='Ridge') -> dict:
    # Select model
    models = {'OLS': LinearRegression(), 'Ridge': Ridge(alpha=1.0), 'Lasso': Lasso(alpha=0.1)}
    model = models.get(model_type, LinearRegression())
    # Fit
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    # Metrics
    return {
        'mse': mean_squared_error(y_test, preds),
        'r2': r2_score(y_test, preds),
        'preds': preds
    }

# ====================================================================
# Section 8: ML Models (RF & XGB)
# ====================================================================

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Function: ml_forecast
# ---------------------
# Purpose:
#   Train RF and XGB and return preds & importances
def ml_forecast(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> dict:
    # Initialize models
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    # Fit
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    # Return preds and feature importances
    return {
        'rf_preds': rf.predict(X_test),
        'xgb_preds': xgb.predict(X_test),
        'rf_imp': rf.feature_importances_,
        'xgb_imp': xgb.feature_importances_
    }

# ====================================================================
# Section 9: Screeners & Alerts (Liquidity & Parameter Sweeps)
# ====================================================================

import pandas as pd
import numpy as np
import itertools

# Function: liquidity_monitor
# ---------------------------
# Purpose:
#   Compute rolling avg spread and flag anomalies
def liquidity_monitor(quotes_df: pd.DataFrame, window: int=50) -> pd.DataFrame:
    df = quotes_df.copy()
    # Spread calculation
    df['Spread'] = df['Ask'] - df['Bid']
    # Rolling average per ticker
    df['RollAvg'] = df.groupby('Ticker')['Spread'].transform(lambda x: x.rolling(window).mean())
    # Alert when spread > 2x rolling avg
    df['Alert'] = df['Spread'] > 2 * df['RollAvg']
    return df

# Function: parameter_sweep
# -------------------------
# Purpose:
#   Evaluate a function over grid of parameters
def parameter_sweep(func, param_grid: dict) -> pd.DataFrame:
    rows = []
    for vals in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), vals))
        out = func(**params)
        rows.append({**params, 'output': out})
    return pd.DataFrame(rows)

# ====================================================================
# Section 10: Automated Risk Summary Report
# ====================================================================

import pandas as pd
from datetime import datetime

# Function: generate_risk_summary
# -------------------------------
# Purpose:
#   Build a simple PV01/DV01 summary for a portfolio
def generate_risk_summary(
    positions_df: pd.DataFrame,  # ['Date','Ticker','Qty','Price']
    zero_df: pd.DataFrame        # zero rates for sensitivities (optional)
) -> pd.DataFrame:
    df = positions_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    # DV01 ≈ Price * Qty * 0.0001
    df['DV01'] = df['Price'] * df['Qty'] * 0.0001
    total_dv01 = df['DV01'].sum()
    # Assemble final report
    report = pd.DataFrame({
        'AsOf': [datetime.utcnow()],
        'Total DV01': [total_dv01]
    })
    return report

# ====================================================================
# Section 11: Interactive Dashboard (Streamlit)
# ====================================================================

import streamlit as st
import pandas as pd

# Function: dashboard_app
# -----------------------
# Purpose:
#   Launch a multi-tab Streamlit dashboard for key tools
def dashboard_app():
    st.title('Intern Analytics Dashboard')
    # Sidebar for navigation
    choice = st.sidebar.radio('Section', ['Curve', 'PnL', 'Vol Forecast', 'Risk Summary'])
    if choice == 'Curve':
        st.header('Zero Curve Fitting')
        uploaded = st.file_uploader('Upload Par Yields CSV')
        if uploaded:
            par_df = pd.read_csv(uploaded)
            # User enters maturities list manually
            maturities = st.text_input('Enter maturities list (comma-separated)')
            # Fit and plot
    elif choice == 'PnL':
        st.header('PnL Attribution')
        # Similar uploads for trades & quotes
    elif choice == 'Vol Forecast':
        st.header('ATM Vol Forecast')
        # Upload option data + run GARCH
    else:
        st.header('Risk Summary')
        # Upload positions, show DV01 report

# ====================================================================
# Section 12: Quick Quant Metrics
# ====================================================================

import numpy as np

# Function: calc_dv01
# -------------------
# Purpose: approximate DV01 for a bond
# Inputs:
#   price: bond price
#   yield_rate: decimal
#   duration: Macaulay duration
# Returns: DV01 value
def calc_dv01(price: float, yield_rate: float, duration: float) -> float:
    # DV01 ≈ Duration × Price × 0.0001
    return duration * price * 0.0001

# Function: macaulay_duration
# ---------------------------
# Purpose: compute weighted average time to cashflows
def macaulay_duration(cashflows: np.ndarray, times: np.ndarray, yield_rate: float) -> float:
    # Discount factors for each CF
    dfs = np.exp(-yield_rate * times)
    # Present value of each CF
    pv_cfs = cashflows * dfs
    # Weighted times
    weighted = times * pv_cfs
    # Sum weighted times / total PV
    return weighted.sum() / pv_cfs.sum()

# Function: modified_duration
# ---------------------------
# Purpose: adjust Macaulay for yield compounding
def modified_duration(macaulay_dur: float, yield_rate: float) -> float:
    return macaulay_dur / (1 + yield_rate)

# Function: spread_dv01
# ---------------------
# Purpose: compute spread DV01 for credit instruments
def spread_dv01(oas: float, duration: float, price: float) -> float:
    # Approx same as DV01 formula
    return duration * price * 0.0001

# End of Toolkit
# ===============================================
