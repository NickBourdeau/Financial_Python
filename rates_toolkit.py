# RATES TRADING MARKET-MAKING TOOLKIT
A comprehensive Python scaffold and cheat sheet for summer projects on a rates trading market-making desk.

====================================================================
1. DATA INGESTION & IMPORT
--------------------------------------------------------------------
# CSV Import with date parsing and dtype enforcement
import pandas as pd
import numpy as np
import sqlalchemy as sa
import yfinance as yf

def load_csv(path):
    df = pd.read_csv(
        path,
        parse_dates=['Date'],       # parse 'Date' column as datetime
        index_col='Date',           # set 'Date' as DataFrame index
        dtype={'Yield': float},     # ensure 'Yield' is float
        na_values=['NA', '--']      # treat 'NA' and '--' as missing
    )
    return df

def load_sql(query, conn_string):
    engine = sa.create_engine(conn_string)                           # create DB engine
    df = pd.read_sql(query, engine, parse_dates=['trade_date'])      # fetch SQL with date
    return df

def fetch_yahoo(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)['Adj Close']  # get adjusted close prices
    return data

def fetch_fred(series_list, start, end, api_key):
    from fredapi import Fred
    fred = Fred(api_key=api_key)                                     # initialize FRED client
    df = pd.concat([fred.get_series(s, start, end).rename(s) for s in series_list], axis=1)
    return df

====================================================================
2. DATA CLEANING & PROCESSING
--------------------------------------------------------------------
import pandas as pd
import numpy as np

# 2.1 BASIC CLEANING PIPELINE
def clean_df(df):
    df = df.copy()
    df = df.sort_index()                                # ensure time order

    # 1) Missing data handling
    df = df.ffill().bfill()                             # forward/backward fill missing
    df = df.dropna(how='all')                           # drop rows all-NaN

    # 2) Duplicate removal
    df = df.drop_duplicates()                           # drop duplicate rows

    # 3) Outlier clipping
    lower = df.quantile(0.01)
    upper = df.quantile(0.99)
    df = df.clip(lower=lower, upper=upper, axis=1)

    return df

# 2.2 ADVANCED STRUCTURE MANIPULATION
def restructure_data(df):
    # Reset and set multi-index
    df = df.reset_index().set_index(['Date','Ticker'])   # multi-index on Date and Ticker

    # Pivot to wide format for curves
    wide = df['Yield'].unstack(level='Ticker')          # DataFrame[Date x Ticker]

    # Melt back to long
    long = wide.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Yield')

    # Group operations
    grp = long.groupby('Ticker')
    summary = grp['Yield'].agg(['mean','std','min','max'])  # descriptive stats per Ticker
    return wide, long, summary

# 2.3 FEATURE ENGINEERING UTILITIES
def feature_engineering(df):
    df_fe = df.copy()
    # Percent change (simple returns)
    df_fe['pct_change']   = df_fe['Yield'].pct_change()
    # Log returns
    df_fe['log_ret']      = np.log(df_fe['Yield'] / df_fe['Yield'].shift(1))

    # Rolling statistics
    df_fe['roll_mean_5']  = df_fe['Yield'].rolling(window=5).mean()
    df_fe['roll_std_5']   = df_fe['Yield'].rolling(window=5).std()
    df_fe['roll_skew_5']  = df_fe['Yield'].rolling(window=5).skew()

    # Exponential weighted metrics
    df_fe['ewma_10']      = df_fe['Yield'].ewm(span=10).mean()
    df_fe['ewma_var_10']  = df_fe['Yield'].ewm(span=10).var()

    # Lag features for panel
    for lag in range(1,6):
        df_fe[f'lag_{lag}'] = df_fe.groupby('Ticker')['Yield'].shift(lag)

    # Window transforms
    df_fe['pct_change_5'] = df_fe['Yield'].pct_change(periods=5)
    df_fe['min_rolling_10'] = df_fe['Yield'].rolling(10).min()
    df_fe['max_rolling_10'] = df_fe['Yield'].rolling(10).max()

    # Binning / discretization
    df_fe['yield_bin']    = pd.cut(df_fe['Yield'], bins=5, labels=False)

    # One-hot encoding
    df_fe = pd.get_dummies(df_fe, columns=['yield_bin'], prefix='bin')

    # Drop NaNs introduced by rolling/lag
    df_fe = df_fe.dropna()
    return df_fe

def ts_split(df, n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df))                       # returns indices
    return splits

====================================================================
3. DATA VISUALIZATION
--------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plot_time_series(df, cols=None):
    cols = cols or df.columns
    df[cols].plot(figsize=(10,5), title='Time Series')  # plot lines
    plt.xlabel('Date'); plt.ylabel('Value'); plt.show()

def plot_histogram(df, col):
    plt.hist(df[col], bins=50, density=True, alpha=0.6)  # density histogram
    plt.title(f'Histogram of {col}'); plt.show()

def plot_scatter(df, x, y):
    plt.scatter(df[x], df[y], alpha=0.5)                # scatter points
    plt.xlabel(x); plt.ylabel(y); plt.show()

def plot_corr_heatmap(df):
    corr = df.corr()                                     # compute correlation matrix
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap'); plt.show()

def plot_yield_curve(curve_df):
    for d in curve_df.index:
        plt.plot(curve_df.columns, curve_df.loc[d], label=d.strftime('%Y-%m'))  
    plt.legend(); plt.title('Yield Curves Over Time'); plt.show()

====================================================================
4. MODEL FITTING & EVALUATION
--------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)                         # train
    preds = model.predict(X_test)                       # predict
    return {
        'mse': mean_squared_error(y_test, preds),       # MSE
        'r2': r2_score(y_test, preds)                   # R^2
    }

def get_models():
    return {
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5),
        'GBM': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
        'XGB': XGBRegressor(n_estimators=200, learning_rate=0.05),
        'SVR': SVR(kernel='rbf', C=1.0, gamma=0.1)
    }

====================================================================
5. CURVE CONSTRUCTION & SENSITIVITIES
--------------------------------------------------------------------
import numpy as np

def bootstrap_zero(par_yields, maturities):
    zeros = np.zeros_like(par_yields)                    # init zero rates
    dfs = []                                              # discount factors list
    for i,(y,t) in enumerate(zip(par_yields, maturities)):
        if i == 0:
            zeros[i] = y                                 # first zero rate equals first par yield
            dfs.append(np.exp(-y*t))                     # discount factor
        else:
            fixed_leg = sum((y/maturities[i]) * dfs[j] for j in range(i))  
            d_i = (1 - fixed_leg)/(1 + y/maturities[i])   # solve discount factor
            dfs.append(d_i)
            zeros[i] = -np.log(d_i)/t                     # compute zero rate
    return zeros

def pv_cash(cfs, times, zeros):
    dfs = np.exp(-zeros * times)                         # discount factors
    return np.dot(cfs, dfs)                              # present value

def dv01_cash(cfs, times, zeros):
    base = pv_cash(cfs, times, zeros)                    # base PV
    bumped = np.exp(-(zeros+1e-4) * times)               # bump zeros by 1bp
    return np.dot(cfs, bumped) - base                    # DV01

def convexity_cash(cfs, times, zeros):
    dfs = np.exp(-zeros * times)
    return np.dot(cfs * times**2, dfs)                   # convexity measure

====================================================================
6. DERIVATIVES PRICING & MONTE CARLO
--------------------------------------------------------------------
import scipy.stats as ss
import numpy as np

def price_fra(zero, s, e, strike):
    P_s = np.exp(-zero[s]*s)                              # DF at start
    P_e = np.exp(-zero[e]*e)                              # DF at end
    delta = e - s                                         # accrual period
    fwd_rate = (P_s/P_e - 1)/delta                       # forward rate
    return (fwd_rate - strike) * delta * P_e             # FRA PV

def price_swap(zero, fixed, tenor):
    times = np.arange(1, tenor+1)
    dfs = np.exp(-zero[times] * times)                   # discount factors
    float_leg = 1 - dfs[-1]                              # floating leg PV
    fixed_leg = fixed * dfs.sum()                        # fixed leg PV
    return float_leg - fixed_leg                         # swap PV

def black_option(F, K, vol, T, P0T, option='call'):
    d1 = (np.log(F/K) + 0.5*vol**2*T)/(vol*np.sqrt(T))   # compute d1
    d2 = d1 - vol*np.sqrt(T)                             # compute d2
    if option=='call':
        return P0T*(F*ss.norm.cdf(d1) - K*ss.norm.cdf(d2))  
    else:
        return P0T*(K*ss.norm.cdf(-d2) - F*ss.norm.cdf(-d1))

def mc_hull_white(r0, kappa, theta, sigma, T, dt, n_paths):
    n_steps = int(T/dt)
    paths = np.zeros((n_paths, n_steps+1))
    paths[:,0] = r0                                      # initial rate
    for t in range(1, n_steps+1):
        z = np.random.randn(n_paths)                     # random shocks
        dr = kappa*(theta - paths[:,t-1])*dt + sigma*np.sqrt(dt)*z  # SDE update
        paths[:,t] = paths[:,t-1] + dr                   # new rate
    return paths

def mc_swap_pv(paths, fixed, tenor, dt):
    pv_list = []
    for path in paths:
        zero_path = path[:tenor]                         # short-rate path
        dfs = np.exp(-np.cumsum(zero_path)*dt)            # pathwise DF
        float_leg = 1 - dfs[-1]                          # floating leg PV
        fixed_leg = fixed * dfs[:tenor].sum()             # fixed leg PV
        pv_list.append(float_leg - fixed_leg)
    return np.mean(pv_list)                             # average PV

====================================================================
7. MARKET-MAKING ALGORITHMS
--------------------------------------------------------------------
import numpy as np

class MarketMaker:
    def __init__(self, spread, max_inv, k_inv, k_time):
        self.spread = spread              # base bid-ask spread
        self.max_inv = max_inv            # maximum inventory limit
        self.k_inv = k_inv                # inventory risk coefficient
        self.k_time = k_time              # time skew coefficient
        self.inventory = 0                # current inventory

    def quote(self, mid_price, t, T):
        inv_adj = self.k_inv * self.inventory
        time_adj = self.k_time * (t / T)
        bid = mid_price - (self.spread/2 + inv_adj + time_adj)
        ask = mid_price + (self.spread/2 + inv_adj + time_adj)
        return bid, ask

    def execute(self, mid_price, t, T):
        bid, ask = self.quote(mid_price, t, T)
        flow = np.random.choice([-1,1])  # random client order size
        price = bid if flow > 0 else ask  # execution price
        if abs(self.inventory + flow) > self.max_inv:
            flow = -np.sign(self.inventory)  # unwind position
        self.inventory += flow
        mtm = self.inventory * mid_price   # mark-to-market value
        pnl = -flow*price + mtm            # PnL from trade + inventory
        return pnl, self.inventory

def simulate_mm(n_steps, mm: MarketMaker, r0, kappa, theta, sigma, dt):
    rates = mc_hull_white(r0, kappa, theta, sigma, T=n_steps*dt, dt=dt, n_paths=1)[0]
    mid_prices = 100 + np.cumsum(rates*dt)             # synthetic mid-price series
    pnls = []
    for t, mid in enumerate(mid_prices):
        pnl,_ = mm.execute(mid, t, n_steps)
        pnls.append(pnl)
    return np.array(pnls)                               # PnL series

====================================================================
8. TIME-SERIES & RATE MODELS
--------------------------------------------------------------------
import statsmodels.api as sm
from arch import arch_model
import numpy as np

def fit_ar1(series):
    model = sm.tsa.ARIMA(series, order=(1,0,0)).fit()  # ARIMA(1,0,0)
    return model

def fit_garch(series):
    am = arch_model(series, vol='Garch', p=1, q=1, mean='Zero')
    res = am.fit(disp='off')                            # fit model quietly
    return res

def fit_vasicek(series, dt):
    dr = series.diff().dropna()
    r_lag = series.shift(1).dropna()
    ols_res = sm.OLS(dr, sm.add_constant(r_lag)).fit()
    const, coef = ols_res.params
    a = -coef/dt
    b = -const/coef
    sigma = np.sqrt(ols_res.scale * 2*a/(1 - np.exp(-2*a*dt)))
    return a, b, sigma

def hjm_drift(vol_func, tenor_grid):
    def mu(t):
        return sum(vol_func(t, T) * vol_func(t, u)
                   for u in tenor_grid if u > t)
    return mu

====================================================================
9. ML FORECASTING & OTHER MODELS
--------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_rf(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, max_depth=5)
    rf.fit(X_train, y_train)                            # train RF
    return rf

def train_xgb(X_train, y_train):
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    xgb.fit(X_train, y_train)                          # train XGB
    return xgb

class VolNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)                # layer1
        self.fc2 = nn.Linear(64, 32)                    # layer2
        self.out = nn.Linear(32, 1)                     # output
    def forward(self, x):
        x = torch.relu(self.fc1(x))                     # ReLU1
        x = torch.relu(self.fc2(x))                     # ReLU2
        return self.out(x).squeeze(-1)                  # forecast

def train_nn(model, X, y, lr=1e-3, epochs=100):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for xb, yb in loader:
            preds = model(xb.float())
            loss = loss_fn(preds, yb.float())
            opt.zero_grad(); loss.backward(); opt.step()
    return model

====================================================================
10. USEFUL PACKAGES & FUNCTIONS
--------------------------------------------------------------------
import QuantLib as ql
from pandas_datareader import data as pdr
from arch import arch_model
import statsmodels.api as sm
from scipy.stats import norm, t, bernoulli, beta, gamma, expon, uniform
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import pyfolio as pf
import empyrical as emp

====================================================================
11. PROBABILITY DISTRIBUTIONS
--------------------------------------------------------------------
from scipy.stats import norm, t, bernoulli, beta, gamma, expon, uniform

# Normal Distribution
dist = norm(loc=0, scale=1)        # standard normal
pdf_val = dist.pdf(0.5)             # PDF at x=0.5
cdf_val = dist.cdf(0.5)             # CDF at x=0.5

# Student's t
t_dist = t(df=5, loc=0, scale=1)
t_pdf = t_dist.pdf(1.0)

# Bernoulli
b_dist = bernoulli(p=0.3)
b_pmf = b_dist.pmf(k=1)

# Beta
beta_dist = beta(a=2, b=5)
beta_pdf = beta_dist.pdf(0.5)

# Gamma
gamma_dist = gamma(a=2, scale=2)
gamma_pdf = gamma_dist.pdf(1.0)

# Exponential
exp_dist = expon(scale=1)
exp_pdf = exp_dist.pdf(0.5)

# Uniform
u_dist = uniform(loc=0, scale=1)
u_pdf = u_dist.pdf(0.5)

====================================================================
12. SQL QUICK REFERENCE
--------------------------------------------------------------------
-- Select columns and rows
SELECT date, yield, ticker
FROM treasury_yields
WHERE date >= '2020-01-01'
  AND ticker IN ('US10Y', 'US5Y');

-- Aggregate: average yield by ticker
SELECT ticker, AVG(yield) AS avg_yield
FROM treasury_yields
GROUP BY ticker;

-- Window function: rolling average
SELECT date,
       yield,
       AVG(yield) OVER (PARTITION BY ticker ORDER BY date ROWS 4 PRECEDING) AS rolling_avg_5
FROM treasury_yields;

-- Join with macro table
SELECT a.date, a.yield, m.cpi
FROM treasury_yields a
LEFT JOIN macro_data m ON a.date = m.date;

-- Create table example
CREATE TABLE sample_data (
    date DATE,
    ticker VARCHAR(10),
    yield FLOAT,
    volume BIGINT
);

====================================================================
13. MISCELLANEOUS PYTHON TOOLS & TRICKS
--------------------------------------------------------------------
# OS & Path
import os
from pathlib import Path
home = Path.home() / 'data'
os.makedirs(home, exist_ok=True)

# JSON & ENV
import json
import yaml
from dotenv import load_dotenv
load_dotenv()
config = json.load(open('config.json'))
cfg_yaml = yaml.safe_load(open('config.yaml'))

# Logging & Debug
import logging
from tqdm import tqdm
import pdb
logging.basicConfig(level=logging.INFO)
for i in tqdm(range(10)):
    if i == 5: pdb.set_trace()

# Caching & Timing
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        logging.info(f"{func.__name__} took {time.time()-start:.4f}s")
        return res
    return wrapper

# Data Structures & Iteration
squares = [x*x for x in range(10)]
square_map = {x: x*x for x in range(10)}
for idx, val in enumerate(squares): pass
for a, b in zip([1,2],[‘a’,’b’]): pass

# Context Managers
with open('file.txt','r') as f:
    content = f.read()

# CLI Args
import sys
args = sys.argv[1:]
if '--debug' in args: logging.getLogger().setLevel(logging.DEBUG)

# Serialization
import pickle
pickle.dump({'a':1}, open('obj.pkl','wb'))
obj = pickle.load(open('obj.pkl','rb'))

# Numeric Tricks
arr = np.arange(10)
even = arr[arr % 2 == 0]
b = arr + 5


====================================================================
14. DATA ANALYSIS UTILITIES
--------------------------------------------------------------------
import pandas as pd
import numpy as np

# 14.1 INDEX & DATETIME OPERATIONS
# --------------------------------------------------
df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=5), 'Value': range(5)})
df = df.set_index('Date')                        # set datetime index

df.resample('M').mean()                            # monthly aggregation
ndays = df.index.day                              # day-of-month array

df.asfreq('D', method='pad')                      # enforce daily freq, forward-fill missing

# 14.2 MULTI-LEVEL INDEXING
# --------------------------------------------------
arrays = [np.array(['A','A','B','B']), np.array([1,2,1,2])]
mi_df = pd.DataFrame({'Value': [10,20,30,40]}, index=arrays)
mi_df.index.names = ['Group','ID']
mi_df.xs('A')                                      # cross-section on first level

# 14.3 GROUPBY & AGGREGATION
# --------------------------------------------------
df = pd.DataFrame({'Ticker':['A','A','B','B'], 'Yield':[1,2,3,4], 'Vol':[0.1,0.2,0.3,0.4]})

# aggregate multiple functions
g = df.groupby('Ticker').agg({'Yield':['mean','std'], 'Vol':'max'})

# apply custom lambda
df['demeaned'] = df.groupby('Ticker')['Yield'].transform(lambda x: x - x.mean())

# rolling within groups
rolling_max = df.groupby('Ticker')['Yield'].rolling(window=2).max().reset_index(level=0, drop=True)

# 14.4 PIVOT / MELT / MERGE
# --------------------------------------------------
wide = df.pivot(index='Ticker', columns='Yield', values='Vol')  # pivot
long = wide.reset_index().melt(id_vars='Ticker', var_name='Yield', value_name='Vol')

# merge two dataframes
df2 = pd.DataFrame({'Ticker':['A','B'], 'Sector':['X','Y']})
merged = pd.merge(df, df2, on='Ticker', how='left')

# join on index
df1 = df.set_index('Ticker')
df2 = df2.set_index('Ticker')
joined = df1.join(df2)

# 14.5 WINDOW & EXPANDING APPLIES
# --------------------------------------------------
s = pd.Series(np.random.randn(10))
s.rolling(window=3).apply(lambda x: x.mean())  # custom rolling apply
s.expanding(min_periods=1).max()               # expanding max

# 14.6 APPLY / MAP / LAMBDA
# --------------------------------------------------
df['Yield_str'] = df['Yield'].map(lambda x: f"{x:.2%}")  # format as percentage string

# elementwise apply
df['Vol_norm'] = df['Vol'].apply(lambda x: (x - df['Vol'].mean())/df['Vol'].std())

# 14.7 CATEGORICAL ENCODING
# --------------------------------------------------
df['Category'] = pd.cut(df['Yield'], bins=[0,1,2,3,4], labels=['Low','Med','High','VeryHigh'])
dum = pd.get_dummies(df['Category'], prefix='Cat')  # one-hot encode

# 14.8 SAMPLING & SUBSETTING
# --------------------------------------------------
df.sample(n=2)                                  # random sample of rows
df[df['Yield']>2]                              # boolean mask subset

# 14.9 STRING METHODS
# --------------------------------------------------
df_text = pd.DataFrame({'Name':['apple','banana','cherry']})
df_text['Name_upper'] = df_text['Name'].str.upper()  # vectorized string op

# 14.10 CUSTOM TRANSFORMS
# --------------------------------------------------
def zscore(x): return (x - x.mean())/x.std()
df['Yield_z'] = df.groupby('Ticker')['Yield'].transform(zscore)