import pandas as pd
import numpy as np

eps = 1e-12

def create_features(df):

    # ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    df['cc_log_return']  = np.log(df['close'] + eps).diff()
    df['oc_log_return']  = np.log((df['close'] + eps) / (df['open'] + eps))
    df['overnight_gap']  = np.log((df['open'] + eps) / (df['close'].shift(1) + eps))

    df['hl_spread_pct']  = (df['high'] - df['low']) / (df['open'] + eps)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)

    df['upper_shadow_pct'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['open'] + eps)
    df['lower_shadow_pct'] = (df[['open','close']].min(axis=1) - df['low'])   / (df['open'] + eps)

    for lag in [1,2,3,5,10]:
        df[f'cc_return_lag_{lag}'] = df['cc_log_return'].shift(lag)

    for lag in [1,2,3,5,10]:
        df[f'oc_return_lag_{lag}'] = df['oc_log_return'].shift(lag)

    for lag in [1,2,3]:
        df[f'gap_lag_{lag}'] = df['overnight_gap'].shift(lag)

    for lag in [1,2,3]:
        df[f'hl_spread_lag_{lag}'] = df['hl_spread_pct'].shift(lag)
        df[f'close_pos_lag_{lag}'] = df['close_position'].shift(lag)

    for w in [3,7,14]:
        df[f'return_ma_{w}'] = df['cc_log_return'].rolling(w).mean()

    for w in [5,7,14]:
        df[f'return_std_{w}'] = df['cc_log_return'].rolling(w).std()

    for w in [5,14]:
        df[f'hl_ma_{w}'] = df['hl_spread_pct'].rolling(w).mean()

    for lag in [1,3,5]:
        df[f'return_accel_{lag}'] = df['cc_log_return'].diff(lag)

    for span in [5,10]:
        df[f'return_ewm_{span}'] = df['cc_log_return'].ewm(span=span, adjust=False).mean()

    delta = df['close'].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ret_ma14  = df['cc_log_return'].rolling(14).mean()
    ret_std14 = df['cc_log_return'].rolling(14).std()

    bb_upper = ret_ma14 + 2 * ret_std14
    bb_lower = ret_ma14 - 2 * ret_std14

    df['bb_position'] = (df['cc_return_lag_1'] - bb_lower) / (bb_upper - bb_lower + eps)

    df['USDINR'] = True
    df['USDAUD'] = False
    df['USDCAD'] = False
    df['USDCHF'] = False
    df['USDCNY'] = False
    df['USDEUR'] = False
    df['USDGBP'] = False
    df['USDJPY'] = False
    df['USDNPR'] = False

    # ---------- Calendar features ----------
    df['day']       = df['date'].dt.day
    df['month']     = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter']   = df['date'].dt.quarter

    df = df.dropna().reset_index(drop=True)

    return df