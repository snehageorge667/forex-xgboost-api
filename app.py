from fastapi import FastAPI
from utils.feature_engineering import create_features

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from datetime import datetime, timedelta

app = FastAPI(title="Forex XGBoost Prediction API")

# load trained model
xgb_model = joblib.load("model.pkl")


def download_data():

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=120)).strftime('%Y-%m-%d')

    raw = yf.download("INR=X", start=start_date, end=end_date)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.reset_index(inplace=True)

    raw.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close'
    }, inplace=True)

    raw = raw.sort_values('date')

    return raw


@app.get("/")
def home():
    return {"message": "Forex XGBoost Forecast API is running"}


@app.get("/predict/xgboost")
def predict_xgboost():

    # download latest data
    raw = download_data()

    # create features
    raw = create_features(raw)

    # model feature list 
    feature_cols = [
        'cc_return_lag_1','cc_return_lag_2','cc_return_lag_3','cc_return_lag_5','cc_return_lag_10',
        'oc_return_lag_1','oc_return_lag_2','oc_return_lag_3','oc_return_lag_5','oc_return_lag_10',
        'gap_lag_1','gap_lag_2','gap_lag_3',
        'hl_spread_lag_1','hl_spread_lag_2','hl_spread_lag_3',
        'close_pos_lag_1','close_pos_lag_2','close_pos_lag_3',
        'return_ma_3','return_ma_7','return_ma_14',
        'return_std_5','return_std_7','return_std_14',
        'hl_ma_5','hl_ma_14',
        'return_accel_1','return_accel_3','return_accel_5',
        'return_ewm_5','return_ewm_10',
        'rsi_14','bb_position',
        'USDAUD','USDCAD','USDCHF','USDCNY','USDEUR','USDGBP','USDINR','USDJPY','USDNPR',
        'day','month','year','dayofweek','quarter'
    ]

    X = raw[feature_cols].iloc[-1:]

    forecast = xgb_model.predict(X)

    # convert log return prediction → price
    last_close = raw['close'].iloc[-1]
    predicted_price = last_close * np.exp(forecast[0])

    # prediction date = today (user request date)
    prediction_date = datetime.today().strftime("%Y-%m-%d")

    return {
        "model": "XGBoost",
        "prediction_date": prediction_date,
        "predicted_close": float(predicted_price)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)