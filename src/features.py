import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# Stochastic oscillator
def stochastic(df, k_window=14, d_window=3):
    df = df.copy()

    low_min = df['EMA_Low'].rolling(k_window).min()
    high_max = df['EMA_High'].rolling(k_window).max()

    df['stoch_k'] = 100 * (df['EMA_Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(d_window).mean()

    return df

# RSI
def RSI(df, k_window):
    df = df.copy()

    delta = df["EMA_Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (Exponential moving average)
    avg_gain = gain.ewm(alpha=1/k_window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/k_window, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

# Williams %R
def Williams(df, k_window):
  df = df.copy()

  # HH and LL for 14 day window
  high_max = df["EMA_High"].rolling(k_window).max()
  low_min = df["EMA_Low"].rolling(k_window).min()

  df["Williams %R"] = ((high_max - df["EMA_Close"]) / (high_max - low_min)) * 100

  return df

# MACD
def MACD(df, fast=12, slow=26, signal=9):
    df = df.copy()

    # Exponential moving averages
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    # MACD line
    df["MACD"] = ema_fast - ema_slow

    # Signal line
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    # Histogram
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df

# OBV
def OBV(df):
    df = df.copy()

    # Price change direction
    direction = np.sign(df["EMA_Close"].diff())

    # Replace NaN (first row) with 0
    direction = direction.fillna(0)

    # Volume with direction
    obv_change = df["Volume"] * direction

    # Cumulative sum
    df["OBV"] = obv_change.cumsum()

    return df

# Price rate of change
def ROC(df, k_window):
  df = df.copy()

  df["ROC (Rate of Change)"] = 100 * ((df["EMA_Close"] - df["EMA_Close"].shift(k_window)) / (df["EMA_Close"].shift(k_window)))

  return df


def full_predictors(df):
    df = df.copy()

    df = stochastic(df, 14, 3)
    df = Williams(df, 14)
    df = RSI(df, k_window=14)
    df = MACD(df, fast=12, slow=26, signal=9)
    df = OBV(df)
    df = ROC(df, 14)


    df["return_5"] = df["EMA_Close"].pct_change(5)
    df["return_14"] = df["EMA_Close"].pct_change(14)
    df["return_90"] = df["EMA_Close"].pct_change(90)
    df["volatility"] = df["EMA_Close"].rolling(7).std()
    df["momentum_7"] = df["EMA_Close"] / df["EMA_Close"].shift(7)
    df["momentum_30"] = df["EMA_Close"] / df["EMA_Close"].shift(30)
    df["momentum_90"] = df["EMA_Close"] / df["EMA_Close"].shift(90)

    df = df.dropna()

    return df