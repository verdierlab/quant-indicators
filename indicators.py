"""
quant-indicators

A lightweight collection of financial indicators implemented with
Pandas and NumPy.

Author: Verdiér
License: MIT
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    """
    Calculate the MACD indicator.

    Returns
    -------
    tuple
        MACD line, Signal line and Histogram.
    """

    ema_fast = ema(series, 12)
    ema_slow = ema(series, 26)

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: int = 2
):
    """
    Calculate Bollinger Bands.
    """

    middle = sma(series, period)

    std = series.rolling(period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate the Average True Range (ATR).
    """

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def vwap(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price.
    """

    return (close * volume).cumsum() / volume.cumsum()


def simple_returns(series: pd.Series) -> pd.Series:
    """
    Calculate simple returns.
    """

    return series.pct_change()


def volatility(
    series: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Rolling historical volatility.
    """

    returns = series.pct_change()

    return returns.rolling(period).std() * np.sqrt(period)
