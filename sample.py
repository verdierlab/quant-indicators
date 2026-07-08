import numpy as np
import pandas as pd

from indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    simple_returns,
    sma,
    volatility,
    vwap,
)


np.random.seed(42)

prices = pd.Series(
    100 + np.random.normal(0, 1, 120).cumsum(),
    name="Close",
)

high = prices + np.random.uniform(0.5, 2.0, len(prices))
low = prices - np.random.uniform(0.5, 2.0, len(prices))

volume = pd.Series(
    np.random.randint(1000, 10000, len(prices)),
    name="Volume",
)

print("=" * 60)
print("QUANT INDICATORS EXAMPLE")
print("=" * 60)

print("\nSimple Moving Average")
print(sma(prices).tail())

print("\nExponential Moving Average")
print(ema(prices).tail())

print("\nRSI")
print(rsi(prices).tail())

macd_line, signal, hist = macd(prices)

print("\nMACD")
print(macd_line.tail())

upper, middle, lower = bollinger_bands(prices)

print("\nBollinger Upper Band")
print(upper.tail())

print("\nATR")
print(atr(high, low, prices).tail())

print("\nVWAP")
print(vwap(prices, volume).tail())

print("\nDaily Returns")
print(simple_returns(prices).tail())

print("\nHistorical Volatility")
print(volatility(prices).tail())
