import pandas as pd
import numpy as np


def moving_average(df: pd.DataFrame, column: str = "close", window: int = 20) -> pd.Series:
    """Calculate moving average for a specific column."""
    return df[column].rolling(window=window).mean()


def rsi(df: pd.DataFrame, column: str = "close", window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(
    df: pd.DataFrame, column: str = "close", window: int = 20, num_std: int = 2
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    ma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"ma": ma, "upper": upper, "lower": lower})


__all__ = ["moving_average", "rsi", "bollinger_bands"]
