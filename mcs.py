"""
Monsoon-Cotton Signal
---------------------
Tests whether rainfall anomalies in the Vidarbha cotton belt (Maharashtra, India)
lead price movements in ICE Cotton futures (CT=F), using free NASA POWER weather
data and yfinance price data.

This is an exploratory signal-discovery script, NOT a validated trading strategy.
No backtest, no transaction costs, no out-of-sample split is applied here —
see the README for suggested next steps to make this more rigorous.

Usage:
    python monsoon_cotton_signal.py
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config — change these to explore a different region / crop pair
# ----------------------------------------------------------------------
LAT, LON = 20.7, 77.0          # Akola / Yavatmal, Vidarbha cotton belt, Maharashtra
START_DATE = "20180101"        # NASA POWER format: YYYYMMDD
END_DATE = "20260630"
TICKER = "CT=F"                 # ICE Cotton No. 2 futures
MAX_LAG_DAYS = 90               # how far out to test lead-lag correlation
GROWING_SEASON_MONTHS = [6, 7, 8, 9, 10]  # Jun-Oct: sowing to harvest window


# ----------------------------------------------------------------------
# Step 1: Pull weather data from NASA POWER (free, no API key required)
# ----------------------------------------------------------------------
def get_nasa_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch daily temperature (C) and corrected precipitation (mm/day)."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    df = pd.DataFrame({
        "temp": data["T2M"],
        "rainfall": data["PRECTOTCORR"],
    })
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df


# ----------------------------------------------------------------------
# Step 2: Pull commodity futures price data via yfinance
# ----------------------------------------------------------------------
def get_price_data(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.Ticker(ticker).history(
        start=pd.to_datetime(start, format="%Y%m%d"),
        end=pd.to_datetime(end, format="%Y%m%d"),
    )
    price = df["Close"]
    price.index = price.index.tz_localize(None)
    price.name = "price"
    return price


# ----------------------------------------------------------------------
# Step 3: Turn raw rainfall into an anomaly / stress signal
# ----------------------------------------------------------------------
def compute_rain_anomaly(weather: pd.DataFrame) -> pd.Series:
    """
    30-day rolling rainfall mean, compared against the 365-day rolling mean/std,
    expressed as a z-score. Positive = wetter than normal, negative = drier
    than normal (drought stress) for that time of year.
    """
    short_ma = weather["rainfall"].rolling(30, min_periods=15).mean()
    long_ma = weather["rainfall"].rolling(365, min_periods=180).mean()
    long_std = weather["rainfall"].rolling(365, min_periods=180).std()
    z = (short_ma - long_ma) / long_std
    z.name = "rain_zscore"
    return z


# ----------------------------------------------------------------------
# Step 4: Lagged cross-correlation — the actual research question
# ----------------------------------------------------------------------
def lagged_correlation(signal: pd.Series, target: pd.Series, max_lag: int) -> pd.Series:
    """
    For each lag L, shift the weather signal forward by L days and correlate
    it with price returns. A peak at lag L suggests the weather anomaly
    today is associated with a price move ~L days later.
    """
    results = {}
    for lag in range(0, max_lag + 1):
        shifted = signal.shift(lag)
        aligned = pd.concat([shifted, target], axis=1).dropna()
        if len(aligned) < 30:
            results[lag] = np.nan
            continue
        results[lag] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return pd.Series(results, name="correlation")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    print("Fetching NASA POWER weather data...")
    weather = get_nasa_weather(LAT, LON, START_DATE, END_DATE)

    print("Fetching cotton futures price data...")
    price = get_price_data(TICKER, START_DATE, END_DATE)

    print("Computing rainfall anomaly signal...")
    rain_z = compute_rain_anomaly(weather)

    merged = pd.concat([rain_z, price], axis=1).dropna(subset=["price"])
    merged["price_return"] = merged["price"].pct_change()

    # Restrict weather stress signal to the growing season only —
    # rainfall in the off-season is agronomically irrelevant to this crop.
    in_season = merged.index.month.isin(GROWING_SEASON_MONTHS)
    seasonal_signal = merged["rain_zscore"].where(in_season)

    print(f"Running lagged correlation analysis (0 to {MAX_LAG_DAYS} days)...")
    corr_by_lag = lagged_correlation(seasonal_signal, merged["price_return"], MAX_LAG_DAYS)

    best_lag = corr_by_lag.abs().idxmax()
    best_corr = corr_by_lag.loc[best_lag]
    print(f"\nStrongest correlation: {best_corr:.3f} at a {best_lag}-day lag")

    # --- Plot 1: rainfall anomaly vs price, over time ---
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(merged.index, merged["rain_zscore"], color="tab:blue", alpha=0.6, label="Rainfall anomaly (z-score)")
    ax1.set_ylabel("Rainfall anomaly (z-score)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged["price"], color="tab:orange", label="Cotton price (CT=F)")
    ax2.set_ylabel("Cotton futures price (USD/lb)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Vidarbha rainfall anomaly vs ICE Cotton futures price")
    fig.tight_layout()
    plt.savefig("rainfall_vs_price.png", dpi=150)
    print("Saved chart: rainfall_vs_price.png")

    # --- Plot 2: correlation vs lag ---
    plt.figure(figsize=(10, 5))
    plt.plot(corr_by_lag.index, corr_by_lag.values, color="tab:green")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(best_lag, color="red", linestyle="--", alpha=0.7, label=f"Peak lag = {best_lag}d")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation (rainfall anomaly vs price return)")
    plt.title("Lead-lag correlation: rainfall anomaly -> cotton price return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lag_correlation.png", dpi=150)
    print("Saved chart: lag_correlation.png")

    # Save merged dataset for further analysis / reuse
    merged.to_csv("merged_weather_price_data.csv")
    print("Saved dataset: merged_weather_price_data.csv")


if __name__ == "__main__":
    main()
