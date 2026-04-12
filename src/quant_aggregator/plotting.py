"""Plotting helpers for exploratory analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prices_by_exchange(
    trades: pd.DataFrame,
    add_vwap: bool = False,
    add_vwm: bool = False,
    add_rwm: bool = False,
    add_sgrd: bool = False,
    add_efficient: bool = False,
) -> None:
    plt.figure(figsize=(10, 5))

    for exchange in trades["exchange"].unique():
        subset = trades[trades["exchange"] == exchange]
        plt.plot(subset.index, subset["price"], linewidth=1, label=exchange)

    if add_efficient and "efficient_price" in trades.columns:
        plt.plot(trades.index, trades["efficient_price"], linewidth=2, label="efficient_price", color="red")

    if add_vwap and "VWAP" in trades.columns:
        plt.plot(trades.index, trades["VWAP"], linewidth=2, label="VWAP", color="black")

    if add_vwm and "VWM" in trades.columns:
        plt.plot(trades.index, trades["VWM"], linewidth=2, label="VWM", color="black")

    if add_rwm and "RWM" in trades.columns:
        plt.plot(trades.index, trades["RWM"], linewidth=2, label="RWM", color="black")

    if add_sgrd and "SGRD" in trades.columns:
        plt.plot(trades.index, trades["SGRD"], linewidth=2, label="SGRD", color="black")

    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("Trades by Exchange")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_kurtosis_score(trades: pd.DataFrame, window: str = "3min", frequency: str = "1s") -> None:
    data = trades.copy().sort_index()

    data["return"] = data.groupby("exchange")["price"].transform(lambda x: np.log(x).diff())

    kurtosis_df = (
        data.groupby("exchange")["return"]
        .rolling(window)
        .kurt()
        .rename("kurtosis")
        .reset_index()
    )

    kurtosis_df["score"] = kurtosis_df["kurtosis"] ** 2

    pivot = kurtosis_df.pivot_table(index="dt", columns="exchange", values="score", aggfunc="last")
    pivot = pivot.resample(frequency).last().ffill()

    ax = pivot.plot(figsize=(12, 6))
    ax.set_title(f"Score = kurtosis² (rolling {window})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly score")
    ax.grid(True)
    plt.show()
