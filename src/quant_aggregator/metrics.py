"""Evaluation metrics for aggregation quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


AGGREGATORS = ("VWAP", "VWM", "RWM", "SGRD")


def rmse_by_aggregator(
    trades: pd.DataFrame,
    target_col: str = "efficient_price",
    aggregators: tuple[str, ...] = AGGREGATORS,
) -> dict[str, float]:
    return {
        name: float(np.sqrt(np.mean((trades[name] - trades[target_col]) ** 2)))
        for name in aggregators
        if name in trades.columns and target_col in trades.columns
    }
