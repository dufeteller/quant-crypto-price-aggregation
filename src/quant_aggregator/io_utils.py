"""Input helpers for trade datasets."""

from __future__ import annotations

import pandas as pd


def prepare_trade_data(
    trades: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timestamp_unit: str = "ms",
) -> pd.DataFrame:
    cleaned = trades.copy()
    cleaned["dt"] = pd.to_datetime(cleaned[timestamp_col], unit=timestamp_unit)
    cleaned = cleaned.sort_values("dt").set_index("dt")

    columns_to_drop = [
        "trade_id",
        "taker_side_sell",
        "timestamp",
        "pair",
    ]
    existing = [col for col in columns_to_drop if col in cleaned.columns]
    if existing:
        cleaned = cleaned.drop(columns=existing)

    return cleaned
