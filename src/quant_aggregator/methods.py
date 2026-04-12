"""Price aggregation methods for fragmented crypto markets."""

from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    clean_values = values[mask]
    clean_weights = weights[mask]

    if clean_values.size == 0:
        return float("nan")
    if clean_values.size == 1:
        return float(clean_values[0])

    order = np.argsort(clean_values)
    sorted_values = clean_values[order]
    sorted_weights = clean_weights[order]

    cum_weights = np.cumsum(sorted_weights)
    threshold = 0.5 * cum_weights[-1]
    idx = np.searchsorted(cum_weights, threshold, side="left")
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def add_vwap(
    trades: pd.DataFrame,
    window: str = "30s",
    price_col: str = "price",
    volume_col: str = "amount",
) -> pd.DataFrame:
    numerator = (trades[price_col] * trades[volume_col]).rolling(window).sum()
    denominator = trades[volume_col].rolling(window).sum()
    trades["VWAP"] = numerator / denominator
    return trades


def add_vwm(
    trades: pd.DataFrame,
    window: str = "30s",
    step: str = "1s",
    price_col: str = "price",
    volume_col: str = "amount",
) -> pd.DataFrame:
    grouped = (
        trades.groupby(trades.index.floor(step))[[price_col, volume_col]]
        .agg(list)
        .rename(columns={price_col: "prices", volume_col: "volumes"})
    )

    full_grid = pd.date_range(grouped.index.min(), grouped.index.max(), freq=step)
    grouped = grouped.reindex(full_grid, fill_value=[])

    window_size = int(np.ceil(pd.to_timedelta(window) / pd.to_timedelta(step)))
    vwm_values = []

    for i in range(len(grouped)):
        rolling_slice = grouped.iloc[max(0, i - window_size + 1) : i + 1]

        prices_concat = (
            np.concatenate(rolling_slice["prices"].to_numpy())
            if len(rolling_slice)
            else np.array([])
        )
        volumes_concat = (
            np.concatenate(rolling_slice["volumes"].to_numpy())
            if len(rolling_slice)
            else np.array([])
        )

        vwm_values.append(_weighted_median(prices_concat, volumes_concat))

    vwm_series = pd.Series(vwm_values, index=full_grid, name="VWM")
    trades["VWM"] = trades.index.floor(step).map(vwm_series)
    return trades


def _weighted_quantile_center(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    clean_values = values[mask]
    clean_weights = weights[mask]

    if len(clean_values) == 0:
        return float("nan")

    order = np.argsort(clean_values)
    sorted_values = clean_values[order]
    sorted_weights = clean_weights[order]

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    if total_weight <= 0:
        return float("nan")

    cumulative_share = cum_weights / total_weight
    idx = np.searchsorted(cumulative_share, 0.5, side="right")

    if idx == 0:
        return float(sorted_values[0])

    left_share, right_share = cumulative_share[idx - 1], cumulative_share[idx]
    left_value, right_value = sorted_values[idx - 1], sorted_values[idx]

    if right_share <= left_share:
        return float(right_value)

    return float(
        left_value + (0.5 - left_share) * (right_value - left_value) / (right_share - left_share)
    )


def _robust_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    upper = _weighted_quantile_center(values, weights)
    lower = -_weighted_quantile_center(-values, weights)

    if np.isnan(upper) and np.isnan(lower):
        return float("nan")
    if np.isnan(upper):
        return lower
    if np.isnan(lower):
        return upper
    return 0.5 * (upper + lower)


def add_rwm(
    trades: pd.DataFrame,
    window: str = "30s",
    step: str = "1s",
    price_col: str = "price",
    volume_col: str = "amount",
) -> pd.DataFrame:
    trades[price_col] = pd.to_numeric(trades[price_col], errors="coerce")
    trades[volume_col] = pd.to_numeric(trades[volume_col], errors="coerce")

    mask = np.isfinite(trades[price_col]) & np.isfinite(trades[volume_col]) & (trades[volume_col] > 0)

    if not mask.any():
        trades["RWM"] = np.nan
        return trades

    valid_prices = trades.loc[mask, price_col].to_numpy(dtype=float)
    valid_volumes = trades.loc[mask, volume_col].to_numpy(dtype=float)

    median_volume = float(np.nanmedian(valid_volumes))
    if not np.isfinite(median_volume) or median_volume <= 0:
        median_volume = 1.0

    robust_weights = np.log1p(valid_volumes / median_volume)

    features = pd.DataFrame(
        {"prices": valid_prices, "weights": robust_weights},
        index=trades.index[mask],
    )

    grouped = features.groupby(pd.Grouper(freq=step))
    grouped_prices = grouped["prices"].apply(lambda s: s.to_numpy(dtype=float))
    grouped_weights = grouped["weights"].apply(lambda s: s.to_numpy(dtype=float))

    full_grid = pd.date_range(grouped_prices.index.min(), grouped_prices.index.max(), freq=step)
    grouped_prices = grouped_prices.reindex(full_grid, fill_value=np.array([], dtype=float))
    grouped_weights = grouped_weights.reindex(full_grid, fill_value=np.array([], dtype=float))

    window_size = int(np.ceil(pd.to_timedelta(window) / pd.to_timedelta(step)))
    rwm_values = np.full(len(full_grid), np.nan, dtype=float)

    for i in range(len(full_grid)):
        start = max(0, i - window_size + 1)
        concatenated_prices = np.concatenate(grouped_prices.iloc[start : i + 1].values)
        concatenated_weights = np.concatenate(grouped_weights.iloc[start : i + 1].values)
        rwm_values[i] = _robust_weighted_mean(concatenated_prices, concatenated_weights)

    rwm_series = pd.Series(rwm_values, index=full_grid, name="RWM")
    trades["RWM"] = trades.index.floor(step).map(rwm_series)
    return trades


def _compute_resampled_vwap_by_exchange(
    trades: pd.DataFrame,
    exchange_col: str,
    price_col: str,
    volume_col: str,
    frequency: str,
    order_window: str,
) -> pd.DataFrame:
    temp = trades[[exchange_col, price_col, volume_col]].copy()
    temp["_pxv"] = temp[price_col] * temp[volume_col]

    pxv_per_step = (
        temp.groupby(exchange_col)["_pxv"].resample(frequency).sum().unstack(exchange_col).fillna(0.0)
    )
    volume_per_step = (
        temp.groupby(exchange_col)[volume_col].resample(frequency).sum().unstack(exchange_col).fillna(0.0)
    )

    rolling_pxv = pxv_per_step.rolling(order_window, min_periods=1).sum()
    rolling_volume = volume_per_step.rolling(order_window, min_periods=1).sum()

    vwap_prices = rolling_pxv / rolling_volume.replace(0.0, np.nan)
    return vwap_prices.ffill()


def _compute_volume_by_step(
    trades: pd.DataFrame,
    exchange_col: str,
    volume_col: str,
    frequency: str,
) -> pd.DataFrame:
    return (
        trades.groupby(exchange_col)[volume_col].resample(frequency).sum().unstack(exchange_col).fillna(0.0)
    )


def _compute_recent_orders(volume_by_step: pd.DataFrame, order_window: str) -> pd.DataFrame:
    return volume_by_step.rolling(order_window, min_periods=1).sum()


def _volume_weighted_kurtosis(returns: np.ndarray, volumes: np.ndarray, min_periods: int = 30) -> float:
    mask = np.isfinite(returns) & np.isfinite(volumes) & (volumes > 0)
    valid_returns = returns[mask]
    valid_volumes = volumes[mask]

    if len(valid_returns) < min_periods:
        return float("nan")

    total_weight = valid_volumes.sum()
    if total_weight <= 0:
        return float("nan")

    weights = valid_volumes / total_weight
    mean_return = np.sum(weights * valid_returns)
    centered = valid_returns - mean_return

    second_moment = np.sum(weights * centered**2)
    if not np.isfinite(second_moment) or second_moment <= EPSILON:
        return float("nan")

    fourth_moment = np.sum(weights * centered**4)
    return float(fourth_moment / (second_moment**2 + EPSILON))


def _compute_weighted_kurtosis(
    returns: pd.DataFrame,
    volume_by_step: pd.DataFrame,
    moment_window: str,
    min_periods: int,
) -> pd.DataFrame:
    kurtosis_df = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    delta = pd.Timedelta(moment_window)

    for exchange in returns.columns:
        exchange_returns = returns[exchange]
        exchange_volumes = volume_by_step[exchange]

        for timestamp in returns.index:
            start_time = timestamp - delta + pd.Timedelta("1ns")
            returns_window = exchange_returns.loc[start_time:timestamp].to_numpy(dtype=float)
            volumes_window = exchange_volumes.loc[start_time:timestamp].to_numpy(dtype=float)
            kurtosis_df.at[timestamp, exchange] = _volume_weighted_kurtosis(
                returns_window,
                volumes_window,
                min_periods=min_periods,
            )

    return kurtosis_df


def _compute_anomaly_score(kurtosis: pd.DataFrame, lower_bound: float = EPSILON) -> pd.DataFrame:
    score = kurtosis**2
    score = score.replace([np.inf, -np.inf], np.nan)
    return score.clip(lower=lower_bound)


def _compute_sgrd_weights(
    recent_orders: pd.DataFrame,
    anomaly_score: pd.DataFrame,
    lower_bound: float = EPSILON,
) -> pd.DataFrame:
    safe_score = anomaly_score.clip(lower=lower_bound)
    weights = np.log1p(recent_orders / safe_score)
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_aggregated_price(
    resampled_prices: pd.DataFrame,
    total_weights: pd.DataFrame,
) -> pd.Series:
    aggregated = []

    for timestamp in resampled_prices.index:
        values = resampled_prices.loc[timestamp].to_numpy(dtype=float)
        weights = total_weights.loc[timestamp].to_numpy(dtype=float)
        aggregated.append(_weighted_median(values, weights))

    return pd.Series(aggregated, index=resampled_prices.index, name="aggregated_price")


def add_sgrd(
    trades: pd.DataFrame,
    frequency: str = "1s",
    moment_window: str = "3min",
    order_window: str = "30s",
    min_periods: int = 30,
    price_col: str = "price",
    volume_col: str = "amount",
    exchange_col: str = "exchange",
) -> pd.DataFrame:
    """
    SGRD aggregation.

    Note for portfolio context:
    Le SGRD est pratique pour agreger le prix lors de journees comme
    `btc-eur-2023-03-05`, ou il y a une forte dispersion du cours entre
    un gros exchange et des plus petits exchanges.

    SGRD is practical to aggregate prices during stressed sessions such as
    `btc-eur-2023-03-05`, where price dispersion can appear between a large
    exchange and smaller exchanges.
    """

    trades[price_col] = pd.to_numeric(trades[price_col], errors="coerce")
    trades[volume_col] = pd.to_numeric(trades[volume_col], errors="coerce")

    resampled_prices = _compute_resampled_vwap_by_exchange(
        trades,
        exchange_col,
        price_col,
        volume_col,
        frequency,
        order_window,
    )

    volume_by_step = _compute_volume_by_step(trades, exchange_col, volume_col, frequency)
    returns = np.log(resampled_prices).diff()

    kurtosis = _compute_weighted_kurtosis(
        returns,
        volume_by_step,
        moment_window=moment_window,
        min_periods=min_periods,
    )

    anomaly_score = _compute_anomaly_score(kurtosis)
    recent_orders = _compute_recent_orders(volume_by_step, order_window=order_window)
    total_weights = _compute_sgrd_weights(recent_orders, anomaly_score)

    aggregated_price = _compute_aggregated_price(resampled_prices, total_weights)
    trades["SGRD"] = trades.index.floor(frequency).map(aggregated_price)
    return trades


def add_all_aggregators(trades: pd.DataFrame) -> pd.DataFrame:
    add_vwap(trades)
    add_vwm(trades)
    add_rwm(trades)
    add_sgrd(trades)
    return trades
