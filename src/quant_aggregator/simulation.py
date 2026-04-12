"""Synthetic fragmented market simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import burr12, norm, t


def simulate_gbm(
    n_steps: int,
    dt: float,
    initial_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    shocks = rng.standard_normal(n_steps)

    log_price = np.empty(n_steps)
    log_price[0] = np.log(initial_price)

    drift_term = (drift - 0.5 * volatility**2) * dt
    vol_term = volatility * np.sqrt(dt)

    for step in range(1, n_steps):
        log_price[step] = log_price[step - 1] + drift_term + vol_term * shocks[step]

    return np.exp(log_price)


def sample_clayton_copula(
    n: int,
    dim: int,
    theta: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng

    latent = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    exponentials = rng.exponential(scale=1.0, size=(n, dim))
    return (1.0 + exponentials / latent[:, None]) ** (-1.0 / theta)


def mixed_innovations_from_uniform(
    uniforms: np.ndarray,
    outlier_prob: float = 0.1,
    normal_sigma: float = 0.01,
    outlier_df: int = 3,
    outlier_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    n, dim = uniforms.shape

    gaussian_core = normal_sigma * norm.ppf(np.clip(uniforms, 1e-12, 1.0 - 1e-12))
    outlier_component = outlier_scale * t.rvs(df=outlier_df, size=(n, dim), random_state=rng)

    outlier_mask = rng.random((n, dim)) < outlier_prob
    return np.where(outlier_mask, outlier_component, gaussian_core)


def apply_ar1(innovations: np.ndarray, ar_coeff: float = 0.5) -> np.ndarray:
    n, dim = innovations.shape
    returns = np.empty_like(innovations)
    returns[0, :] = innovations[0, :]

    for step in range(1, n):
        returns[step, :] = ar_coeff * returns[step - 1, :] + (1.0 - ar_coeff) * innovations[step, :]

    return returns


def simulate_volumes(
    n: int,
    dim: int,
    heavy_tails: bool = False,
    burr_params: tuple[float, float] = (2.0, 5.0),
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng

    if not heavy_tails:
        return np.abs(rng.standard_normal((n, dim))) * scale

    c_param, d_param = burr_params
    return burr12.rvs(c=c_param, d=d_param, size=(n, dim), random_state=rng) * scale


def simulate_fragmented_market(
    n_steps: int = 1440,
    dt: float = 1 / 1440,
    exchanges: tuple[str, ...] = ("emolas", "niamor", "onurb", "omit", "xela"),
    initial_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.5,
    copula_theta: float = 2.0,
    outlier_prob: float = 0.1,
    normal_sigma: float = 0.01,
    outlier_df: int = 3,
    outlier_scale: float = 0.05,
    ar_coeff: float = 0.5,
    heavy_tail_volumes: bool = False,
    volume_scale: float = 10.0,
    burr_params: tuple[float, float] = (2.0, 5.0),
    seed: int = 0,
    start_ts: str = "2003-11-23 00:00:00",
    freq: str = "1s",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_exchanges = len(exchanges)

    efficient_price = simulate_gbm(
        n_steps=n_steps,
        dt=dt,
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        rng=rng,
    )

    copula_uniforms = sample_clayton_copula(n=n_steps, dim=n_exchanges, theta=copula_theta, rng=rng)
    innovations = mixed_innovations_from_uniform(
        copula_uniforms,
        outlier_prob=outlier_prob,
        normal_sigma=normal_sigma,
        outlier_df=outlier_df,
        outlier_scale=outlier_scale,
        rng=rng,
    )

    exchange_returns = apply_ar1(innovations, ar_coeff=ar_coeff)
    observed_prices = efficient_price[:, None] * (1.0 + exchange_returns)
    volumes = simulate_volumes(
        n_steps,
        n_exchanges,
        heavy_tails=heavy_tail_volumes,
        burr_params=burr_params,
        scale=volume_scale,
        rng=rng,
    )

    timestamps = pd.date_range(start=start_ts, periods=n_steps, freq=freq)

    frames = []
    for exchange_idx, exchange in enumerate(exchanges):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "exchange": exchange,
                    "efficient_price": efficient_price,
                    "price": observed_prices[:, exchange_idx],
                    "return": exchange_returns[:, exchange_idx],
                    "amount": volumes[:, exchange_idx],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)
