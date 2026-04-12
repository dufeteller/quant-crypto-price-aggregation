# Quant Crypto Price Aggregation

Quant-focused portfolio project: comparison of multiple price aggregation methods in a fragmented crypto market (multi-exchange setting).

## Why this project

- Market microstructure modeling through synthetic multi-exchange simulations.
- Design, implementation, and comparison of robust price aggregators, including the proposal of a novel method (SGRD).
- Quantitative evaluation using RMSE against a simulated efficient price benchmark.
- Structured, reproducible codebase designed for research-grade experimentation.

## Repository structure

- `src/quant_aggregator/methods.py`: aggregation methods (VWAP, VWM, RWM, SGRD)
- `src/quant_aggregator/simulation.py`: synthetic market generator
- `src/quant_aggregator/metrics.py`: evaluation metrics
- `src/quant_aggregator/io_utils.py`: dataset preparation helpers
- `src/quant_aggregator/plotting.py`: visualization helpers
- `notebook.ipynb`: end-to-end demo notebook
- `requirements.txt`: dependencies

## Data confidentiality

Real market datasets are not versioned in this repository.

Methodological note:

SGRD is useful for aggregating prices during sessions on the BTC-EUR pair, such as March 5, 2023, where strong price dispersion can appear between a large exchange and smaller exchanges.

## How SGRD works

SGRD is a robust cross-exchange aggregation method.

1. For each exchange, prices are first resampled with a short rolling VWAP.
2. Log-returns are computed and a volume-weighted rolling kurtosis is estimated.
3. An anomaly score is built from kurtosis: `score = kurtosis^2`.
4. Exchange weights are computed with:
   `weight = log(1 + recent_volume / score)`.
5. The final market price is the weighted median across exchanges.

Intuition:
- High kurtosis indicates unstable/heavy-tail behavior, so that venue gets penalized.
- Higher recent volume increases weight, but in a controlled logarithmic way.
- Weighted median improves robustness to outliers and temporary exchange dislocations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Minimal usage

```python
from quant_aggregator.simulation import simulate_fragmented_market
from quant_aggregator.methods import add_all_aggregators
from quant_aggregator.metrics import rmse_by_aggregator

trades = simulate_fragmented_market(seed=42)
trades["timestamp"] = trades["timestamp"].astype("datetime64[ns]")
trades = trades.sort_values("timestamp").set_index("timestamp")

add_all_aggregators(trades)
print(rmse_by_aggregator(trades, target_col="efficient_price"))
```

## Author

Romain DUFETELLE<br>
LinkedIn: https://www.linkedin.com/in/romain-dufetelle-262828319/<br>
GitHub: https://github.com/dufeteller
