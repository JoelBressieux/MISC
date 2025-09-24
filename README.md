# Backtesting Playground

This repository hosts a heavily-commented backtesting playground built around a
simple 12-month momentum strategy on the S&P 500 universe. It is intentionally
modular so that new strategies and optimisation tricks can join the party
without knocking over the furniture.

## Quick start

```bash
pip install -r requirements.txt
python -m backtesting.main --years 10
```

Running the script prints a couple of handy tables comparing the strategy to the
S&P 500, and also saves a cumulative performance chart inside the `output/`
folder. If the chart looks great feel free to frame it. If it looks terrible,
maybe pretend it was “out-of-sample stress testing”.

## Structure

- `backtesting/data.py` – data access helpers and the S&P 500 universe loader.
- `backtesting/strategies.py` – strategy definitions, starting with 12-month momentum.
- `backtesting/engine.py` – the glue that turns signals into returns.
- `backtesting/metrics.py` – performance statistics like Sharpe, volatility, etc.
- `backtesting/reporting.py` – table builders for human-friendly output.
- `backtesting/plotting.py` – cumulative performance plotting.
- `backtesting/main.py` – CLI entry point tying everything together.

## Next steps

The framework is ready for additional strategies and optimisation techniques.
Drop new classes into `strategies.py`, reuse the engine, and let the reporting
module do the rest. Snacks optional but highly recommended.
