# Repository: qca-monte-carlo-replication-of-factor-recovery-and-for

## Overview
This repository implements two experiments from Stock & Watson (2002) on large approximate factor models:

1. **Experiment 1** (`src/monte_carlo.py`): Monte Carlo replication of Table 1 — factor recovery and forecast efficiency
2. **Experiment 2** (`src/empirical.py`): Empirical expanding-window PC forecasting for U.S. industrial production

## Repository Structure
```
src/
  __init__.py
  dgp.py              # Data Generating Process for Monte Carlo
  pca_estimation.py   # PC factor estimation (SVD-based)
  factor_selection.py # Bai-Ng ICp1/2/3, AIC/BIC selection
  statistics.py       # R²_{F̂,F} and S²_{ŷ,ỹ} statistics
  scenarios.py        # Table 1 scenario definitions (families A-F)
  monte_carlo.py      # Monte Carlo orchestration (2000 reps/scenario)
  empirical.py        # Expanding-window empirical forecasting
  visualization.py    # Matplotlib/seaborn plots

tests/
  test_dgp.py
  test_pca.py
  test_factor_selection.py
  test_statistics.py
  test_monte_carlo.py
  test_empirical.py

results/
  RESULTS.md          # Key metrics and findings
  table1_monte_carlo.csv
  table2_empirical_forecasts.csv
  table2_empirical_metrics.csv
  fig*.png            # Visualizations

main.py               # Entry point: runs both experiments
```

## Key Implementation Decisions

### lambda*(R_i²) Calibration
Formula: `sqrt(R_i² / (r_true * (1 - R_i²)))`
Derived so that expected R_i² matches target in baseline scenario (alpha=0, iid, homoskedastic).

### S² Statistic (Eq. 18)
Used version: `S² = 1 - E[(ŷ - ỹ)²] / E[ỹ²]` (denominator = E[ỹ²]).

### Spatial Boundary Conditions (Eq. 13)
Zero-padding: v_{0,t} = v_{N+1,t} = 0.

### GARCH Initialization
sigma²_{i,0} = delta_0/(1-delta_1-delta_2) (unconditional variance).

### PCA Normalization
Lambda_hat' Lambda_hat / N = I_k, F_hat = X Lambda_hat / N.
Implemented via SVD/eigendecomposition of X'X or XX'.

### Forecasting Regression
Always includes intercept. Regression sample: t=1,...,T-1 (y_{t+1} on F_t).

## Running the Experiments
```bash
# Full run (2000 replications)
python main.py

# Quick test (100 replications)
python main.py --quick

# Run tests
pytest tests/
```

## Data Sources
- **Experiment 1**: Synthetic data only (no external data needed)
- **Experiment 2**: FRED data via massive API (MASSIVE_TOKEN env var required)
  - Falls back to synthetic data if API unavailable

## Dependencies
See requirements.txt. Key: numpy, pandas, scipy, matplotlib, seaborn, massive.
