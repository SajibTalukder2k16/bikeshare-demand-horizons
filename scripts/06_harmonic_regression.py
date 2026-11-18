"""Step 6: dynamic harmonic regression (DHR) baseline, scored with the rolling-origin protocol.

Model (Hyndman & Athanasopoulos, FPP3, Sec. 12.1 / 10.5): y_t = x_t' b + n_t, where x_t holds
  - Fourier terms of the clock hour (daily, K_d pairs), of the hour of week (weekly, K_w pairs) and
    of the day of year (annual, 2 pairs),
  - the holiday-or-weekend indicator and its interaction with the daily Fourier terms,
  - the COVID-period indicator and the nine target-time weather variables (realized, as in the
    primary analysis of every other pipeline),
and n_t follows a (seasonal) ARIMA process. Estimation is two-stage: b by least squares on the
training years, then the ARIMA parameters by maximum likelihood on the training residuals.

Rolling origin: parameters are fixed after training (no refit in validation or test). The Kalman
filter is run over the whole residual series with those parameters; the H-step forecast for target
row T uses only the filtered state at row T-H (Z T^H a_{T-H|T-H}), i.e. demand up to T-H, plus the
regression part at T. Configurations (transform x Fourier orders x error model) are selected per
series and horizon on validation R^2, as for the feature models.

Usage:  python scripts/06_harmonic_regression.py --level system
        python scripts/06_harmonic_regression.py --level station [--stations ...]
Output: results/dhr_<level>.csv (selected config, test metrics) and results/dhr_<level>_all.csv
"""

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

import config  # noqa: E402
from src import data_io, features  # noqa: E402
from src.metrics import evaluate  # noqa: E402

TRANSFORMS = ["raw", "log1p"]
FOURIER = [(6, 3), (10, 8)]                       # (K_daily, K_weekly)
ERRORS = [((2, 0, 1), (0, 0, 0, 0)),              # ARMA(2,1)
          ((2, 0, 0), (1, 0, 0, 24)),             # AR(2) x seasonal AR(1)_24
          ((1, 1, 1), (0, 0, 0, 0)),              # ARIMA(1,1,1): level adapts to the latest data
          ((2, 0, 1), (0, 1, 1, 24))]             # daily seasonal differencing, seasonal MA(1)_24
# A weekly (168 h) seasonal error term needs ~170 Kalman states and was too slow to estimate on 52k
# hourly rows; weekly seasonality enters through the Fourier regressors instead.
ERRORS_STATION = [ERRORS[1], ERRORS[3]]


def design(df: pd.DataFrame, holidays: set, kd: int, kw: int) -> np.ndarray:
    """Regression matrix on the rows of df (needs datetime, hour, day_of_week and weather)."""
    d = features.add_calendar_features(df.copy(), holidays)
    hour, how = d["hour"].values, (d["day_of_week"] * 24 + d["hour"]).values
    doy = d["datetime"].dt.dayofyear.values
    we = d["is_holiday_or_weekend"].values.astype(float)
    cols = [np.ones(len(d)), we, d["is_covid_period"].values.astype(float)]
    for k in range(1, kd + 1):
        s, c = np.sin(2 * np.pi * k * hour / 24), np.cos(2 * np.pi * k * hour / 24)
        cols += [s, c, s * we, c * we]
    for k in range(1, kw + 1):
        cols += [np.sin(2 * np.pi * k * how / 168), np.cos(2 * np.pi * k * how / 168)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    w = d[features._WEATHER_NOW].values.astype(float)
    w = (w - w.mean(0)) / (w.std(0) + 1e-9)
    return np.column_stack(cols + [w])


def fit_forecast(y: np.ndarray, X: np.ndarray, train: np.ndarray, targets: np.ndarray,
                 horizons, transform: str, order, sorder) -> dict:
    """{H: forecasts for the target positions}; parameters estimated on the `train` positions only."""
    z = np.log1p(y) if transform == "log1p" else y.astype(float)
    beta, *_ = np.linalg.lstsq(X[train], z[train], rcond=None)
    resid = z - X @ beta
    mod = SARIMAX(resid[train], order=order, seasonal_order=sorder, trend="n")
    res = mod.fit(disp=False, maxiter=200)
    full = res.apply(resid, refit=False)                      # same parameters, whole series
    a = full.filter_results.filtered_state                    # (k_states, n): a_{t|t}
    Z = full.filter_results.design[:, :, 0]                   # (1, k)
    T = full.filter_results.transition[:, :, 0]               # (k, k)
    out, Tp = {}, np.eye(T.shape[0])
    for h in range(1, max(horizons) + 1):
        Tp = T @ Tp
        if h in horizons:
            eta = (Z @ Tp @ a[:, targets - h]).ravel()        # residual forecast from origin T-h
            pred = X[targets] @ beta + eta
            out[h] = np.expm1(pred) if transform == "log1p" else pred
    return out


def evaluate_series(df: pd.DataFrame, holidays: set, keys: dict, errors) -> pd.DataFrame:
    s = data_io.continuous_hourly(df)                         # elapsed-hour index, as for foundation models
    cov = df.set_index("datetime").reindex(s.index).ffill().bfill().reset_index().rename(
        columns={"index": "datetime"})
    cov["datetime"] = s.index
    year = s.index.year
    train = np.where((year >= config.TRAIN_YEARS[0]) & (year <= config.TRAIN_YEARS[1]))[0]
    val = np.where(year == config.VAL_YEAR)[0]
    test = np.where(year == config.TEST_YEAR)[0]
    # the same scored test rows as the feature models (rows present in the processed data)
    present = set(pd.to_datetime(df["datetime"]))
    test = np.array([p for p in test if s.index[p] in present])
    val = np.array([p for p in val if s.index[p] in present])
    rows = []
    for target in config.TARGETS:
        y = s[target].values
        for tr, (kd, kw), (order, sorder) in itertools.product(TRANSFORMS, FOURIER, errors):
            X = design(cov, holidays, kd, kw)
            t0 = time.time()
            try:
                pv = fit_forecast(y, X, train, val, config.HORIZONS, tr, order, sorder)
                pt = fit_forecast(y, X, train, test, config.HORIZONS, tr, order, sorder)
            except Exception as e:  # noqa: BLE001  (a failed fit is recorded, never silently dropped)
                print(f"    fit failed {tr} {kd},{kw} {order}{sorder}: {e}")
                continue
            cfg = f"{tr}|Kd{kd}Kw{kw}|{order}x{sorder[:3]}_{sorder[3]}"
            for h in config.HORIZONS:
                rows.append({**keys, "target": target, "horizon": h, "config": cfg,
                             "val_R2": r2_score(y[val], np.maximum(pv[h], 0)),
                             **evaluate(y[test], pt[h])})
            print(f"    {target} {cfg}: val R2 H8={rows[-3]['val_R2']:.3f} test={rows[-3]['R2']:.3f} "
                  f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def select(all_rows: pd.DataFrame, keys) -> pd.DataFrame:
    idx = all_rows.groupby(keys + ["target", "horizon"])["val_R2"].idxmax()
    return all_rows.loc[idx].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["system", "station"], required=True)
    p.add_argument("--stations", nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    holidays = data_io.load_holidays()
    out_all = args.out_dir / f"dhr_{args.level}_all.csv"
    if args.level == "system":
        allr = evaluate_series(data_io.load_system_data(), holidays, {}, ERRORS)
        keys = []
    else:
        keys = ["station_id"]
        grid = data_io.load_station_grid(horizon=1)
        done = pd.read_csv(out_all, dtype={"station_id": str}) if out_all.exists() else pd.DataFrame()
        finished = set(done["station_id"]) if len(done) else set()
        parts = [done] if len(done) else []
        for i, sid in enumerate(args.stations or data_io.list_stations(), 1):
            ok, _ = data_io.station_is_active(sid)
            if not ok or sid in finished:
                continue
            t0 = time.time()
            raw = data_io.load_station_series(sid, grid)
            parts.append(evaluate_series(raw, holidays, {"station_id": sid}, ERRORS_STATION))
            pd.concat(parts).to_csv(out_all, index=False)  # checkpoint
            print(f"  [{i}] {sid} done ({time.time() - t0:.0f}s)")
        allr = pd.concat(parts)
    allr.to_csv(out_all, index=False)
    sel = select(allr, keys)
    sel.to_csv(args.out_dir / f"dhr_{args.level}.csv", index=False)
    if args.level == "system":
        print(sel[["target", "horizon", "config", "val_R2", "R2", "MAE"]].to_string(index=False))
    print(f"-> {args.out_dir / f'dhr_{args.level}.csv'}")


if __name__ == "__main__":
    main()
