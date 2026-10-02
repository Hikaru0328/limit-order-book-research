"""Direct-horizon OLS/VAR prediction; coefficients are not Mori kernels."""
import numpy as np


def design(y, lags, horizon):
    y = np.asarray(y, float)
    if y.ndim != 2 or lags < 0 or horizon < 1 or len(y) <= lags+horizon:
        raise ValueError('invalid shape, history, horizon, or sample count')
    t = np.arange(lags, len(y)-horizon)
    x = np.concatenate([np.ones((len(t), 1))] + [y[t-k] for k in range(lags+1)], axis=1)
    return x, y[t+horizon]


def compare_var(train, test, lags, horizon=1):
    """Train only on train; independent test; identical target rows for both fits.

    Even the current-only fit uses the same training rows as the history fit.
    R2 uses test target variance, with no test information entering fitting.
    """
    xt, yt = design(train, lags, horizon)
    xv, yv = design(test, lags, horizon)
    scores = []
    for cols in (train.shape[1]+1, xt.shape[1]):
        coef = np.linalg.lstsq(xt[:, :cols], yt, rcond=None)[0]
        error = yv - xv[:, :cols] @ coef
        denom = np.sum((yv-yv.mean(axis=0))**2)
        scores.append(float(1-np.sum(error**2)/denom) if denom > 0 else float('nan'))
    return dict(r2_current=scores[0], r2_history=scores[1], delta_r2=scores[1]-scores[0],
                train_targets=len(yt), test_targets=len(yv))
