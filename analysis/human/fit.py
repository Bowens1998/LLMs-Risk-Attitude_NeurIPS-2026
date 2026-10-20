"""Ordered logistic regression of R_D on x = B_C / 100.

logit P(Y <= k | x) = theta_k - beta * x,  k = 1..K-1   (statsmodels OrderedModel)
AUC = integral over x in [0, 1] of E[Y | x], by the trapezoidal rule on a 201-point grid.
Only categories observed in the fitted sample enter E[Y | x].
"""
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from statsmodels.miscmodels.ordinal_model import OrderedModel

GRID = np.linspace(0.0, 1.0, 201)


@dataclass
class OLRFit:
    auc: float
    beta: float
    expected: np.ndarray  # E[Y | x] on GRID


def fit_olr(bc, rd):
    """Fit the OLR to beliefs on the 0-100 scale; None if the sample is degenerate or the fit fails."""
    x = np.asarray(bc, float) / 100.0
    y = np.asarray(rd, int)
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return None
    try:
        model = OrderedModel(y, x.reshape(-1, 1), distr='logit')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(method='bfgs', disp=False, maxiter=1000)
    except (ValueError, np.linalg.LinAlgError):
        return None
    expected = np.asarray(result.predict(exog=GRID.reshape(-1, 1))) @ np.asarray(model.labels, float)
    auc, beta = float(trapezoid(expected, GRID)), float(np.asarray(result.params)[0])
    if not (np.isfinite(auc) and np.isfinite(beta)):
        return None
    return OLRFit(auc, beta, expected)
