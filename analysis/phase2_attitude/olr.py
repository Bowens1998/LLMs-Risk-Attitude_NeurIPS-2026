"""Ordered logistic regression of R_D on B_C, expected-decision curves and AUC.

logit P(Y <= k | x) = theta_k - beta * x,  x = B_C / 100,  k = 1..K-1
E[Y | x] = sum_k k * P(Y = k | x) over the categories observed in the cell
AUC = integral of E[Y | x] over [0, 1] (trapezoid rule, 1,001-point grid)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import trapezoid
from statsmodels.miscmodels.ordinal_model import OrderedModel

GRID = np.linspace(0.0, 1.0, 1001)


@dataclass
class OLRFit:
    model: OrderedModel
    result: object
    bc: np.ndarray  # 0-100 scale

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.model.labels, dtype=float)

    @property
    def beta(self) -> float:
        return float(self.result.params[0])

    @property
    def beta_se(self) -> float:
        return float(self.result.bse[0])

    @property
    def beta_p(self) -> float:
        """Two-sided Wald p-value of the slope."""
        return float(self.result.pvalues[0])

    def expected(self, x: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """E[R_D | x] for x on the 0-1 scale (lies within the observed category range)."""
        exog = np.asarray(x, dtype=float).reshape(-1, 1)
        if params is None:
            probs = self.result.predict(exog=exog)
        else:
            probs = self.model.predict(params, exog=exog)
        return np.asarray(probs) @ self.labels

    @cached_property
    def auc(self) -> float:
        return float(trapezoid(self.expected(GRID), GRID))

    @cached_property
    def auc_support(self) -> float:
        """Mean curve height over the cell's own observed belief range."""
        lo, hi = self.bc.min() / 100, self.bc.max() / 100
        grid = np.linspace(lo, hi, 1001)
        return float(trapezoid(self.expected(grid), grid) / (hi - lo))

    def coefficient_band(self, x: np.ndarray, rng: np.random.Generator, n_draws: int = 300):
        """Pointwise 2.5/97.5 percentiles of E[R_D | x] over multivariate-normal
        coefficient draws from the fitted covariance (approximate curve band)."""
        draws = rng.multivariate_normal(self.result.params, self.result.cov_params(), n_draws)
        curves = []
        for params in draws:
            try:
                curves.append(self.expected(x, params))
            except Exception:
                pass
        if len(curves) < 20:
            raise RuntimeError('too few valid coefficient draws for a curve band')
        curves = np.asarray(curves)
        return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def fit_olr(bc, rd) -> OLRFit:
    """Fit the proportional-odds model by BFGS maximum likelihood (B_C on 0-100)."""
    bc = np.asarray(bc, dtype=float)
    rd = np.asarray(rd, dtype=int)
    model = OrderedModel(rd, (bc / 100.0).reshape(-1, 1), distr='logit')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = model.fit(method='bfgs', disp=False, maxiter=1000)
    return OLRFit(model, result, bc)


def auc_beta(bc, rd) -> tuple[float, float]:
    fit = fit_olr(bc, rd)
    return fit.auc, fit.beta
