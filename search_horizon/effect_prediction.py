"""Repeated-draw moments for a conditional intervention effect."""
from __future__ import annotations

import numpy as np


def effect_moments(new, reuse, forecast=None, identical_requests=False):
    """Estimate mean effect and its square using independent draws per action.

    The cross-draw U-statistic removes generation variance from the squared
    effect. Finite-sample estimates can be negative and are retained.
    """
    a, b = np.asarray(new, dtype=float), np.asarray(reuse, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('Squared-effect estimation requires two draws per action')
    delta = float(a.mean()-b.mean())
    second = float(((a.sum()**2-(a*a).sum())/(len(a)*(len(a)-1)))
                   + ((b.sum()**2-(b*b).sum())/(len(b)*(len(b)-1)))
                   - 2*a.mean()*b.mean())
    if identical_requests:
        delta, second = 0.0, 0.0
    result = {'mean_effect': delta, 'squared_effect': second}
    if forecast is not None:
        result['noise_corrected_squared_error'] = float(forecast**2-2*forecast*delta+second)
        result['risk_difference_vs_zero_forecast'] = float(forecast**2-2*forecast*delta)
    return result
