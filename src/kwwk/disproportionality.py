"""Frequentist disproportionality baselines: PRR and ROR.

Both compare how often the event is reported with the target drug against how
often it is reported with every other drug, using the 2x2 table (a, b, c, d).
They are undefined when a cell on the diagonal of interest is zero; those
quarters return NaN and never raise an alarm.
"""

from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def _arrays(*cells):
    return [np.asarray(x, dtype=float) for x in cells]


def prr(a, b, c, d):
    """Proportional Reporting Ratio with its 95% interval and Yates chi-square.

    PRR = [a / (a + b)] / [c / (c + d)]   (Evans et al., 2001)
    """
    a, b, c, d = _arrays(a, b, c, d)
    n = a + b + c + d
    ok = (a > 0) & (c > 0) & (b + a > 0) & (c + d > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (a / (a + b)) / (c / (c + d))
        se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        lower = np.exp(np.log(value) - Z95 * se)
        upper = np.exp(np.log(value) + Z95 * se)
        num = n * (np.abs(a * d - b * c) - n / 2) ** 2
        den = (a + b) * (c + d) * (a + c) * (b + d)
        chi2 = np.where(den > 0, num / den, np.nan)
    nan = np.full_like(n, np.nan)
    return {
        "prr": np.where(ok, value, nan),
        "prr_lower": np.where(ok, lower, nan),
        "prr_upper": np.where(ok, upper, nan),
        "chi2": np.where(ok, chi2, nan),
    }


def ror(a, b, c, d):
    """Reporting Odds Ratio (a*d)/(b*c) with its 95% interval."""
    a, b, c, d = _arrays(a, b, c, d)
    ok = (a > 0) & (b > 0) & (c > 0) & (d > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        lower = np.exp(np.log(value) - Z95 * se)
        upper = np.exp(np.log(value) + Z95 * se)
    nan = np.full_like(value, np.nan)
    return {
        "ror": np.where(ok, value, nan),
        "ror_lower": np.where(ok, lower, nan),
        "ror_upper": np.where(ok, upper, nan),
    }


def evans_signal(a, prr_value, chi2, min_cases=3, min_prr=2.0, min_chi2=4.0):
    """Evans' screening rule: at least 3 cases, PRR >= 2 and chi-square >= 4."""
    a, prr_value, chi2 = _arrays(a, prr_value, chi2)
    with np.errstate(invalid="ignore"):
        return (a >= min_cases) & (prr_value >= min_prr) & (chi2 >= min_chi2)
