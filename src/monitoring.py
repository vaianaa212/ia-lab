from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # PSI for a single numeric feature
    eps = 1e-6
    q = np.linspace(0, 1, bins+1)
    cuts = np.unique(np.quantile(expected, q))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_perc = (e_counts / max(e_counts.sum(),1)) + eps
    a_perc = (a_counts / max(a_counts.sum(),1)) + eps
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def drift_report(df_ref: pd.DataFrame, df_new: pd.DataFrame, numeric_cols: list) -> Dict[str, float]:
    out = {}
    for c in numeric_cols:
        out[c] = population_stability_index(df_ref[c].to_numpy(), df_new[c].to_numpy())
    return out
