import numpy as np
from scipy.optimize import curve_fit

def formula(k, n, t):
    return 1 - np.exp(-k * t**n)

def compute_X_t(f_array, f0, finf):
    f_array = np.array(f_array, dtype=float)
    numerator = f0 - f_array
    denominator = f0 - finf
    if denominator == 0:
        raise ValueError("Initial and final frequencies must not be equal.")
    X_t = numerator / denominator
    X_t = np.clip(X_t, 0, 1)
    return X_t

def fit(t, f_array):
    f0 = f_array[0]
    finf = f_array[-1]

    X_t = compute_X_t(f_array, f0, finf)
    popt, _ = curve_fit(formula, t, X_t, p0=[1e-4, 1.0], bounds=(0, np.inf))
    return popt
