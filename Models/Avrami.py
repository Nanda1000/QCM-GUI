#Avrami fitting model

import numpy as np
from scipy.optimize import curve_fit

def parameter(freqs, impedance):
    Z1 = np.abs(impedance)
    
    min_index = np.argmin(Z1)
    
    ft = freqs[min_index]
    
    return ft
    
"""Avrami fitting model for crystallization kinetics
X(t) =  △f(t)/△fmax
△f(t) = f0 - f(t)
△fmax = f0 - f_inf

f_inf is the final resonance frequency once the crystallization is complete

X(t) = 1 - exp(-kt^n)

"""

def avrami(f_array, f0, fmax, ft):
    fmax = min(f_array)
    fmaxed = f0 - fmax
    delta_f = f0 - ft
    if fmaxed == 0:
        return
    return delta_f / fmaxed

def formula(k, n, t):
    X = 1 - np.exp(-1 * k * t**n)
    return X

#Fitting the model


def fit(t, f_array):
    f0 = f_array[0]
    f_max = f_array[-1]
    X_t = avrami(f_array, f0, f_max)
    pop,_ = curve_fit(formula, t, X_t, p0=[1e-4, 1.0])
    return pop
    
