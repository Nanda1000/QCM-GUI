#Sauerbrey Equation


import numpy as np

#Gathering freqency data and calculating it from acquired data from nanovna

def parameter_sauerbrey(freqs, impedance):
    Z1 = np.abs(impedance)
    
    min_index = np.argmin(Z1)
    
    ft = freqs[min_index]
    
    
    
    return ft
    
"""
Sauerbrey equation to estimate mass deposited from frequency shift.
    f0: initial resonance frequency (Hz)
    ft: shifted frequency (Hz)
    area: active electrode area in m^2
    p: quartz density (kg/m^3)
    mu: shear modulus (Pa)
    
    △m = mass change = - △fA√(p*mu)/f0^2

"""

def sauerbrey(f0, p, u, ft, A):
    delta_f = f0 - ft  # Frequency change
    m = -delta_f * A * np.sqrt(p * u) / (f0 ** 2)  # Negative sign for mass loading
    
    return m
    