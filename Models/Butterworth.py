# This model is to fit electrical parameters or to find them

import numpy as np
import serial.tools.list_ports
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import pyqtgraph as pg

import warnings

"""Some of the key parameters to find the motional resistance(damping/viscosity effect)
Resonance Frequency which can tell about mass loading effect and for the full modeling
we need to find Motional Inductance and Capacitance and also Static Capacitance

There are some key formulas to use, Below are the ones

    Zm(f) = Rm + 2πjfLm + 1/2πjfCm (Motional Impedance)
    
    Zco(f) = 1/2πjfC0 (Static Capacitance Impedance)
    
    1/Z(f) = 1/Zm(f) + 1/Zc0(f) (Overall Impedance)

"""

def butterworth(fs, Rm, Lm, Cm, C0):
    #frequency need to be entered as the value can be obtained from where impedance is minimum or maximum  is magnitude of S21 ans S11
    
    w = 2 * np.pi * fs
    j = 1j
    

    if np.any(w == 0):
        w = np.where(w == 0, 1e-12, w)  
    if np.any(Cm == 0):
        Cm = np.where(Cm == 0, 1e-12, Cm)  
    if np.any(C0 == 0):
        C0 = np.where(C0 == 0, 1e-12, C0) 
    
    Zm = Rm + j * w * Lm + 1 / (j * w * Cm)
    Z0 = 1 / (j * w * C0)
    
    #Parallel 
    Y = 1/Zm + 1/Z0

    Y = np.where(Y == 0, 1e-12, Y)
    Z_tot = 1/Y
    return Z_tot, Zm

    
"""Interpolation done to find the points crossed for qualify factor or to find the frequency points"""
    
def half_power_threshold(freqs, Z1, R):
    half_power_freqs = []
    for i in range(1, len(Z1)):
       if(Z1[i-1]-R) * (Z1[i] -R) <= 0:
          f1, f2 = freqs[i-1], freqs[i]
          z1, z2 = Z1[i-1], Z1[i]
          slope = (z2 -z1)/(f2-f1)
          cross = f1 + (R -z1) / slope
          half_power_freqs.append(cross)
          
    return half_power_freqs



def parameter(freqs, impedance, Resistance=None):
    freqs = np.array(freqs)
    impedance = np.array(impedance)

    # Handle insufficient data for spline (less than 4 points)
    if len(freqs) < 4:
        min_index = np.argmin(np.abs(impedance))
        fs = freqs[min_index]
        Rm = Resistance[min_index] if Resistance is not None else impedance[min_index].real
        return Rm, None, None, None, fs

    # Handle real vs complex impedance
    if np.iscomplexobj(impedance):
        Z1 = np.abs(impedance)
    else:
        warnings.warn("Only real impedance provided. Cannot extract full electrical parameters.")
        Z1 = impedance

    # Spline smoothing to find resonance frequency fs
    try:
        spline = UnivariateSpline(freqs, Z1, s=1)
        fine_freqs = np.linspace(freqs[0], freqs[-1], 5000)
        fine_Z1 = spline(fine_freqs)
    except Exception as e:
        raise RuntimeError(f"Spline fitting failed: {e}")

    min_index = np.argmin(Z1)
    fs = freqs[min_index]
    
    max_index = np.argmax(Z1)
    fp = freqs[max_index]
    
    if Z1[min_index] < Z1[max_index]:
        f = fs
        return f
    else:
        fp = fp
        return fp


    # Find nearest real value for fs and Rm
    nearest_index = np.argmin(np.abs(freqs - f))
    Zfs = impedance[nearest_index]
    Rm = Zfs.real

    # Half-power bandwidth for Q calculation
    R = Rm * np.sqrt(2)
    half_power_freqs = half_power_threshold(freqs, Z1, R)
    if len(half_power_freqs) >= 2:
        df = abs(half_power_freqs[-1] - half_power_freqs[0])
        Q = f / df
    else:
        Q = 8000  # fallback

    # Compute Lm and Cm
    Lm = Rm * Q / (2 * np.pi * f) if f > 0 else 1e-6
    Cm = 1 / (Lm * (2 * np.pi * f) ** 2) if Lm > 0 and f > 0 else 1e-12

    # Estimate static capacitance C0
    try:
        high_freqs = freqs[-10:]
        high_ImZ = np.imag(impedance[-10:])
        if np.all(high_ImZ != 0):
            C0_vals = -1 / (2 * np.pi * high_freqs * high_ImZ)
            C0 = np.mean(C0_vals)
        else:
            C0 = 1e-12
    except Exception:
        C0 = 1e-12

    return Rm, Lm, Cm, C0, fs

# Residuals for fitting model
def fit_data(parameters, f, Z_measured):
    Rm, Lm, Cm, C0 = parameters
    Z_model = butterworth(f, Rm, Lm, Cm, C0)
    residual = np.concatenate([np.real(Z_model - Z_measured), np.imag(Z_model - Z_measured)])
    return residual



