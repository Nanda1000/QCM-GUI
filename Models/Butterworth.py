# This model is to fit electrical parameters or to find them

import numpy as np
import serial.tools.list_ports
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import pyqtgraph as pg
from test.test import acquire_data

"""Some of the key parameters to find the motional resistance(damping/viscosity effect)
Resonance Frequency which can tell about mass loading effect and for the full modeling
we need to find Motional Inductance and Capacitance and also Static Capacitance

There are some key formulas to use, Below are the ones

    Zm(f) = Rm + 2πjfLm + 1/2πjfCm (Motional Impedance)
    
    Zco(f) = 1/2πjfC0 (Static Capacitance Impedance)
    
    1/Z(f) = 1/Zm(f) + 1/Zc0(f) (Overall Impedance)

"""

def butterworth(fs, Rm, Lm, Cm, C0):
    #frequency need to be entered as the value can be obtained from where impedance is minimum  is magnitude of S11
    
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
    return Z_tot

    
"""Interpolation done to find the points crossed for qualify factor or to find the frequency points"""
    
def half_power_threshold(freqs, Z1, R):
    half_power_freqs = []
    for i in range(1, len(Z1)):
       if(Z1[i-1]-R) * (Z1[i] -R) < 0:
          f1, f2 = freqs[i-1], freqs[i]
          z1, z2 = Z1[i-1], Z1[i]
          slope = (z2 -z1)/(f2-f1)
          cross = f1 + (R -z1) / slope
          half_power_freqs.append(cross)
          
    return half_power_freqs

def parameter(freqs, impedance):
    Z1 = np.abs(impedance)

    # Use spline smoothing to robustly find resonance frequency
    spline = UnivariateSpline(freqs, Z1, s=1)
    fine_freqs = np.linspace(freqs[0], freqs[-1], 5000)
    fine_Z1 = spline(fine_freqs)
    min_index = np.argmin(fine_Z1)
    fs = fine_freqs[min_index]

    # Find nearest actual data index to fs
    nearest_index = np.argmin(np.abs(freqs - fs))
    Zfs = impedance[nearest_index]
    Rm = Zfs.real

    # Half-power bandwidth
    R = Rm * np.sqrt(2)
    half_power_freqs = half_power_threshold(freqs, Z1, R)
    if len(half_power_freqs) >= 2:
        df = abs(half_power_freqs[-1] - half_power_freqs[0])
        Q = fs / df
    else:
        Q = 8000  # fallback

    Lm = Rm * Q / (2 * np.pi * fs)
    Cm = 1 / (Lm * (2 * np.pi * fs) ** 2)

    # Estimate C0 from multiple high-frequency points
    high_freqs = freqs[-10:]
    high_ImZ = np.imag(impedance[-10:])
    if np.all(high_ImZ != 0):
        C0_vals = -1 / (2 * np.pi * high_freqs * high_ImZ)
        C0 = np.mean(C0_vals)
    else:
        C0 = 1e-12  # default fallback

    return Rm, Lm, Cm, C0, fs


# Residuals for fitting model
def fit_data(parameters, fs, Z_measured):
    Rm, Lm, Cm, C0 = parameters
    Z_model = butterworth(fs, Rm, Lm, Cm, C0)
    residual = np.concatenate([np.real(Z_model - Z_measured), np.imag(Z_model - Z_measured)])
    return residual


# --- Main execution ---

if __name__ == "__main__":
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise Exception("No serial ports found.")

    device_path = ports[0].device
    freqs, _, impedance, _, _, _ = acquire_data(device_path)

    Rm, Lm, Cm, C0, fs = parameter(freqs, impedance)
    initial_guess = [Rm, Lm, Cm, C0]
    result = least_squares(fit_data, initial_guess, args=(freqs, impedance))

    if not result.success:
        raise RuntimeError("Fit failed:", result.message)

    Rm_fit, Lm_fit, Cm_fit, C0_fit = result.x
    Z_fit = butterworth(freqs, Rm_fit, Lm_fit, Cm_fit, C0_fit)
