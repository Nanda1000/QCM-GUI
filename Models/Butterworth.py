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
    """
    Compute total BVD impedance at frequencies fs.
    fs: frequencies array
    Rm, Lm, Cm: motional parameters
    C0: static capacitance
    Returns: Z_total
    """
    w = 2 * np.pi * fs
    j = 1j

    # Avoid division by zero
    w = np.where(w == 0, 1e-12, w)
    Cm = np.where(Cm == 0, 1e-12, Cm)
    C0 = np.where(C0 == 0, 1e-12, C0)

    Zm = Rm + j * w * Lm + 1 / (j * w * Cm)
    Z0 = 1 / (j * w * C0)
    Y = 1 / Zm + 1 / Z0
    Y = np.where(Y == 0, 1e-12, Y)
    return 1 / Y


def calculate_Q_from_impedance(freqs, Z1):
    abs_Z = np.abs(Z1)
    min_idx = np.argmin(abs_Z)
    fs = freqs[min_idx]
    min_Z = abs_Z[min_idx]

    threshold = min_Z * np.sqrt(2)

    # Robust spline fitting
    s_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    spline = None
    for s in s_values:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spline = UnivariateSpline(freqs, abs_Z - threshold, s)
            fine_freqs = np.linspace(freqs[0], freqs[-1], 5000)
            fine_Z = spline(fine_freqs)
            break
        except Exception:
            spline = None
            continue
    else:
        spline = None

    # If spline fails or no roots, fallback to nearest points
    if spline is not None:
        roots = spline.roots()
        left_roots = roots[roots < fs]
        right_roots = roots[roots > fs]

        if len(left_roots) == 0:
            left_roots = [fine_freqs[np.where(fine_Z < 0)[0][0]]] if np.any(fine_Z < 0) else [fs]
        if len(right_roots) == 0:
            right_roots = [fine_freqs[np.where(fine_Z < 0)[0][-1]]] if np.any(fine_Z < 0) else [fs]

        bandwidth = right_roots[-1] - left_roots[0]
    else:
        # fallback to simple linear search
        indices = np.where(abs_Z <= threshold)[0]
        if len(indices) >= 2:
            bandwidth = freqs[indices[-1]] - freqs[indices[0]]
        else:
            bandwidth = np.nan

    Q = fs / bandwidth if bandwidth > 0 else np.nan
    return fs, bandwidth, Q



def parameter(freqs, impedance):
    """
    Extract crystal parameters:
    Rm, Lm, Cm, C0, fs, Q, D
    Only uses impedance half-power method for Q.
    """
    freqs = np.array(freqs, dtype=float)
    Z = np.array(impedance, dtype=complex)

    # Clean invalid data
    mask = np.isfinite(freqs) & np.isfinite(Z)
    freqs, Z = freqs[mask], Z[mask]
    if len(freqs) < 5:
        return None, None, None, None, None, None, None

    # Series resonance fs (minimum |Z|)
    absZ = np.abs(Z)
    fs_index = np.argmin(absZ)
    if 1 < fs_index < len(absZ)-2:
        f_fit = freqs[fs_index-1:fs_index+2]
        z_fit = absZ[fs_index-1:fs_index+2]
        coeffs = np.polyfit(f_fit, z_fit, 2)
        fs = -coeffs[1] / (2 * coeffs[0])
    else:
        fs = freqs[fs_index]

    # Motional resistance
    Rm = abs(np.real(Z[fs_index]))

  
    _, _, Q = calculate_Q_from_impedance(freqs, Z)
    print(f"[INFO] Q from impedance half-power: {Q:.1f}")

 
    if not np.isnan(Q) and Q > 0:
        Lm = Rm * Q / (2 * np.pi * fs)
        Cm = 1 / (Lm * (2 * np.pi * fs)**2)
    else:
        Lm, Cm = np.nan, np.nan
        print("[WARNING] Q calculation failed, Lm and Cm will be NaN")

    try:
        Y = 1 / Z
        B = Y.imag
        C0 = abs(np.mean(B[-10:] / (2 * np.pi * freqs[-10:])))
    except:
        C0 = np.nan

    D = 1 / Q if not np.isnan(Q) and Q > 0 else np.nan

    return Rm, Lm, Cm, C0, fs, Q, D
