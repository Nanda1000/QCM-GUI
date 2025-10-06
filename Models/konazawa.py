#Konazawa equation

import numpy as np

def parameter_konazawa(freqs, impedance):
    Z1 = np.abs(impedance)
    
    #Check for Series or Parllel resonant frequency
    Admitt = 1/Z1
    conduct = Admitt.real
    
    max_index = np.argmax(conduct)
    fs = freqs[max_index]
    
    
    ft = fs - 100 #Subtracting 100Hz offset to estimate frequency shift
    
    return ft
    
def calculate_Q_from_impedance(freqs, Z):
    """
    Calculate Q using the 3 dB bandwidth method (series resonance).
    freqs : array of frequencies [Hz]
    Z     : array of complex impedance values
    Returns: fs, bandwidth, Q
    """
    abs_Z = np.abs(Z)

    # Find series resonance (min |Z|)
    min_idx = np.argmin(abs_Z)
    fs = freqs[min_idx]
    min_Z = abs_Z[min_idx]

    # 3 dB threshold
    threshold = min_Z * np.sqrt(2)

    # Find crossing points
    # Left side
    left_idx = np.where(abs_Z[:min_idx] >= threshold)[0]
    if len(left_idx) > 0:
        f1 = freqs[left_idx[-1]]
    else:
        f1 = freqs[0]

    # Right side
    right_idx = np.where(abs_Z[min_idx:] >= threshold)[0]
    if len(right_idx) > 0:
        f2 = freqs[min_idx:][right_idx[0]]
    else:
        f2 = freqs[-1]

    # Bandwidth and Q
    bandwidth = f2 - f1
    Q = fs / bandwidth if bandwidth > 0 else np.nan

    return fs, bandwidth, Q


def konazawa(f0, p, u, ft, p1):
    f = f0 -ft
    n = (f*(np.sqrt((np.pi * u * p)/(p1 * f0**3))))**2
    return n