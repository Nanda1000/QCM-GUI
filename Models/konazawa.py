#Konazawa equation

import numpy as np

def parameter_konazawa(freqs, impedance):
    Z1 = np.abs(impedance)
    
    #Check for Series or Parllel resonant frequency
    
    
    min_index = np.argmin(Z1)
    fs = freqs[min_index]
    
    max_index = np.argmax(Z1)
    fp = freqs[max_index]
    
    if Z1[min_index] < Z1[max_index]:
        f0 = fs
        return fs
    else:
        f0 = fp
    return f0
    


def konazawa(f0, p, u, ft, p1):
    f = f0 -ft
    n = (f*(np.sqrt((np.pi * u * p)/(p1 * f0**3))))**2
    return n