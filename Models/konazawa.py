#Konazawa equation

import numpy as np

def parameter(freqs, impedance):
    Z1 = np.abs(impedance)
    
    min_index = np.argmin(Z1)
    
    ft = freqs[min_index]
    
    return ft
    


def konazawa(f0, p, u, ft, p1):
    f = f0 -ft
    n = (f*(np.sqrt((np.pi * u * p)/(p1 * f0**3))))**2
    return n