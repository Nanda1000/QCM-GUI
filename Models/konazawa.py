#Konazawa equation

import numpy as np

def parameter_konazawa(freqs, impedance):
    Z1 = np.abs(impedance)
    
    #Check for Series or Parllel resonant frequency
    Admitt = 1/Z1
    conduct = Admitt.real
    
    max_index = np.argmax(conduct)
    fs = freqs[max_index]
    
    ft = fs
    
    return ft
    


def konazawa(f0, p, u, ft, p1):
    f = f0 -ft
    n = (f*(np.sqrt((np.pi * u * p)/(p1 * f0**3))))**2
    return n