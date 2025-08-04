import numpy as np
from scipy.optimize import curve_fit

def formula(t, k, n):
    """Avrami equation: X(t) = 1 - exp(-k * t^n)
    
    Parameters:
    t: time array
    k: crystallization rate constant
    n: Avrami exponent
    """
    return 1 - np.exp(-k * np.power(t, n))

def compute_X_t(f_array, f0, finf):
    """Compute crystallization fraction from frequency data"""
    f_array = np.array(f_array, dtype=float)
    numerator = f0 - f_array
    denominator = f0 - finf
    if abs(denominator) < 1e-10:  # More robust check for near-zero denominator
        raise ValueError("Initial and final frequencies must not be equal (f0 != finf).")
    X_t = numerator / denominator
    X_t = np.clip(X_t, 0, 1)
    return X_t

def fit(t, f_array, f0=None, finf=None):
    """Fit Avrami model to time-frequency data
    
    Parameters:
    t: time array
    f_array: frequency array
    f0: initial frequency (optional, will use first value if not provided)
    finf: final frequency (optional, will use last value if not provided)
    
    Returns:
    tuple: (k, n) - crystallization rate and Avrami exponent
    """
    # Handle input validation
    t = np.array(t, dtype=float)
    f_array = np.array(f_array, dtype=float)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(t) & np.isfinite(f_array)
    if not np.any(valid_mask):
        raise ValueError("No valid data points found")
    
    t_clean = t[valid_mask]
    f_clean = f_array[valid_mask]
    
    if len(t_clean) < 3:
        raise ValueError("Need at least 3 valid data points for fitting")
    
    # Use provided f0 and finf, or compute from data
    if f0 is None:
        f0 = f_clean[0]
    if finf is None:
        finf = f_clean[-1]
    
    # Compute crystallization fraction
    X_t = compute_X_t(f_clean, f0, finf)
    
    # Check if we have meaningful variation in X_t
    if np.std(X_t) < 1e-10:
        raise ValueError("Insufficient variation in crystallization fraction for fitting")
    
    try:
        # Fit with better initial guess and bounds
        # Initial guess: k=1e-3, n=1.0
        # Bounds: k > 0, 0.1 < n < 5 (typical range for Avrami exponent)
        popt, pcov = curve_fit(formula, t_clean, X_t, 
                              p0=[1e-3, 1.0], 
                              bounds=([1e-10, 0.1], [np.inf, 5.0]),
                              maxfev=5000)
        
        # Check if fit is reasonable
        k_fit, n_fit = popt
        if not (np.isfinite(k_fit) and np.isfinite(n_fit)):
            raise ValueError("Fit resulted in non-finite parameters")
            
        return popt
        
    except Exception as e:
        raise ValueError(f"Avrami fitting failed: {str(e)}")
