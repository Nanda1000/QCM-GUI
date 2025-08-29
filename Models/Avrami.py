import numpy as np
from scipy.optimize import curve_fit

def formula(t, k, n):
    return 1 - np.exp(-k * t**n)

def compute_X_t(f_array, f0, finf):
    f_array = np.array(f_array, dtype=float)
    if finf < f0:
        # Sign Flip as the frequency value range is increasing, it gets negative
        """To get correct crystallinity values, sign flip is needed"""
        return (f0 - f_array) / (f0 - finf)
    else:
        
        return (f_array - f0) / (finf - f0)


def validate_data(t, f_array):
    """
    Validate data for Avrami fitting.
    
    Args:
        t: time array
        f_array: frequency array
    
    Returns:
        tuple: (t_clean, f_clean) - cleaned arrays
    """
    # Convert to numpy arrays
    t = np.array(t, dtype=float)
    f_array = np.array(f_array, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(t) | np.isnan(f_array))
    t_clean = t[mask]
    f_clean = f_array[mask]
    
    if len(t_clean) < 3:
        raise ValueError("Insufficient data points for fitting (need at least 3 points)")
    
    # Check for monotonic behavior (frequency should generally decrease during crystallization)
    if len(f_clean) > 3:
        # Check if frequency changes significantly
        freq_range = np.max(f_clean) - np.min(f_clean)
        if freq_range < 1e-6:
            raise ValueError("Frequency data shows insufficient variation for Better fitting")
    
    return t_clean, f_clean



def fit(t, f_array):
    t_clean, f_clean = validate_data(t, f_array)

    f0 = f_clean[0]
    finf = f_clean[-1]

    if abs(f0 - finf) < 50:  # Hz threshold
        print("Warning: small frequency change, results may be noisy")


    X_t = compute_X_t(f_clean, f0, finf)

    if np.any(np.isnan(X_t)) or np.any(np.isinf(X_t)):
        raise ValueError("X(t) contains NaN or Inf values.")

    if np.std(X_t) < 1e-3:
        raise ValueError("Crystallization fraction variation is too low.")

    bounds = ([1e-10, 0.1], [1e-1, 10.0])
    p0 = [1e-4, 1.0]  # initial guess

    try:
        popt, _ = curve_fit(formula, t_clean, X_t, p0=p0, bounds=bounds, maxfev=20000)
        k, n = popt

        if k <= 0 or n <= 0:
            raise ValueError("Fitted parameters must be positive.")
        return k, n
    except RuntimeError as e:
        raise ValueError(f"Fit did not converge: {str(e)}")
    except Exception as e:
        raise ValueError(f"Fitting error: {str(e)}")
