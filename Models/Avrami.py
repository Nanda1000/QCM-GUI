import numpy as np
from scipy.optimize import curve_fit

def formula(k, n, t):
    return 1 - np.exp(-k * t**n)

def compute_X_t(f_array, f0, finf):
    f_array = np.array(f_array, dtype=float)
    numerator = f0 - f_array
    denominator = f0 - finf
    if denominator == 0:
        raise ValueError("Initial and final frequencies must not be equal.")
    X_t = numerator / denominator
    X_t = np.clip(X_t, 0, 1)
    return X_t

def validate_data_for_avrami(t, f_array, f0, finf):
    """
    Validate data suitability for Avrami fitting.
    
    Returns:
        dict: Validation results with status and messages
    """
    validation = {
        'is_valid': True,
        'messages': [],
        'warnings': []
    }
    
    # Check basic requirements
    if len(t) < 3:
        validation['is_valid'] = False
        validation['messages'].append("At least 3 data points are required")
    
    if f0 == finf:
        validation['is_valid'] = False
        validation['messages'].append("Initial and final frequencies must be different")
    
    if f0 <= 0 or finf <= 0:
        validation['is_valid'] = False
        validation['messages'].append("Frequencies must be positive")
    
    # Check data quality
    if len(t) > 0:
        X_t = compute_X_t(f_array, f0, finf)
        
        # Check for sufficient variation
        if np.std(X_t) < 1e-6:
            validation['is_valid'] = False
            validation['messages'].append("Insufficient variation in crystallization fraction")
        elif np.std(X_t) < 0.01:
            validation['warnings'].append("Low variation in crystallization fraction - fit may be unreliable")
        
        # Check for monotonic behavior
        if not np.all(np.diff(X_t) >= -1e-6):  # Allow small numerical errors
            validation['warnings'].append("Crystallization fraction is not monotonically increasing")
        
        # Check frequency range
        freq_range = abs(f0 - finf)
        if freq_range < 1e-6:
            validation['is_valid'] = False
            validation['messages'].append("Frequency range is too small")
        elif freq_range < 1e-3:
            validation['warnings'].append("Small frequency range - results may be unreliable")
    
    return validation

def fit(t, f_array):
    # Input validation
    if len(t) == 0 or len(f_array) == 0:
        raise ValueError("Time and frequency arrays cannot be empty")
    
    if len(t) != len(f_array):
        raise ValueError(f"Time array length ({len(t)}) must match frequency array length ({len(f_array)})")
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(t) & np.isfinite(f_array)
    if not np.any(valid_mask):
        raise ValueError("No valid data points found after removing NaN/infinite values")
    
    t_clean = t[valid_mask]
    f_clean = f_array[valid_mask]
    
    if len(t_clean) < 3:
        raise ValueError("At least 3 valid data points are required for fitting")
    
    # Calculate X_t
    f0 = f_clean[0]
    finf = f_clean[-1]
    
    if f0 == finf:
        raise ValueError("Initial and final frequencies must be different for meaningful crystallization analysis")
    
    X_t = compute_X_t(f_clean, f0, finf)
    
    # Check if X_t has sufficient variation
    if np.std(X_t) < 1e-6:
        raise ValueError("Insufficient variation in crystallization fraction for meaningful fitting")
    
    # Ensure time values are non-negative and monotonically increasing
    if np.any(t_clean < 0):
        raise ValueError("Time values must be non-negative")
    
    if not np.all(np.diff(t_clean) >= 0):
        raise ValueError("Time values must be monotonically non-decreasing")
    
    try:
        # Use more robust initial parameters and bounds
        p0 = [1e-4, 1.0]  # Initial guess for k and n
        bounds = ([1e-8, 0.1], [1e2, 10.0])  # Reasonable bounds for k and n
        
        # Try different initial conditions if the first one fails
        initial_conditions = [
            [1e-4, 1.0],   # Standard
            [1e-3, 2.0],   # Higher k, n=2
            [1e-5, 0.5],   # Lower k, n=0.5
            [1e-2, 3.0],   # Much higher k, n=3
        ]
        
        best_fit = None
        best_error = np.inf
        
        for p0_try in initial_conditions:
            try:
                popt, pcov = curve_fit(formula, t_clean, X_t, p0=p0_try, bounds=bounds, maxfev=10000)
                
                # Calculate fitting error
                y_fit = formula(popt[0], popt[1], t_clean)
                error = np.mean((X_t - y_fit)**2)
                
                if error < best_error:
                    best_error = error
                    best_fit = popt
                    
            except (RuntimeError, ValueError, TypeError):
                continue
        
        if best_fit is None:
            raise ValueError("Failed to converge to optimal parameters with any initial condition. Check data quality.")
        
        # Validate the fitted parameters
        k, n = best_fit
        if not np.isfinite(k) or not np.isfinite(n):
            raise ValueError("Fitted parameters are not finite")
        
        if k <= 0 or n <= 0:
            raise ValueError("Fitted parameters must be positive")
        
        return best_fit
        
    except RuntimeError as e:
        if "Optimal parameters not found" in str(e):
            raise ValueError("Failed to converge to optimal parameters. Try different initial conditions or check data quality.")
        else:
            raise ValueError(f"Curve fitting failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error during fitting: {str(e)}")
