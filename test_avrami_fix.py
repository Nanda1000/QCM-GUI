#!/usr/bin/env python3
"""
Test script to verify Avrami fitting fixes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Models.Avrami import fit, compute_X_t, formula, validate_data_for_avrami

def create_test_data():
    """Create test data for Avrami fitting"""
    # Create time array
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(100)]
    t_seconds = np.array([i for i in range(100)])
    
    # Create frequency data with crystallization behavior
    f0 = 10000000  # 10 MHz initial frequency
    finf = 9990000  # 9.99 MHz final frequency
    
    # Simulate crystallization with some noise
    k_true = 0.001
    n_true = 2.0
    X_t = formula(k_true, n_true, t_seconds)
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.01, len(X_t))
    X_t_noisy = np.clip(X_t + noise, 0, 1)
    
    # Convert back to frequency
    freqs = f0 - X_t_noisy * (f0 - finf)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Frequency(Hz)': freqs,
        'Resistance(Ω)': np.random.uniform(100, 200, len(timestamps)),
        'Phase': np.random.uniform(-90, 90, len(timestamps))
    })
    
    return data, f0, finf, k_true, n_true

def test_avrami_fitting():
    """Test the Avrami fitting functionality"""
    print("Testing Avrami fitting fixes...")
    
    # Create test data
    data, f0, finf, k_true, n_true = create_test_data()
    
    # Extract time and frequency arrays - use simple array instead of pandas
    t_seconds = np.array([i for i in range(100)])
    freqs = np.array(data["Frequency(Hz)"])
    
    print(f"Test data created:")
    print(f"  - Time points: {len(t_seconds)}")
    print(f"  - Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")
    print(f"  - True parameters: k={k_true:.6f}, n={n_true:.2f}")
    
    # Test validation
    print("\nTesting data validation...")
    validation = validate_data_for_avrami(t_seconds, freqs, f0, finf)
    print(f"  - Valid: {validation['is_valid']}")
    if validation['messages']:
        print(f"  - Errors: {validation['messages']}")
    if validation['warnings']:
        print(f"  - Warnings: {validation['warnings']}")
    
    # Test fitting
    print("\nTesting Avrami fitting...")
    try:
        k_fitted, n_fitted = fit(t_seconds, freqs)
        print(f"  - Fitted k: {k_fitted:.6f} (true: {k_true:.6f})")
        print(f"  - Fitted n: {n_fitted:.2f} (true: {n_true:.2f})")
        print(f"  - k error: {abs(k_fitted - k_true)/k_true*100:.2f}%")
        print(f"  - n error: {abs(n_fitted - n_true)/n_true*100:.2f}%")
        print("  ✓ Fitting successful!")
    except Exception as e:
        print(f"  ✗ Fitting failed: {e}")
        return False
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with insufficient data
    try:
        fit(t_seconds[:2], freqs[:2])
        print("  ✗ Should have failed with insufficient data")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected insufficient data: {e}")
    
    # Test with identical frequencies
    try:
        identical_freqs = np.full_like(freqs, f0)
        fit(t_seconds, identical_freqs)
        print("  ✗ Should have failed with identical frequencies")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected identical frequencies: {e}")
    
    print("\n✓ All tests passed!")
    return True

def test_error_handling():
    """Test error handling scenarios"""
    print("\nTesting error handling...")
    
    # Test empty arrays
    try:
        fit([], [])
        print("  ✗ Should have failed with empty arrays")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly handled empty arrays: {e}")
    
    # Test mismatched lengths
    try:
        fit([1, 2, 3], [1, 2])
        print("  ✗ Should have failed with mismatched lengths")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly handled mismatched lengths: {e}")
    
    # Test with NaN values
    try:
        t_with_nan = np.array([1, 2, np.nan, 4])
        f_with_nan = np.array([1, 2, 3, 4])
        fit(t_with_nan, f_with_nan)
        print("  ✓ Correctly handled NaN values")
    except Exception as e:
        print(f"  ✗ Failed to handle NaN values: {e}")
        return False
    
    print("  ✓ All error handling tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("AVRAMI FITTING FIX TEST")
    print("=" * 50)
    
    success = True
    success &= test_avrami_fitting()
    success &= test_error_handling()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED - Avrami fitting fixes are working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review the fixes")
    print("=" * 50)