import numpy as np

class phaseshiftmethod:
    def phaseshift(self, s21_values):
        s21 = np.abs(s21_values)
        s21_db = 20*np.log10(s21)
        phase = np.angle(s21_values, deg=True)
        return phase, s21, s21_db
    
    def phaseshiftfrequency(self, phase, freqs, s21_db):
        # Find resonance frequency (peak of |S21|)
        peak_idx = np.argmax(s21_db)
        fs = freqs[peak_idx]
        peak_val = s21_db[peak_idx]
        
        # Find -3dB points
        target_val = peak_val - 3
        
        # Left of peak (-3dB point)
        left_idx = np.where(s21_db[:peak_idx] <= target_val)[0]
        if len(left_idx) > 0:
            idx = left_idx[-1]
            # Interpolate for better accuracy if not at boundary
            if idx < peak_idx - 1:
                # Linear interpolation between points
                x1, x2 = freqs[idx], freqs[idx + 1]
                y1, y2 = s21_db[idx], s21_db[idx + 1]
                f1 = x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
            else:
                f1 = freqs[idx]
        else:
            f1 = freqs[0]
        
        # Right of peak (-3dB point)
        right_idx = np.where(s21_db[peak_idx:] <= target_val)[0]
        if len(right_idx) > 0:
            idx = peak_idx + right_idx[0]
            # Interpolate for better accuracy if not at boundary
            if idx < len(freqs) - 1:
                x1, x2 = freqs[idx], freqs[idx + 1]
                y1, y2 = s21_db[idx], s21_db[idx + 1]
                f2 = x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
            else:
                f2 = freqs[idx]
        else:
            f2 = freqs[-1] 
        
        results = {
            "fs": fs,
            "peak_val": peak_val,
            "f1": f1,
            "f2": f2,
            "s21_db": s21_db[peak_idx],
        }
        return results
    
    def phaseshiftcalculation(self, results):
        try:
            fs = results["fs"]
            f1 = results["f1"]
            f2 = results["f2"]
            peak_val = results["peak_val"]
            
            BW = abs(f2 - f1)
            Q = fs / BW if BW != 0 else np.inf
            
            # Resonance resistance (series circuit in matched system, Z0 = 50 Ω)
            lin_mag = 10**(peak_val/20)
            Rm = 100 * (1/lin_mag - 1) 
            Rm = max(Rm, 0.001)  
            
            # Equivalent circuit parameters
            if Q > 0 and fs > 0 and Rm > 0:
                Lm = (Q * Rm) / (2 * np.pi * fs)
                Cm = 1 / (2 * np.pi * fs * Q * Rm)
            else:
                Lm = None
                Cm = None
            
            return Cm, Rm, Lm, Q, BW, fs
            
        except Exception as e:
            return None, None, None, None, None, None, None, None