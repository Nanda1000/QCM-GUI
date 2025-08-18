import numpy as np


class phaseshiftmethod:
    def phaseshift(self, s21_values):
        s21 = np.abs(s21_values)
        s21_db = 20*np.log10(s21)
        
        phase = np.angle(s21_values, deg=True)
        
        
        
        return phase, s21, s21_db

    def phaseshiftfrequency(self,phase, freqs, s21_db):
        
        targets = [-45, 0, 45]
        results = {}
        
        
        for target in targets:
            index = (np.abs(phase - target)).argmin()
            results[target] = {
                "freq": freqs[index],
                "phase": phase[index],
                "s21_db": s21_db[index]
        }

        return results

    def phaseshiftcalculation(self, results):
        
        try:
            fs = results[0]["freq"]  
            f1 = results[45]["freq"]    
            f2 = results[-45]["freq"]    
            s21_logmag = results[0]["s21_db"]
            Rm = 2 * 50 * (10 **(-s21_logmag/20)-1)
            Reff = 2 *50 + Rm
            delta_f = abs(f1 - f2)
            Cm = delta_f/(2*np.pi* fs**2 * Reff)
            Lm = Reff/(2*np.pi*delta_f)
        
            return Cm, Rm, Lm, Reff, delta_f
        except Exception as e:
            return None, None, None, None, None, None, None, None