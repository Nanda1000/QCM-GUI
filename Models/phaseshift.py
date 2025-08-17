from ..GUI.Acquire_data import NanoVNA
import numpy as np


class phaseshiftmethod:
    def phaseshift(self):
        s21 = np.abs(s21_values)
        s21_db = 20*np.log(s21)
        
        phase2 = []
        
        phase = np.atan2(parse_s21_data(self, lines))
        phase1 = phase * 180 /(np.pi)
        
        phase2 = np.array[phase1]
        
        
        return phase1, s21, s21_db

    def phaseshiftfrequency(s21, phase1, s21_db):
        
        fs, f1, f2 = s21[phase1]
        logmag = s21_db[phase1]
        
        
        for i in phase1:
            fs = s21[0]
            f1 = s21[45]
            f2 = s21[-45]
            s21_logmag = s21_db[0]
            
        return fs, f1, f2, s21_logmag

    def phaseshiftcalculation(fs, f1, f2, s21_logmag):
        Rm = 2 * 50 * (10 **(-s21_logmag/20)-1)
        Reff = 2 *50 + Rm
        delta_f = f1 - f2
        Cm = delta_f/(2*np.pi* fs**2 * Reff)
        Lm = Reff/(2*np.pi*delta_f)
        
        return Cm, Rm, Lm, Reff, delta_f