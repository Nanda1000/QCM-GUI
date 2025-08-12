# Acquire_data_debug.py
import serial
import time

class NanoVNA_ASCII_Debug:
    def __init__(self, port, notify_callback=print):
        self.port = port
        self.ser = serial.Serial(port, baudrate=115200, timeout=1)
        self.notify = notify_callback

    def is_alive(self):
        try:
            self.ser.write(b'info\n')
            time.sleep(0.2)
            resp = self.ser.read_all().decode(errors='ignore')
            if resp.strip():
                self.notify(f"[ASCII] Alive check passed on {self.port}: {resp.strip().splitlines()[0]}")
                return True
        except Exception as e:
            self.notify(f"[ASCII] Alive check failed on {self.port}: {e}")
        return False

    def scan(self, start, stop, points):
        # Minimal scan command for testing
        self.ser.write(f'sweep {int(start)} {int(stop)} {points}\n'.encode())
        time.sleep(0.2)
        return [], []  # For brevity here
