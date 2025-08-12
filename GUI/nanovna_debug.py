# nanovna_debug.py
import serial
import time

class NanoVNAv2_Debug:
    def __init__(self, port, notify_callback=print):
        self.port = port
        self.ser = serial.Serial(port, baudrate=115200, timeout=1)
        self.notify = notify_callback

    def is_alive(self):
        try:
            self.ser.write(b'capture\n')  # Example binary/alt command
            time.sleep(0.2)
            resp = self.ser.read(16)  # read some bytes
            if resp:
                self.notify(f"[Binary] Alive check passed on {self.port}, got {len(resp)} bytes")
                return True
        except Exception as e:
            self.notify(f"[Binary] Alive check failed on {self.port}: {e}")
        return False
