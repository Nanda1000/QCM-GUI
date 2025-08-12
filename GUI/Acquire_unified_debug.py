# Acquire_unified_debug.py
import serial.tools.list_ports
from Acquire_data_debug import NanoVNA_ASCII_Debug
from nanovna_debug import NanoVNAv2_Debug

class UnifiedNanoVNA_Debug:
    def __init__(self, port=None, notify_callback=print):
        self.port = port
        self.notify = notify_callback
        self.driver = None

    def list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        ports = [self.port] if self.port else self.list_ports()
        for p in ports:
            self.notify(f"Trying {p} with ASCII protocol...")
            try:
                ascii_driver = NanoVNA_ASCII_Debug(p, self.notify)
                if ascii_driver.is_alive():
                    self.driver = ascii_driver
                    self.notify(f"Connected to {p} via ASCII protocol")
                    return True
            except Exception as e:
                self.notify(f"ASCII connection error on {p}: {e}")

            self.notify(f"Trying {p} with Binary protocol...")
            try:
                binary_driver = NanoVNAv2_Debug(p, self.notify)
                if binary_driver.is_alive():
                    self.driver = binary_driver
                    self.notify(f"Connected to {p} via Binary protocol")
                    return True
            except Exception as e:
                self.notify(f"Binary connection error on {p}: {e}")

        self.notify("No NanoVNA found on any port")
        return False

    def scan(self, start, stop, points):
        if self.driver:
            return self.driver.scan(start, stop, points)
        else:
            raise RuntimeError("Not connected")
