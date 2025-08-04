import serial
from datetime import datetime
import time
import numpy as np
import serial.tools.list_ports

class NanoVNA:
    def __init__(self, port=None, baud=115200, timeout=1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self.Z0 = 50  # Characteristic impedance in Ohms

    def connect(self):
        if self.port is None:
            available_ports = list(serial.tools.list_ports.comports())
            if not available_ports:
                raise Exception("No serial ports found. Please connect the NanoVNA.")
            
            # Attempt to connect to the first available port
            self.port = available_ports[0].device
            print(f"Attempting to connect to auto-detected port: {self.port}")

        try:
            self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.timeout)
            time.sleep(2)  # Allow device to initialize
            self.ser.reset_input_buffer()
            self.ser.write(b'ver\r\n')
            response = self.ser.readline().decode(errors='ignore').strip()
            if not response:
                raise Exception("No response from NanoVNA")
            print("Connected:", response)
        except Exception as e:
            raise Exception(f"Connection failed: {e}")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def scan(self, start_freq, stop_freq, points):
        """
        Send scan command to NanoVNA and read S11 data.
        start_freq, stop_freq in Hz
        points: number of data points

        Returns:
            freqs: list of frequencies in Hz
            s11_data: list of complex S11 values
        """
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")

        # NanoVNA firmware expects frequencies in kHz usually
        start_khz = int(start_freq / 1e3)
        stop_khz = int(stop_freq / 1e3)

        cmd = f"scan {start_khz} {stop_khz} {points}\r\n"
        self.ser.write(cmd.encode())

        freqs = []
        s11_data = []
        timeout_counter = 0
        max_timeout = 100  # ~5 seconds timeout with sleep 0.05

        while len(freqs) < points and timeout_counter < max_timeout:
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line or ',' not in line:
                timeout_counter += 1
                time.sleep(0.05)
                continue
            try:
                parts = line.split(',')
                if len(parts) < 3:
                    timeout_counter += 1
                    continue
                freq_khz = float(parts[0])
                real = float(parts[1])
                imag = float(parts[2])
                freqs.append(freq_khz * 1e3)  # Convert kHz back to Hz
                s11_data.append(complex(real, imag))
                timeout_counter = 0  # reset timeout on success
            except Exception:
                timeout_counter += 1
                time.sleep(0.05)
                continue

        if len(freqs) < points:
            raise Exception(f"Incomplete data received: expected {points} got {len(freqs)}")

        return freqs, s11_data

def acquire_data(device_path, start_freq=1e6, stop_freq=10e6, points=201):
    """
    Connects to NanoVNA, performs scan, and returns frequencies and resistances.

    Arguments:
        device_path: serial port path
        start_freq, stop_freq: in Hz
        points: number of points to acquire

    Returns:
        freqs: list of frequencies (Hz)
        resistance: list of real resistance values (Ohms)
        impedance: list of complex impedance values
        reactance: list of imaginary reactance values (Ohms)
        phase: list of phase values (degrees)
        time_array: list of time values (seconds)
    """
    device = NanoVNA(device_path)
    device.connect()
    
    start_time = datetime.now()

    try:
        freqs, s11 = device.scan(start_freq, stop_freq, points)
    finally:
        device.disconnect()
        
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    time_array = np.linspace(0, elapsed_time, len(freqs))

    Z0 = 50
    impedance = []
    resistance = []
    reactance = []
    phase = []

    for s in s11:
        denom = 1 - s
        if abs(denom) < 1e-12:
            # Avoid division by zero
            z = complex(np.inf, np.inf)
            phase_val = 0  # Default phase when impedance is infinite
        else:
            z = Z0 * (1 + s) / denom
            phase_val = np.angle(z, deg=True)
        impedance.append(z)
        resistance.append(z.real)
        reactance.append(z.imag)
        phase.append(phase_val)

    return freqs, resistance, impedance, reactance, phase, time_array
