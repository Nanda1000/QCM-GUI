# updated_nano_vna.py
import serial
import serial.tools.list_ports
import time
import numpy as np
import csv

class NanoVNA:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.Z0 = 50  # Characteristic impedance

    def connect(self, retries=3):
        for attempt in range(retries):
            try:
                if self.port is None:
                    ports = list(serial.tools.list_ports.comports())
                    if not ports:
                        raise Exception("No serial ports found.")
                    self.port = ports[0].device
                    print(f"Auto-detected port: {self.port}")
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                time.sleep(2)
                self.ser.reset_input_buffer()
                self._send_command("ver")
                response = self._read_line()
                if response:
                    print(f"Connected to NanoVNA: {response}")
                    return
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        raise RuntimeError("Unable to connect to NanoVNA after multiple attempts")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected from NanoVNA")

    def _send_command(self, cmd):
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")
        self.ser.write((cmd.strip() + "\r\n").encode())

    def _read_line(self):
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")
        try:
            return self.ser.readline().decode(errors='ignore').strip()
        except:
            return ''

    def _read_lines_until(self, end_marker='ch0', timeout=3):
        lines, start = [], time.time()
        while time.time() - start < timeout:
            line = self._read_line()
            if not line:
                continue
            if line == end_marker:
                break
            lines.append(line)
        return lines

    def calibrate_open_short_load(self):
        print("Calibrating...")
        for step in ["open", "short", "load"]:
            self._send_command(f"cal {step}")
            print(f"Connect {step.upper()} standard and press ENTER")
            input()
            time.sleep(1)
        self._send_command("cal done")
        print("Calibration complete.")

    def scan(self, start_freq, stop_freq, points):
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")

        start_khz = int(start_freq / 1e3)
        stop_khz = int(stop_freq / 1e3)

        self._send_command(f"frequencies {start_khz} {stop_khz} {points}")
        time.sleep(0.1)

        self._send_command("frequencies")
        freq_lines = self._read_lines_until('ch0')
        freqs = [float(line) * 1e3 for line in freq_lines if line.replace('.', '', 1).isdigit()]

        if len(freqs) != points:
            raise Exception("Frequency points mismatch")

        self._send_command("data 0")
        data_lines = self._read_lines_until('ch0')
        s11 = []
        for line in data_lines:
            try:
                real_str, imag_str = line.split(',')
                s11.append(complex(float(real_str), float(imag_str)))
            except:
                continue

        if len(s11) != len(freqs):
            raise Exception("S11 points mismatch")

        return np.array(freqs), np.array(s11)

    def s11_to_impedance(self, s11):
        s11 = np.asarray(s11)
        denom = 1 - s11
        denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
        return self.Z0 * (1 + s11) / denom

def acquire_data(device, start_freq=1e6, stop_freq=10e6, points=201):
    start_time = time.time()
    freqs, s11 = device.scan(start_freq, stop_freq, points)
    impedance = device.s11_to_impedance(s11)
    resistance = impedance.real
    reactance = impedance.imag
    phase = np.angle(impedance, deg=True)
    elapsed = time.time() - start_time
    time_array = np.linspace(0, elapsed, len(freqs))
    return freqs, resistance, impedance, reactance, phase, time_array

if __name__ == "__main__":
    vna = NanoVNA()
    try:
        vna.connect()
        vna.calibrate_open_short_load()
        with open("nanovna_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Frequency (Hz)", "Resistance (Ohms)", "Reactance (Ohms)", "Phase (deg)"])
            while True:
                freqs, R, Z, X, phase, t = acquire_data(vna, start_freq=1e6, stop_freq=10e6, points=201)
                timestamp = time.time()
                for f, r, x, p in zip(freqs, R, X, phase):
                    writer.writerow([timestamp, f, r, x, p])
                f.flush()
                time.sleep(2)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vna.disconnect()
