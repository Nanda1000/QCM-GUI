import serial
import serial.tools.list_ports
import time
import numpy as np
from datetime import datetime

class NanoVNA:
    def __init__(self, port=None, baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False

    def _auto_detect_port(self):
        ports = list(serial.tools.list_ports.comports())
        keywords = ("nano", "vna", "ch340", "ch341", "usb-serial")
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in keywords):
                print(f"Auto-detected port {p.device} ({p.description})")
                return p.device
        if ports:
            print(f"No NanoVNA hint found, using first port {ports[0].device}")
            return ports[0].device
        return None

    def connect(self, retries=3):
        if self.is_connected:
            return True

        if self.port is None:
            self.port = self._auto_detect_port()
            if self.port is None:
                print("No serial port available")
                return False

        for attempt in range(1, retries + 1):
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    time.sleep(0.2)

                print(f"Opening {self.port} @ {self.baudrate} (attempt {attempt})")
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.timeout)
                time.sleep(0.5)

                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()

                self._send_command("info")
                time.sleep(0.05)
                response = self._read_response(1.0)
                if "nano" in response.lower():
                    self.is_connected = True
                    print(f"Connected: {response.strip()}")
                    return True
                else:
                    print(f"Unexpected response: {response}")
                    self.ser.close()
                    time.sleep(0.5)

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if self.ser and self.ser.is_open:
                    self.ser.close()
                time.sleep(0.5)

        print("Unable to connect to NanoVNA after retries")
        return False

    def _send_command(self, cmd):
        if not self.ser or not self.ser.is_open:
            raise Exception("Serial device not open")
        line = (cmd.strip() + "\n").encode("utf-8")
        self.ser.write(line)
        self.ser.flush()

    def _read_line(self, timeout=None):
        if not self.ser or not self.ser.is_open:
            raise Exception("Serial device not open")
        orig = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            raw = self.ser.readline()
            if not raw:
                return ""
            return raw.decode("utf-8", errors="ignore").strip()
        finally:
            if timeout is not None:
                self.ser.timeout = orig

    def _read_response(self, duration=1.0):
        lines = []
        start = time.time()
        while time.time() - start < duration:
            line = self._read_line(timeout=0.1)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def sweep(self, start_freq, stop_freq, points):
        start_khz = int(start_freq / 1e3)
        stop_khz = int(stop_freq / 1e3)
        self._send_command(f"sweep {start_khz} {stop_khz} {points}")
        time.sleep(0.05)

    def get_frequencies(self):
        self._send_command("frequencies")
        freqs = []
        while True:
            line = self._read_line(timeout=0.5)
            if line == "ch>" or line == "":
                break
            try:
                f = float(line)
                if f < 1e6:  # if in kHz
                    f *= 1e3
                freqs.append(f)
            except:
                pass
        return np.array(freqs)

    def get_data0(self):
        self._send_command("data 0")
        s11 = []
        while True:
            line = self._read_line(timeout=0.5)
            if line == "ch>" or line == "":
                break
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    real = float(parts[0])
                    imag = float(parts[1])
                    s11.append(complex(real, imag))
                except:
                    pass
        return np.array(s11)

    def s11_to_impedance(self, s11):
        Z0 = 50.0
        denom = 1.0 - s11
        denom = np.where(np.abs(denom) < 1e-15, 1e-15 + 0j, denom)
        Z = Z0 * (1.0 + s11) / denom
        Z = np.where(np.abs(Z) > 1e6, 1e6 + 0j, Z)
        Z = np.where(np.abs(Z) < 1e-12, 1e-12 + 0j, Z)
        return Z


def main():
    vna = NanoVNA()
    if not vna.connect():
        print("Connection failed")
        return

    start_freq = 50e3  # 50 kHz
    stop_freq = 900e6  # 900 MHz
    points = 101

    try:
        while True:
            vna.sweep(start_freq, stop_freq, points)
            time.sleep(0.1)  # give device time to sweep

            freqs = vna.get_frequencies()
            s11 = vna.get_data0()
            if len(freqs) != len(s11):
                print("Warning: freq and data length mismatch")

            Z = vna.s11_to_impedance(s11)
            resistance = Z.real
            reactance = Z.imag
            magnitude = np.abs(Z)
            phase = np.angle(Z, deg=True)

            print(f"\nScan at {datetime.now()}: {len(freqs)} points")
            for f, R, X, mag, ph in zip(freqs, resistance, reactance, magnitude, phase):
                print(f"{f/1e6:7.3f} MHz | R={R:7.2f} Ω | X={X:7.2f} Ω | |Z|={mag:7.2f} Ω | Φ={ph:7.2f}°")

            time.sleep(2)  # pause between scans, adjust as needed

    except KeyboardInterrupt:
        print("Acquisition stopped by user")
    finally:
        if vna.ser and vna.ser.is_open:
            vna.ser.close()


if __name__ == "__main__":
    main()
