# Acquire_data.py
"""Robust NanoVNA interface for threaded acquisition.

- Does not call Qt widgets from worker threads.
- Accepts an optional notify_callback(level, title, message) callable
  so the GUI can display messages safely on the main thread.
"""
import serial
import serial.tools.list_ports
import time
import numpy as np
import threading
import queue
from datetime import datetime
from typing import Optional, Callable, Tuple, List

# Logging fallback if GUI doesn't provide notify_callback
import logging
logger = logging.getLogger("AcquireData")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class NanoVNA:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 timeout: float = 1.0, notify_callback: Optional[Callable] = None):
        """
        notify_callback(level, title, message) - optional callable from GUI to show info/warnings/critical.
        level: "info"|"warning"|"critical"
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.Z0 = 50.0
        self.is_connected = False
        self.acquisition_active = False
        self.data_queue = queue.Queue(maxsize=20)
        self.error_count = 0
        self.max_errors = 10
        self.last_successful_scan = None
        self.notify_callback = notify_callback
        self.acquisition_thread: Optional[threading.Thread] = None

    def _notify(self, level: str, title: str, msg: str):
        try:
            if callable(self.notify_callback):
                self.notify_callback(level, title, msg)
            else:
                if level == "info":
                    logger.info(f"{title}: {msg}")
                elif level == "warning":
                    logger.warning(f"{title}: {msg}")
                elif level == "critical":
                    logger.critical(f"{title}: {msg}")
                else:
                    logger.debug(f"{title}: {msg}")
        except Exception:
            logger.exception("Error while calling notify_callback")

    def _auto_detect_port(self) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self._notify("warning", "Auto-detect", "No serial ports found")
            return None

        keywords = ("nano", "vna", "ch340", "ch341", "usb-serial")
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in keywords):
                self._notify("info", "Auto-detect", f"Auto-detected port {p.device} ({p.description})")
                return p.device

        self._notify("info", "Auto-detect", f"No NanoVNA hint found, using first port {ports[0].device}")
        return ports[0].device

    def connect(self, retries: int = 3) -> bool:
        """Attempt a serial connection and verify device by sending 'info' or 'ver'."""
        if self.is_connected:
            return True

        if self.port is None:
            self.port = self._auto_detect_port()
            if self.port is None:
                self._notify("critical", "Connect", "No serial port available")
                return False

        for attempt in range(1, retries + 1):
            try:
                if self.ser and getattr(self.ser, "is_open", False):
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    time.sleep(0.2)

                self._notify("info", "Connect", f"Opening {self.port} @ {self.baudrate} (attempt {attempt})")
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout, write_timeout=self.timeout)
                time.sleep(0.5)

                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except Exception:
                    pass

                # ---- Try 'info' first ----
                self._send_command("info")
                time.sleep(0.05)
                lines = []
                while True:
                    line = self._read_line(timeout=0.5)
                    if not line:
                        break
                    lines.append(line)
                response = "\n".join(lines)

                # ---- If 'info' fails, try 'ver' ----
                if not response.strip():
                    self._send_command("ver")
                    time.sleep(0.05)
                    line = self._read_line(timeout=2.0)
                    if line:
                        response = line

                # ---- Accept if 'nano' appears anywhere ----
                if response and "nano" in response.lower():
                    self.is_connected = True
                    self.error_count = 0
                    self._notify("info", "Connect", f"Connected: {response.strip()}")
                    return True
                else:
                    self._notify("warning", "Connect", f"Unexpected response: {response}")
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    time.sleep(0.5)

            except Exception as e:
                self._notify("warning", "Connect", f"Attempt {attempt} failed: {e}")
                try:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                except Exception:
                    pass
                time.sleep(0.5)

        self.is_connected = False
        self._notify("critical", "Connect", "Unable to connect to NanoVNA after retries")
        return False

    def disconnect(self):
        try:
            self.stop_acquisition()
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.is_connected = False
            self._notify("info", "Disconnect", "Disconnected from NanoVNA")
        except Exception as e:
            self._notify("warning", "Disconnect", f"Error during disconnect: {e}")

    def _send_command(self, cmd: str):
        if not self.ser or not getattr(self.ser, "is_open", False):
            raise Exception("Serial device not open")
        line = (cmd.strip() + "\n").encode("utf-8")
        self.ser.write(line)
        self.ser.flush()

    def _read_line(self, timeout: Optional[float] = None) -> str:
        if not self.ser or not getattr(self.ser, "is_open", False):
            raise Exception("Serial device not open")
        orig = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            raw = self.ser.readline()
            if not raw:
                return ""
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
        finally:
            if timeout is not None:
                self.ser.timeout = orig

    def _read_lines_until(self, end_marker: str = "ch0", timeout: float = 5.0) -> List[str]:
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=0.5)
            if not line:
                continue
            if line.strip().lower() == end_marker.lower():
                break
            lines.append(line.strip())
        return lines

    def scan(self, start_freq: float, stop_freq: float, points: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_connected or not self.ser or not getattr(self.ser, "is_open", False):
            raise Exception("Device not connected")

        start_khz = int(start_freq / 1e3)
        stop_khz = int(stop_freq / 1e3)
        self._notify("info", "Scan", f"Starting scan {start_khz}kHz - {stop_khz}kHz, {points} pts")

        self._send_command(f"sweep {start_khz} {stop_khz} {points}")
        time.sleep(0.05)

        self._send_command("frequencies")
        freq_lines = self._read_lines_until(end_marker="ch0", timeout=8.0)
        freqs = []
        for line in freq_lines:
            try:
                value = float(line)
                if value > 1e6:
                    freqs.append(value)
                else:
                    freqs.append(value * 1e3)
            except ValueError:
                continue
        if not freqs:
            raise Exception("No frequencies received from device")

        self._send_command("data 0")
        data_lines = self._read_lines_until(end_marker="ch0", timeout=8.0)
        s11 = []
        for line in data_lines:
            if not line:
                continue
            if "," in line:
                try:
                    real_s, imag_s = line.split(",", 1)
                    s11.append(complex(float(real_s.strip()), float(imag_s.strip())))
                except ValueError:
                    continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        s11.append(complex(float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        if not s11:
            raise Exception("No S11 data received from device")

        min_len = min(len(freqs), len(s11))
        freqs_arr = np.array(freqs[:min_len], dtype=float)
        s11_arr = np.array(s11[:min_len], dtype=complex)

        self.last_successful_scan = time.time()
        self.error_count = 0
        self._notify("info", "Scan", f"Scan completed: {len(freqs_arr)} points")
        return freqs_arr, s11_arr

    def s11_to_impedance(self, s11: np.ndarray) -> np.ndarray:
        s11 = np.asarray(s11, dtype=complex)
        denom = 1.0 - s11
        small = np.abs(denom) < 1e-15
        denom[small] = 1e-15 + 0j
        Z = self.Z0 * (1.0 + s11) / denom
        Z = np.where(np.abs(Z) > 1e6, 1e6 + 0j, Z)
        Z = np.where(np.abs(Z) < 1e-12, 1e-12 + 0j, Z)
        return Z

    def start_acquisition(self, start_freq: float, stop_freq: float, points: int,
                          interval: float = 2.0, callback: Optional[Callable] = None) -> bool:
        if self.acquisition_active:
            self._notify("warning", "Acquisition", "Acquisition already active")
            return False
        if not self.is_connected:
            self._notify("critical", "Acquisition", "Device not connected")
            return False

        self.acquisition_active = True

        def worker():
            scan_count = 0
            while self.acquisition_active:
                start_time = time.time()
                try:
                    freqs, s11 = self.scan(start_freq, stop_freq, points)
                    impedance = self.s11_to_impedance(s11)
                    resistance = impedance.real
                    reactance = impedance.imag
                    phase = np.angle(impedance, deg=True)
                    magnitude = np.abs(impedance)
                    scan_count += 1
                    elapsed = time.time() - start_time

                    data_package = {
                        "timestamp": datetime.now(),
                        "scan_count": scan_count,
                        "frequencies": freqs,
                        "impedance": impedance,
                        "resistance": resistance,
                        "reactance": reactance,
                        "phase": phase,
                        "magnitude": magnitude,
                        "s11": s11,
                        "acquisition_time": elapsed
                    }

                    try:
                        self.data_queue.put_nowait(data_package)
                    except queue.Full:
                        try:
                            _ = self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.data_queue.put_nowait(data_package)
                        except queue.Full:
                            pass

                    if callable(callback):
                        try:
                            callback(data_package)
                        except Exception:
                            self._notify("warning", "Callback", "Callback raised an exception")

                    remaining = max(0.0, interval - (time.time() - start_time))
                    if remaining > 0:
                        time.sleep(remaining)

                except Exception as e:
                    self.error_count += 1
                    self._notify("warning", "Acquisition", f"Scan error #{self.error_count}: {e}")
                    if self.error_count >= self.max_errors:
                        self._notify("critical", "Acquisition", "Max errors reached, stopping acquisition")
                        break
                    time.sleep(min(1.0, interval))

            self.acquisition_active = False
            self._notify("info", "Acquisition", f"Acquisition stopped after {scan_count} scans")

        self.acquisition_thread = threading.Thread(target=worker, daemon=True)
        self.acquisition_thread.start()
        self._notify("info", "Acquisition", "Acquisition thread started")
        return True

    def stop_acquisition(self):
        if self.acquisition_active:
            self._notify("info", "Acquisition", "Stopping acquisition")
            self.acquisition_active = False
            if self.acquisition_thread and self.acquisition_thread.is_alive():
                self.acquisition_thread.join(timeout=2.0)
            self._notify("info", "Acquisition", "Acquisition stopped")


def acquire_data(device_or_port, start_freq=1e6, stop_freq=10e6, points=201):
    if isinstance(device_or_port, NanoVNA):
        vna = device_or_port
    else:
        vna = NanoVNA(port=device_or_port)
        if not vna.connect():
            raise RuntimeError("Could not connect to NanoVNA")
    freqs, s11 = vna.scan(start_freq, stop_freq, points)
    impedance = vna.s11_to_impedance(s11)
    resistance = impedance.real
    reactance = impedance.imag
    phase = np.angle(impedance, deg=True)
    magnitude = np.abs(impedance)
    time_array = np.linspace(0, len(freqs) - 1, len(freqs))
    return freqs, resistance, impedance, reactance, phase, time_array
