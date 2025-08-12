"""
Acquire_unified.py

Unified NanoVNA connection & acquisition wrapper with device detection.

- Probes serial ports and uses simple firmware interrogation to classify device families.
- Attempts to instantiate a model-specific driver (if present in nanovna.py).
- Falls back to ASCII serial driver from Acquire_data.py if no model-specific driver is available.
- Keeps the simple API your GUI expects.

Usage:
    from Acquire_unified import UnifiedNanoVNA as NanoVNA
    vna = NanoVNA(port=None)   # port None => auto-detect
    vna.connect()
    freqs, s11 = vna.scan(1e6, 10e6, 201)
"""
from __future__ import annotations
from typing import Optional, Callable, Any, Dict, Tuple
import logging
import time
import serial
from serial.tools import list_ports
import threading
import queue
from datetime import datetime
import numpy as np

logger = logging.getLogger("AcquireUnified")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# Try to import model classes from nanovna module (if present)
_try_nv_mod = None
try:
    import nanovna as _try_nv_mod  # your nanovna.py (may define many classes)
except Exception:
    _try_nv_mod = None

# Try to import the ASCII serial NanoVNA you already have
try:
    from Acquire_data import NanoVNA as ASCII_NanoVNA  # your existing ASCII implementation
except Exception:
    ASCII_NanoVNA = None

# -------------------------------------------------------------------
# Known USB VID/PID tuples and friendly name table (common devices)
# (these are used as hints only; final determination uses firmware text)
# -------------------------------------------------------------------
USB_KNOWN = {
    (0x0483, 0x5740): "NanoVNA",
    (0x16C0, 0x0483): "AVNA",
    (0x04B4, 0x0008): "S-A-A-2",  # often reported by V2 devices
    # add more tuples if you know VID/PID for other boards
}

# The canonical mapping from textual detection result -> driver class name
# We'll try to resolve these names to actual classes inside nanovna module or fallback.
NAME2DEVICE_CANDIDATES = {
    "S-A-A-2": ["NanoVNA_V2", "NanoVNAv2", "NanoVNA_V2", "NanoVNA_V2_driver"],
    "AVNA": ["AVNA"],
    "H4": ["NanoVNA_H4", "NanoVNA_H4_driver"],
    "H": ["NanoVNA_H"],
    "F_V2": ["NanoVNA_F_V2", "NanoVNA_F_V2_driver"],
    "F_V3": ["NanoVNA_F_V3"],
    "F": ["NanoVNA_F"],
    "NanoVNA": ["NanoVNA", "NanoVNA_class"],
    "tinySA": ["TinySA"],
    "tinySA_Ultra": ["TinySA_Ultra", "TinySAUltra"],
    "JNCRadio": ["JNCRadio_VNA_3G", "JNCRadio"],
    "SV4401A": ["SV4401A"],
    "SV6301A": ["SV6301A"],
    "LiteVNA64": ["LiteVNA64"],
    "Unknown": []
}

# -------------------------------------------------------------------
# Low-level helpers for probing serial devices
# -------------------------------------------------------------------
def _safe_open_port(device: str, baudrate: int = 115200, timeout: float = 0.2):
    """Open a serial.Serial or return None on failure."""
    try:
        ser = serial.Serial(device, baudrate=baudrate, timeout=timeout, write_timeout=timeout)
        # small delay to let device settle
        time.sleep(0.05)
        # flush to start from clean state
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        return ser
    except Exception as e:
        logger.debug("Open port %s failed: %s", device, e)
        return None

def _probe_send_cr(ser: serial.Serial) -> str:
    """Send carriage return(s) and read a chunk of initial data as string."""
    try:
        # drain a tiny bit then send CR
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        ser.write(b"\r")
        time.sleep(0.05)
        ser.write(b"\r")
        time.sleep(0.05)
        # read available bytes (non-blocking due to ser.timeout)
        data = ser.read(512)
        try:
            return data.decode("ascii", errors="ignore")
        except Exception:
            return ""
    except Exception as e:
        logger.debug("Probe CR failed: %s", e)
        return ""

def _get_firmware_info(ser: serial.Serial, retries: int = 3, line_wait: float = 0.05) -> str:
    """Send 'info' and accumulate response lines until prompt or timeout."""
    out_lines = []
    try:
        # flush first
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        for _ in range(retries):
            try:
                ser.write(b"info\r")
            except Exception:
                try:
                    # try ascii string encoding fallback
                    ser.write("info\r".encode("ascii"))
                except Exception:
                    return ""
            time.sleep(line_wait)
            # read available lines up to a limit
            for _ in range(64):
                data = ser.readline()
                if not data:
                    break
                try:
                    s = data.decode("ascii", errors="ignore").strip()
                except Exception:
                    s = ""
                if not s:
                    continue
                # stop if we reached a shell prompt or 'ch>' style echo
                if s.startswith("ch>") or s.startswith(">") or s.endswith("ch>"):
                    break
                # skip echoes
                if s.lower() == "info":
                    continue
                out_lines.append(s)
            if out_lines:
                break
        return "\n".join(out_lines).strip()
    except Exception as e:
        logger.debug("get_info failed: %s", e)
        return ""

def _is_lite_vna_64(ser: serial.Serial) -> bool:
    """Best-effort check for LiteVNA64; many implementations expose a known banner."""
    info = _get_firmware_info(ser, retries=1)
    if not info:
        return False
    # a heuristic: LiteVNA64 often contains 'LiteVNA' or 'Lite' in info text
    return ("lite" in info.lower() or "litevna" in info.lower())

def _class_for_name(name: str):
    """Try to resolve a device-name key into an actual class from nanovna module or fallback to ASCII driver."""
    if _try_nv_mod:
        # check candidate class names
        candidates = NAME2DEVICE_CANDIDATES.get(name, [])
        for cname in candidates:
            cls = getattr(_try_nv_mod, cname, None)
            if cls is not None:
                return cls
        # also try direct name match
        cls = getattr(_try_nv_mod, name, None)
        if cls:
            return cls
    # fallback: return None (caller will decide to use ASCII)
    return None

# -------------------------------------------------------------------
# High-level detection flow
# -------------------------------------------------------------------
def scan_serial_ports_for_vna(verbose: bool = False):
    """Scan available serial ports and return a list of (port, hint_name, hwinfo_str)."""
    results = []
    ports = list_ports.comports()
    for p in ports:
        dev = p.device
        vid = getattr(p, "vid", None)
        pid = getattr(p, "pid", None)
        hint = USB_KNOWN.get((vid, pid), "")
        ser = _safe_open_port(dev)
        if ser is None:
            continue
        try:
            banner = _probe_send_cr(ser)
            # if banner indicates v2 lite numeric start (some firmwares reply with '2...' or ascii banner)
            fw_info = _get_firmware_info(ser)
            # try lite detection if needed
            if not hint and _is_lite_vna_64(ser):
                hint = "LiteVNA64"
            results.append((dev, hint, banner, fw_info))
        finally:
            try:
                ser.close()
            except Exception:
                pass
    return results

def classify_device_from_probe(banner: str, fw_info: str, port_hint: str = "") -> str:
    """
    From the banner (response to CR) and firmware info ('info' response),
    return one of the NAME2DEVICE_CANDIDATES keys or 'Unknown'.
    """
    b = (banner or "").lower()
    info = (fw_info or "").lower()

    # quick banner-based heuristics
    if b.startswith("ch>") or b.startswith("\r\nch>") or b.startswith("?"):
        # classic NanoVNA v1 / H variants
        # if firmware info mentions 'nanoVNA-H 4' etc, use that
        if "nanoVna-h 4".lower() in info or "nanoVNA-H 4".lower() in info:
            return "H4"
        if "nanovna-h".lower() in info:
            return "H"
        return "NanoVNA"

    # some V2 devices respond with '2' or numeric headers
    if b and b[0].isdigit():
        # use fw_info to decide v2 vs lite
        if "lite" in info or "litevna" in info:
            return "LiteVNA64"
        return "S-A-A-2"

    # scan firmware info text for keywords
    for key, candidates in (
        ("avna", ("avna",)),
        ("nanovna-h 4", ("h4",)),
        ("nanovna-h", ("h",)),
        ("nanovna-f_v2", ("f_v2", "f v2")),
        ("nanovna-f_v3", ("f_v3", "f v3")),
        ("nanovna-f", ("f ",)),
        ("nanovna", ("nanovna",)),
        ("tinySA4", ("tiny", "tinysa")),
        ("jncradio", ("jncradio",)),
        ("sv4401a", ("sv4401a",)),
        ("sv6301a", ("sv6301a",)),
        ("litevna", ("lite",)),
    ):
        if key in info:
            # map certain textual keywords to canonical keys
            if "avna" in key:
                return "AVNA"
            if "lite" in key or "litevna" in key:
                return "LiteVNA64"
            if "jncradio" in key:
                return "JNCRadio"
            if "sv4401a" in key:
                return "SV4401A"
            if "sv6301a" in key:
                return "SV6301A"
            if "tinysa" in key or "tiny" in key:
                return "tinySA"
            if "nanovna-h 4" in key:
                return "H4"
            if "nanovna-h" in key:
                return "H"
            if "nanovna-f_v2" in key:
                return "F_V2"
            if "nanovna-f_v3" in key:
                return "F_V3"
            if "nanovna-f" in key:
                return "F"
            if "nanovna" in key:
                return "NanoVNA"
    # fallback to supplied port hint
    if port_hint:
        return port_hint
    return "Unknown"

# -------------------------------------------------------------------
# Unified adapter class (exposed to GUI)
# -------------------------------------------------------------------
class UnifiedNanoVNA:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 timeout: float = 0.2, notify_callback: Optional[Callable] = None,
                 prefer_ascii_first: bool = True):
        """
        port: explicit serial port (COMx or /dev/ttyX) or None to auto-scan
        prefer_ascii_first: if True, try ASCII serial driver before binary model classes
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.notify_callback = notify_callback
        self.prefer_ascii_first = prefer_ascii_first

        self._impl = None
        self.is_connected = False
        self.acquisition_active = False
        self.data_queue = queue.Queue(maxsize=20)
        self._acq_thread = None

    def _notify(self, level: str, title: str, msg: str):
        try:
            if callable(self.notify_callback):
                self.notify_callback(level, title, msg)
            else:
                if level == "info":
                    logger.info("%s: %s", title, msg)
                elif level == "warning":
                    logger.warning("%s: %s", title, msg)
                else:
                    logger.error("%s: %s", title, msg)
        except Exception:
            logger.exception("notify error")

    def connect(self, retries: int = 2) -> bool:
        """
        Connect to device. If self.port is None, auto-scan available serial ports.
        Returns True on success.
        """
        # If already connected
        if self.is_connected:
            return True

        # If explicit port is given, just probe it
        ports_to_try = []
        if self.port:
            ports_to_try = [self.port]
        else:
            ports_to_try = [p.device for p in list_ports.comports()]

        # Optionally attempt ASCII serial driver first (works for many H/H4/F/tinySA)
        if self.prefer_ascii_first and ASCII_NanoVNA:
            for p in ports_to_try:
                try:
                    impl = ASCII_NanoVNA(port=p, baudrate=self.baudrate, timeout=self.timeout,
                                        notify_callback=self.notify_callback)
                    ok = impl.connect(retries=1)
                    if ok:
                        self._impl = impl
                        self.is_connected = True
                        self._notify("info", "Connect", f"Connected via ASCII driver on {p}")
                        return True
                except Exception:
                    continue


        # Probe each serial port and attempt to classify + instantiate a model-specific class
        probes = scan_serial_ports_for_vna()
        # If explicit port was set, filter to that
        if self.port:
            probes = [pr for pr in probes if pr[0] == self.port]

        for dev, port_hint, banner, fw_info in probes:
            model_name = classify_device_from_probe(banner, fw_info, port_hint)
            self._notify("info", "Probe", f"{dev} -> model hint '{model_name}'")
            # try to resolve a class for this model_name
            cls = _class_for_name(model_name)
            # if class found, try to instantiate and connect/prepare
            if cls is not None:
                try:
                    # many binary/scikit-rf classes accept an address or pyvisa resource string;
                    # we will try a simple constructor patterns
                    inst = None
                    try:
                        inst = cls(dev)  # try common pattern
                    except Exception:
                        try:
                            inst = cls(address=dev)
                        except Exception:
                            try:
                                inst = cls()  # last resort
                            except Exception:
                                inst = None
                    if inst is None:
                        raise RuntimeError("Could not instantiate driver class")
                    # try connect if present
                    connect_fn = getattr(inst, "connect", None)
                    ok = True
                    if callable(connect_fn):
                        ok = connect_fn()
                    if ok:
                        self._impl = inst
                        self.is_connected = True
                        self._notify("info", "Connect", f"Connected to {model_name} on {dev} using {cls.__name__}")
                        return True
                except Exception as e:
                    self._notify("warning", "Driver", f"Driver {cls.__name__} failed for {dev}: {e}")
            else:
                # No model-specific class found; try ASCII fallback on that port
                if ASCII_NanoVNA:
                    try:
                        impl = ASCII_NanoVNA(port=dev, baudrate=self.baudrate, timeout=self.timeout,
                                             notify_callback=self.notify_callback)
                        ok = impl.connect(retries=1)
                        if ok:
                            self._impl = impl
                            self.is_connected = True
                            self._notify("info", "Connect", f"Connected via ASCII fallback on {dev}")
                            return True
                    except Exception as e:
                        logger.debug("ASCII fallback failed on %s: %s", dev, e)
                        continue

        # Final attempt: try binary driver classes without port hints (let driver try pyvisa resources)
        if _try_nv_mod:
            for key in NAME2DEVICE_CANDIDATES:
                cls = _class_for_name(key)
                if cls is None:
                    continue
                try:
                    # try construct with no args
                    inst = None
                    try:
                        inst = cls()
                    except Exception:
                        try:
                            inst = cls(None)
                        except Exception:
                            inst = None
                    if inst is None:
                        continue
                    connect_fn = getattr(inst, "connect", None)
                    ok = True
                    if callable(connect_fn):
                        ok = connect_fn()
                    if ok:
                        self._impl = inst
                        self.is_connected = True
                        self._notify("info", "Connect", f"Connected using driver {cls.__name__}")
                        return True
                except Exception:
                    continue

        self._notify("critical", "Connect", "Unable to connect to any NanoVNA on scanned ports")
        return False

    def disconnect(self):
        try:
            if self._impl:
                # try to stop acquisition and disconnect if available
                try:
                    fn = getattr(self._impl, "stop_acquisition", None)
                    if callable(fn):
                        fn()
                except Exception:
                    pass
                try:
                    fn = getattr(self._impl, "disconnect", None)
                    if callable(fn):
                        fn()
                except Exception:
                    pass
            self.is_connected = False
            self._notify("info", "Disconnect", "Disconnected")
        except Exception as e:
            self._notify("warning", "Disconnect", str(e))

    def scan(self, start_freq: float, stop_freq: float, points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single scan and return (freqs_Hz, s11_complex_array).
        Will delegate to underlying driver when possible.
        """
        if not self.is_connected or self._impl is None:
            raise RuntimeError("Device not connected")

        # Preferred unified interface: a scan(start, stop, points) that returns freqs,s11
        scan_fn = getattr(self._impl, "scan", None)
        if callable(scan_fn):
            freqs, s11 = scan_fn(start_freq, stop_freq, points)
            return np.asarray(freqs, dtype=float), np.asarray(s11, dtype=complex)

        # scikit-rf-style: get_s11_s21 or get_sdata
        get_s11_s21 = getattr(self._impl, "get_s11_s21", None) or getattr(self._impl, "get_sdata", None)
        if callable(get_s11_s21):
            try:
                out = get_s11_s21()
                # out may be a skrf.Network or tuple of networks
                # prefer the s11 network
                s11_net = out if hasattr(out, "s") else (out[0] if isinstance(out, tuple) else None)
                if s11_net is None and isinstance(out, tuple) and len(out) >= 1:
                    s11_net = out[0]
                if s11_net is not None and hasattr(s11_net, "s") and hasattr(s11_net, "frequency"):
                    s11_arr = s11_net.s[:, 0, 0]
                    freqs = getattr(s11_net.frequency, "f", None) or getattr(s11_net.frequency, "hz", None)
                    return np.asarray(freqs, dtype=float), np.asarray(s11_arr, dtype=complex)
            except Exception as e:
                raise RuntimeError(f"Driver returned invalid network: {e}")

        raise RuntimeError("Underlying driver does not expose a recognized scan interface")

    def s11_to_impedance(self, s11_array):
        fn = getattr(self._impl, "s11_to_impedance", None)
        if callable(fn):
            return fn(s11_array)
        # default conversion
        s11 = np.asarray(s11_array, dtype=complex)
        Z0 = 50.0
        denom = 1.0 - s11
        small = np.abs(denom) < 1e-15
        denom[small] = 1e-15 + 0j
        Z = Z0 * (1.0 + s11) / denom
        Z = np.where(np.abs(Z) > 1e6, 1e6 + 0j, Z)
        Z = np.where(np.abs(Z) < 1e-12, 1e-12 + 0j, Z)
        return Z

    def start_acquisition(self, start_freq: float, stop_freq: float, points: int,
                          interval: float = 2.0, callback: Optional[Callable] = None) -> bool:
        """
        Start continuous acquisition. If underlying driver supports threaded acquisition it will be used,
        otherwise a fallback thread will poll scan(...) at the requested interval and call callback(data_package).
        """
        if not self.is_connected or self._impl is None:
            self._notify("warning", "Acquisition", "Device not connected")
            return False
        if self.acquisition_active:
            self._notify("warning", "Acquisition", "Already active")
            return False

        # Delegate to driver if method exists
        start_fn = getattr(self._impl, "start_acquisition", None)
        if callable(start_fn):
            try:
                ok = start_fn(start_freq, stop_freq, points, interval=interval, callback=callback)
                if ok:
                    self.acquisition_active = True
                    return True
            except Exception as e:
                self._notify("warning", "Acquisition", f"Driver start_acquisition failed: {e}")

        # Fallback: create thread that periodically calls scan(...)
        self.acquisition_active = True

        def worker():
            scan_count = 0
            while self.acquisition_active:
                t0 = time.time()
                try:
                    freqs, s11 = self.scan(start_freq, stop_freq, points)
                    impedance = self.s11_to_impedance(s11)
                    resistance = impedance.real
                    reactance = impedance.imag
                    phase = np.angle(impedance, deg=True)
                    magnitude = np.abs(impedance)
                    scan_count += 1
                    elapsed = time.time() - t0
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
                    # non-blocking queue push
                    try:
                        self.data_queue.put_nowait(data_package)
                    except queue.Full:
                        try:
                            _ = self.data_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.data_queue.put_nowait(data_package)
                        except Exception:
                            pass
                    if callable(callback):
                        try:
                            callback(data_package)
                        except Exception:
                            logger.exception("callback error")
                    remaining = max(0.0, interval - (time.time() - t0))
                    if remaining > 0:
                        time.sleep(remaining)
                except Exception as e:
                    logger.warning("acquisition scan error: %s", e)
                    time.sleep(min(1.0, interval))
            self.acquisition_active = False
            self._notify("info", "Acquisition", f"Acquisition thread stopped after {scan_count} scans")

        self._acq_thread = threading.Thread(target=worker, daemon=True)
        self._acq_thread.start()
        self._notify("info", "Acquisition", "Acquisition thread started (fallback)")
        return True

    def stop_acquisition(self):
        if not self.acquisition_active:
            return
        # delegate first
        try:
            fn = getattr(self._impl, "stop_acquisition", None)
            if callable(fn):
                fn()
        except Exception:
            pass
        self.acquisition_active = False
        # join thread briefly
        if self._acq_thread and self._acq_thread.is_alive():
            self._acq_thread.join(timeout=1.0)
        self._notify("info", "Acquisition", "Acquisition stop requested")

# alias
Unified = UnifiedNanoVNA

# if this file is run as a script, list detected ports and probes
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Scanning serial ports for VNAs...")
    probes = scan_serial_ports_for_vna(verbose=True)
    for dev, hint, banner, fwinfo in probes:
        print(f"{dev} -> hint='{hint}' banner='{banner[:120]}'")
        print("info:\n", fwinfo)
