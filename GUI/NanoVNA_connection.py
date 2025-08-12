# NanoVNA Connection and Control Module
# Custom implementation for GUI integration

import logging
import platform
import threading
import time
from collections import namedtuple
from typing import Optional, Callable, Dict, List, Tuple
from threading import RLock, Event
import serial
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo
import numpy as np

logger = logging.getLogger(__name__)

# Device definitions
USBDevice = namedtuple("USBDevice", "vid pid name")

KNOWN_DEVICES = (
    USBDevice(0x0483, 0x5740, "NanoVNA"),
    USBDevice(0x16C0, 0x0483, "AVNA"),
    USBDevice(0x04B4, 0x0008, "NanoVNA-V2"),
)

class SerialInterface:
    """Custom serial interface for NanoVNA communication"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.timeout = 0.2
        self.serial_port: Optional[serial.Serial] = None
        self.lock = RLock()
        self.is_open = False
        self.device_type = "Unknown"
        
    def open(self) -> bool:
        """Open serial connection"""
        try:
            with self.lock:
                if self.is_open:
                    return True
                    
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                self.is_open = True
                logger.info(f"Opened connection to {self.port}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to open {self.port}: {e}")
            self.is_open = False
            return False
    
    def close(self):
        """Close serial connection"""
        try:
            with self.lock:
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()
                self.is_open = False
                logger.info(f"Closed connection to {self.port}")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def write(self, data: str) -> bool:
        """Write data to device"""
        try:
            with self.lock:
                if not self.is_open or not self.serial_port:
                    return False
                self.serial_port.write(data.encode('ascii'))
                return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False
    
    def readline(self) -> str:
        """Read line from device"""
        try:
            with self.lock:
                if not self.is_open or not self.serial_port:
                    return ""
                return self.serial_port.readline().decode('ascii', errors='ignore').strip()
        except Exception as e:
            logger.error(f"Read error: {e}")
            return ""
    
    def read_until_prompt(self, prompt: str = "ch>", max_lines: int = 100) -> List[str]:
        """Read lines until prompt is found"""
        lines = []
        for _ in range(max_lines):
            line = self.readline()
            if not line:
                continue
            if line.startswith(prompt):
                break
            lines.append(line)
        return lines

class DeviceDetector:
    """Detects and identifies NanoVNA devices"""
    
    @staticmethod
    def get_available_ports() -> List[Dict]:
        """Get list of available serial ports with device info"""
        ports = []
        
        for port_info in list_ports.comports():
            # Fix Windows V2 hardware info if needed
            if platform.system() == "Windows" and port_info.vid is None:
                if r"PORTS\VID_04B4&PID_0008" in str(port_info.hwid):
                    port_info.vid, port_info.pid = 0x04B4, 0x0008
            
            device_name = DeviceDetector._identify_device(port_info)
            
            port_data = {
                'device': port_info.device,
                'description': port_info.description or "Unknown",
                'vid': port_info.vid,
                'pid': port_info.pid,
                'device_type': device_name,
                'manufacturer': getattr(port_info, 'manufacturer', 'Unknown')
            }
            ports.append(port_data)
            
        return ports
    
    @staticmethod
    def _identify_device(port_info: ListPortInfo) -> str:
        """Identify device type from USB VID/PID"""
        for device in KNOWN_DEVICES:
            if port_info.vid == device.vid and port_info.pid == device.pid:
                return device.name
        return "Unknown"
    
    @staticmethod
    def detect_device_version(interface: SerialInterface) -> str:
        """Detect specific NanoVNA version by communicating with device"""
        try:
            if not interface.open():
                return "Unknown"
            
            # Clear any pending data
            DeviceDetector._drain_serial(interface)
            
            # Send carriage return and check response
            interface.write("\r")
            time.sleep(0.1)
            
            response = interface.readline()
            
            if response.startswith("ch>"):
                return "NanoVNA-v1"
            elif response.startswith("\r\nch>") or response.startswith("2"):
                # Get more detailed info
                interface.write("info\r")
                time.sleep(0.1)
                info_lines = interface.read_until_prompt()
                info_text = "\n".join(info_lines)
                
                # Identify specific variants
                if "NanoVNA-H 4" in info_text:
                    return "NanoVNA-H4"
                elif "NanoVNA-H" in info_text:
                    return "NanoVNA-H"
                elif "NanoVNA-F" in info_text:
                    return "NanoVNA-F"
                elif "S-A-A-2" in info_text or response.startswith("2"):
                    return "NanoVNA-V2"
                
            return "NanoVNA"
            
        except Exception as e:
            logger.error(f"Device detection error: {e}")
            return "Unknown"
        finally:
            interface.close()
    
    @staticmethod
    def _drain_serial(interface: SerialInterface):
        """Drain any pending data from serial buffer"""
        if interface.serial_port:
            old_timeout = interface.serial_port.timeout
            interface.serial_port.timeout = 0.05
            
            for _ in range(50):
                data = interface.serial_port.read(128)
                if not data:
                    break
                    
            interface.serial_port.timeout = old_timeout

class NanoVNADevice:
    """Main NanoVNA device communication class"""
    
    def __init__(self, interface: SerialInterface):
        self.interface = interface
        self.device_type = "Unknown"
        self.is_connected = False
        self.acquisition_active = False
        self.acquisition_thread: Optional[threading.Thread] = None
        self.stop_event = Event()
        self.scan_callback: Optional[Callable] = None
        
        # Device capabilities
        self.max_frequency = 900e6  # Default for basic NanoVNA
        self.min_frequency = 50e3
        self.max_points = 201
        
    def connect(self) -> bool:
        """Connect to NanoVNA device"""
        try:
            if not self.interface.open():
                return False
            
            self.device_type = DeviceDetector.detect_device_version(self.interface)
            self._setup_device_capabilities()
            
            # Verify connection with a simple command
            if not self._test_connection():
                self.interface.close()
                return False
            
            self.is_connected = True
            logger.info(f"Connected to {self.device_type} on {self.interface.port}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from device"""
        try:
            self.stop_acquisition()
            self.interface.close()
            self.is_connected = False
            logger.info("Disconnected from NanoVNA")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def _setup_device_capabilities(self):
        """Setup device-specific capabilities"""
        capabilities = {
            "NanoVNA-H4": {"max_freq": 1500e6, "max_points": 401},
            "NanoVNA-H": {"max_freq": 900e6, "max_points": 201},
            "NanoVNA-V2": {"max_freq": 3e9, "max_points": 401},
            "NanoVNA-F": {"max_freq": 1500e6, "max_points": 201},
        }
        
        if self.device_type in capabilities:
            caps = capabilities[self.device_type]
            self.max_frequency = caps["max_freq"]
            self.max_points = caps["max_points"]
    
    def _test_connection(self) -> bool:
        """Test if device is responding properly"""
        try:
            self.interface.write("frequencies\r")
            time.sleep(0.2)
            response = self.interface.read_until_prompt()
            return len(response) > 0
        except Exception:
            return False
    
    def scan_frequencies(self, start: float, stop: float, points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform frequency sweep and return frequencies and S11 data"""
        try:
            if not self.is_connected:
                raise ValueError("Device not connected")
            
            # Set sweep parameters
            self.interface.write(f"sweep {int(start)} {int(stop)} {points}\r")
            time.sleep(0.1)
            
            # Get frequency data
            self.interface.write("frequencies\r")
            time.sleep(0.2)
            freq_lines = self.interface.read_until_prompt()
            
            frequencies = []
            for line in freq_lines:
                try:
                    freq = float(line.strip())
                    frequencies.append(freq)
                except ValueError:
                    continue
            
            if not frequencies:
                raise ValueError("No frequency data received")
            
            # Get S11 data
            self.interface.write("data 0\r")  # S11 data
            time.sleep(0.5)  # More time for data transfer
            s11_lines = self.interface.read_until_prompt()
            
            s11_complex = []
            for line in s11_lines:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        real = float(parts[0])
                        imag = float(parts[1])
                        s11_complex.append(complex(real, imag))
                except (ValueError, IndexError):
                    continue
            
            if len(s11_complex) != len(frequencies):
                logger.warning(f"Frequency/S11 length mismatch: {len(frequencies)} vs {len(s11_complex)}")
                # Truncate to shorter length
                min_len = min(len(frequencies), len(s11_complex))
                frequencies = frequencies[:min_len]
                s11_complex = s11_complex[:min_len]
            
            return np.array(frequencies), np.array(s11_complex)
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            raise
    
    def s11_to_impedance(self, s11: np.ndarray, z0: float = 50.0) -> np.ndarray:
        """Convert S11 to impedance"""
        return z0 * (1 + s11) / (1 - s11)
    
    def start_acquisition(self, start: float, stop: float, points: int, 
                         interval: float = 2.0, callback: Optional[Callable] = None) -> bool:
        """Start continuous data acquisition"""
        try:
            if self.acquisition_active:
                return False
            
            self.scan_callback = callback
            self.stop_event.clear()
            self.acquisition_active = True
            
            # Start acquisition thread
            self.acquisition_thread = threading.Thread(
                target=self._acquisition_worker,
                args=(start, stop, points, interval),
                daemon=True
            )
            self.acquisition_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start acquisition: {e}")
            return False
    
    def stop_acquisition(self):
        """Stop continuous data acquisition"""
        try:
            if self.acquisition_active:
                self.stop_event.set()
                self.acquisition_active = False
                
                if self.acquisition_thread and self.acquisition_thread.is_alive():
                    self.acquisition_thread.join(timeout=2.0)
                
                logger.info("Acquisition stopped")
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")
    
    def _acquisition_worker(self, start: float, stop: float, points: int, interval: float):
        """Background acquisition worker thread"""
        scan_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Perform scan
                frequencies, s11 = self.scan_frequencies(start, stop, points)
                impedance = self.s11_to_impedance(s11)
                
                # Prepare data package
                data_package = {
                    "timestamp": time.time(),
                    "scan_count": scan_count,
                    "frequencies": frequencies,
                    "s11": s11,
                    "impedance": impedance,
                    "resistance": impedance.real,
                    "phase": np.angle(impedance, deg=True)
                }
                
                # Call callback if provided
                if self.scan_callback:
                    try:
                        self.scan_callback(data_package)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                scan_count += 1
                
                # Wait for next scan
                self.stop_event.wait(interval)
                
            except Exception as e:
                logger.error(f"Acquisition worker error: {e}")
                # Try to continue after error
                self.stop_event.wait(1.0)

class NanoVNAController:
    """High-level controller for NanoVNA operations"""
    
    def __init__(self, notify_callback: Optional[Callable] = None):
        self.device: Optional[NanoVNADevice] = None
        self.notify_callback = notify_callback
        
        # Current sweep settings
        self.sweep_start = 1e6
        self.sweep_stop = 10e6
        self.sweep_points = 201
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available NanoVNA devices"""
        return DeviceDetector.get_available_ports()
    
    def connect_to_device(self, port: str = None) -> bool:
        """Connect to NanoVNA device"""
        try:
            if self.device:
                self.device.disconnect()
            
            # Auto-detect if no port specified
            if not port:
                available = self.get_available_devices()
                nano_devices = [d for d in available if "NanoVNA" in d['device_type']]
                if not nano_devices:
                    self._notify("error", "No Devices", "No NanoVNA devices found")
                    return False
                port = nano_devices[0]['device']
            
            # Create interface and device
            interface = SerialInterface(port)
            self.device = NanoVNADevice(interface)
            
            # Connect
            if self.device.connect():
                self._notify("info", "Connected", f"Connected to {self.device.device_type}")
                return True
            else:
                self._notify("error", "Connection Failed", "Could not connect to device")
                return False
                
        except Exception as e:
            self._notify("error", "Connection Error", str(e))
            return False
    
    def disconnect_device(self):
        """Disconnect from current device"""
        if self.device:
            self.device.disconnect()
            self.device = None
            self._notify("info", "Disconnected", "Device disconnected")
    
    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self.device is not None and self.device.is_connected
    
    def perform_sweep(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform single frequency sweep"""
        if not self.is_connected():
            raise ValueError("Device not connected")
        
        frequencies, s11 = self.device.scan_frequencies(
            self.sweep_start, self.sweep_stop, self.sweep_points
        )
        impedance = self.device.s11_to_impedance(s11)
        
        return frequencies, s11, impedance
    
    def start_continuous_sweep(self, interval: float = 2.0, callback: Optional[Callable] = None) -> bool:
        """Start continuous sweeping"""
        if not self.is_connected():
            return False
        
        return self.device.start_acquisition(
            self.sweep_start, self.sweep_stop, self.sweep_points,
            interval, callback
        )
    
    def stop_continuous_sweep(self):
        """Stop continuous sweeping"""
        if self.device:
            self.device.stop_acquisition()
    
    def set_sweep_parameters(self, start: float, stop: float, points: int):
        """Set sweep parameters"""
        self.sweep_start = start
        self.sweep_stop = stop
        self.sweep_points = points
    
    def _notify(self, level: str, title: str, message: str):
        """Send notification via callback"""
        if self.notify_callback:
            try:
                self.notify_callback(level, title, message)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

# Example usage for integration with your GUI
def create_nanovna_controller(gui_notify_callback):
    """Factory function to create NanoVNA controller for GUI"""
    return NanoVNAController(notify_callback=gui_notify_callback)