# updated_nano_vna.py - Enhanced with robust logging and error handling
import serial
import serial.tools.list_ports
import time
import numpy as np
import csv
import logging
from datetime import datetime
import threading
from typing import Optional, Tuple, List
import queue
from PyQt6.QtWidgets import QMessageBox


class NanoVNA:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.Z0 = 50  # Characteristic impedance
        self.is_connected = False
        self.acquisition_active = False
        self.data_queue = queue.Queue()
        self.error_count = 0
        self.max_errors = 10
        self.last_successful_scan = None
        
    def connect(self, retries=3):
        """Connect to NanoVNA with enhanced error handling and auto-detection"""
        for attempt in range(retries):
            try:
                if self.port is None:
                    self.port = self._auto_detect_port()
                    
                print(f"Attempting connection to port: {self.port} (Attempt {attempt+1})")
                
                # Close any existing connection
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    time.sleep(0.5)
                
                self.ser = serial.Serial(
                    self.port, 
                    self.baudrate, 
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                time.sleep(2)  # Allow device to initialize
                
                # Clear buffers
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                
                # Test connection with version command
                self._send_command("ver")
                response = self._read_line(timeout=3)
                
                if response and ("nanovna" in response.lower() or "nanovna" in response.lower()):
                    print(f"Successfully connected to NanoVNA: {response}")
                    self.is_connected = True
                    self.error_count = 0
                    return True
                else:
                    QMessageBox.warning(f"Unexpected response: {response}")
                    
            except Exception as e:
                QMessageBox.error(f"Connection attempt {attempt+1} failed: {e}")
                if self.ser and self.ser.is_open:
                    self.ser.close()
                time.sleep(1)
        
        QMessageBox.error("Unable to connect to NanoVNA after multiple attempts")
        self.is_connected = False
        return False

    def _auto_detect_port(self):
        """Auto-detect NanoVNA port"""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            raise Exception("No serial ports found")
        
        # Try to find NanoVNA-specific ports first
        for port in ports:
            if any(keyword in port.description.lower() for keyword in ['nano', 'vna', 'ch340', 'ch341']):
                QMessageBox.info(f"Auto-detected potential NanoVNA port: {port.device} - {port.description}")
                return port.device
        
        # Fallback to first available port
        QMessageBox.info(f"Using first available port: {ports[0].device}")
        return ports[0].device

    def disconnect(self):
        """Safely disconnect from device"""
        try:
            self.stop_acquisition()
            if self.ser and self.ser.is_open:
                self.ser.close()
                QMessageBox.info("Disconnected from NanoVNA")
            self.is_connected = False
        except Exception as e:
            QMessageBox.error(f"Error during disconnect: {e}")

    def _send_command(self, cmd):
        """Send command with error handling"""
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")
        
        try:
            command = (cmd.strip() + "\r\n").encode('utf-8')
            self.ser.write(command)
            self.ser.flush()  # Ensure command is sent
            QMessageBox(f"Sent command: {cmd}")
        except Exception as e:
            QMessageBox.error(f"Error sending command '{cmd}': {e}")
            raise

    def _read_line(self, timeout=None):
        """Read line with timeout and error handling"""
        if not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")
        
        original_timeout = self.ser.timeout
        if timeout:
            self.ser.timeout = timeout
            
        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            QMessageBox(f"Received: {line}")
            return line
        except Exception as e:
            QMessageBox.error(f"Error reading line: {e}")
            return ''
        finally:
            if timeout:
                self.ser.timeout = original_timeout

    def _read_lines_until(self, end_marker='ch0', timeout=5):
        """Read multiple lines until end marker with timeout"""
        lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                line = self._read_line(timeout=0.5)
                if not line:
                    continue
                    
                if line == end_marker:
                    QMessageBox(f"Found end marker: {end_marker}")
                    break
                    
                lines.append(line)
                
            except Exception as e:
                QMessageBox.error(f"Error reading lines: {e}")
                break
        
        if time.time() - start_time >= timeout:
            QMessageBox.warning(f"Timeout waiting for end marker: {end_marker}")
            
        return lines

    def scan(self, start_freq, stop_freq, points):
        """Perform frequency scan with enhanced error handling"""
        if not self.is_connected or not self.ser or not self.ser.is_open:
            raise Exception("Device not connected")

        try:
            start_khz = int(start_freq / 1e3)
            stop_khz = int(stop_freq / 1e3)
            
            QMessageBox(f"Starting scan: {start_khz}kHz to {stop_khz}kHz, {points} points")
            
            # Set frequency range
            self._send_command(f"sweep {start_khz} {stop_khz} {points}")
            time.sleep(0.2)  # Allow command to process
            
            # Request frequency data
            self._send_command("frequencies")
            freq_lines = self._read_lines_until('ch0', timeout=10)
            
            # Parse frequencies
            freqs = []
            for line in freq_lines:
                try:
                    freq = float(line) * 1e3  # Convert kHz to Hz
                    freqs.append(freq)
                except ValueError:
                    QMessageBox.warning(f"Invalid frequency data: {line}")
                    continue
            
            if len(freqs) == 0:
                raise Exception("No valid frequency data received")
            
            QMessageBox(f"Received {len(freqs)} frequency points")
            
            # Request S11 data
            self._send_command("data 0")
            data_lines = self._read_lines_until('ch0', timeout=10)
            
            # Parse S11 complex data
            s11 = []
            for line in data_lines:
                try:
                    if ',' in line:
                        real_str, imag_str = line.split(',', 1)
                        s11_complex = complex(float(real_str.strip()), float(imag_str.strip()))
                        s11.append(s11_complex)
                    else:
                        QMessageBox.warning(f"Invalid S11 data format: {line}")
                        continue
                except ValueError as e:
                    QMessageBox.warning(f"Error parsing S11 data '{line}': {e}")
                    continue
            
            if len(s11) == 0:
                raise Exception("No valid S11 data received")
            
            # Ensure data consistency
            min_len = min(len(freqs), len(s11))
            if min_len == 0:
                raise Exception("No valid data points received")
            
            freqs = np.array(freqs[:min_len])
            s11 = np.array(s11[:min_len])
            
            QMessageBox(f"Scan completed successfully: {len(freqs)} points")
            self.last_successful_scan = time.time()
            self.error_count = 0  # Reset error count on successful scan
            
            return freqs, s11
            
        except Exception as e:
            self.error_count += 1
            QMessageBox.error(f"Scan failed (error #{self.error_count}): {e}")
            
            if self.error_count >= self.max_errors:
                QMessageBox(f"Maximum errors ({self.max_errors}) reached. Disconnecting.")
                self.disconnect()
            
            raise

    def s11_to_impedance(self, s11):
        """Convert S11 to impedance with numerical stability"""
        s11 = np.asarray(s11, dtype=complex)
        
        # Handle numerical issues
        denom = 1 - s11
        # Replace very small denominators to avoid division by zero
        small_mask = np.abs(denom) < 1e-15
        denom[small_mask] = 1e-15 + 0j
        
        impedance = self.Z0 * (1 + s11) / denom
        
        # Sanity check on impedance values
        impedance = np.where(np.abs(impedance) > 1e6, 1e6 + 0j, impedance)
        impedance = np.where(np.abs(impedance) < 1e-6, 1e-6 + 0j, impedance)
        
        return impedance

    def start_acquisition(self, start_freq, stop_freq, points, interval=2.0, callback=None):
        """Start continuous data acquisition in a separate thread"""
        if self.acquisition_active:
            QMessageBox.warning("Acquisition already active")
            return False
        
        if not self.is_connected:
            QMessageBox.error("Device not connected")
            return False
        
        self.acquisition_active = True
        QMessageBox.info(f"Starting acquisition: {start_freq/1e6:.1f}-{stop_freq/1e6:.1f}MHz, {points} points, {interval}s interval")
        
        def acquisition_worker():
            scan_count = 0
            while self.acquisition_active:
                try:
                    start_time = time.time()
                    
                    # Perform scan
                    freqs, s11 = self.scan(start_freq, stop_freq, points)
                    impedance = self.s11_to_impedance(s11)
                    
                    # Calculate derived parameters
                    resistance = impedance.real
                    reactance = impedance.imag
                    phase = np.angle(impedance, deg=True)
                    magnitude = np.abs(impedance)
                    
                    elapsed = time.time() - start_time
                    scan_count += 1
                    
                    # Prepare data package
                    data_package = {
                        'timestamp': datetime.now(),
                        'scan_count': scan_count,
                        'frequencies': freqs,
                        'impedance': impedance,
                        'resistance': resistance,
                        'reactance': reactance,
                        'phase': phase,
                        'magnitude': magnitude,
                        's11': s11,
                        'acquisition_time': elapsed
                    }
                    
                    # Add to queue
                    try:
                        self.data_queue.put_nowait(data_package)
                    except queue.Full:
                        QMessageBox.warning("Data queue full, dropping oldest data")
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(data_package)
                        except queue.Empty:
                            pass
                    
                    # Call callback if provided
                    if callback:
                        try:
                            callback(data_package)
                        except Exception as e:
                            QMessageBox.error(f"Callback error: {e}")
                    
                    QMessageBox.info(f"Scan #{scan_count} completed in {elapsed:.2f}s")
                    
                    # Wait for next scan
                    remaining_time = max(0, interval - elapsed)
                    if remaining_time > 0:
                        time.sleep(remaining_time)
                        
                except Exception as e:
                    QMessageBox.error(f"Acquisition error: {e}")
                    if not self.is_connected or self.error_count >= self.max_errors:
                        logger.critical("Stopping acquisition due to connection issues")
                        break
                    time.sleep(1)  # Brief pause before retry
            
            QMessageBox.info(f"Acquisition stopped. Total scans: {scan_count}")
        
        # Start acquisition thread
        self.acquisition_thread = threading.Thread(target=acquisition_worker, daemon=True)
        self.acquisition_thread.start()
        
        return True

    def stop_acquisition(self):
        """Stop continuous data acquisition"""
        if self.acquisition_active:
            QMessageBox.info("Stopping acquisition...")
            self.acquisition_active = False
            if hasattr(self, 'acquisition_thread') and self.acquisition_thread.is_alive():
                self.acquisition_thread.join(timeout=5)
            QMessageBox.info("Acquisition stopped")

    def get_latest_data(self):
        """Get the latest data from the queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def get_connection_status(self):
        """Get detailed connection status"""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'acquisition_active': self.acquisition_active,
            'error_count': self.error_count,
            'queue_size': self.data_queue.qsize(),
            'last_successful_scan': self.last_successful_scan
        }

def acquire_data(device_or_port, start_freq=1e6, stop_freq=10e6, points=201):
    """
    Legacy function for backward compatibility
    Can accept either a NanoVNA instance or a port string
    """
    if isinstance(device_or_port, str):
        # Port string provided - create temporary NanoVNA instance
        vna = NanoVNA(port=device_or_port)
        try:
            if not vna.connect():
                raise Exception("Failed to connect to device")
            
            freqs, s11 = vna.scan(start_freq, stop_freq, points)
            impedance = vna.s11_to_impedance(s11)
            
        finally:
            vna.disconnect()
    else:
        # NanoVNA instance provided
        vna = device_or_port
        freqs, s11 = vna.scan(start_freq, stop_freq, points)
        impedance = vna.s11_to_impedance(s11)
    
    # Calculate derived parameters
    resistance = impedance.real
    reactance = impedance.imag
    phase = np.angle(impedance, deg=True)
    
    # Create time array (for backward compatibility)
    time_array = np.linspace(0, 1, len(freqs))
    
    return freqs, resistance, impedance, reactance, phase, time_array

def save_data_to_csv(data_package, filename=None):
    """Save data package to CSV file"""
    if filename is None:
        timestamp = data_package['timestamp'].strftime("%Y%m%d_%H%M%S")
        filename = f"nanovna_scan_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Frequency_Hz', 'Resistance_Ohm', 'Reactance_Ohm', 
                'Phase_deg', 'Magnitude_Ohm', 'S11_real', 'S11_imag'
            ])
            
            # Write data
            for i in range(len(data_package['frequencies'])):
                writer.writerow([
                    data_package['frequencies'][i],
                    data_package['resistance'][i],
                    data_package['reactance'][i],
                    data_package['phase'][i],
                    data_package['magnitude'][i],
                    data_package['s11'][i].real,
                    data_package['s11'][i].imag
                ])
        
        QMessageBox.info(f"Data saved to {filename}")
        return filename
    
    except Exception as e:
        QMessageBox.error(f"Error saving data to CSV: {e}")
        return None

if __name__ == "__main__":
    # Example usage with continuous logging
    vna = NanoVNA()
    
    def data_callback(data):
        """Callback function to process each scan"""
        print(f"Scan #{data['scan_count']}: "
              f"{len(data['frequencies'])} points, "
              f"acquisition time: {data['acquisition_time']:.2f}s")
    
    try:
        # Connect to device
        if not vna.connect():
            raise Exception("Failed to connect to NanoVNA")
        
        # Start continuous acquisition
        success = vna.start_acquisition(
            start_freq=1e6,
            stop_freq=10e6,
            points=201,
            interval=2.0,
            callback=data_callback
        )
        
        if not success:
            raise Exception("Failed to start acquisition")
        
        # Run for specified duration or until interrupted
        print("Acquisition started. Press Ctrl+C to stop...")
        
        while vna.acquisition_active:
            time.sleep(1)
            
            # Optionally process data from queue
            latest_data = vna.get_latest_data()
            if latest_data:
                # Save to individual files or process as needed
                # save_data_to_csv(latest_data)
                pass
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        QMessageBox.error(f"Error: {e}")
    finally:
        vna.disconnect()