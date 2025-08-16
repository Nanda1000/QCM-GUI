import serial
import serial.tools.list_ports
import time
import numpy as np
import re

class NanoVNA:
    def __init__(self, port=None, baud=115200, z0=50):
        self.port = port
        self.baud = baud
        self.z0 = z0
        self.ser = None
        self.current_points = 101  # Default points
        self.last_start_freq = 50000  # Default start frequency
        self.last_stop_freq = 900000000  # Default stop frequency
        self.prompt_pattern = re.compile(r'ch[01]?>|ch>', re.IGNORECASE)
        
    def scan_ports(self):
        """Scan for available serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        print("\n[INFO] Scanning for available serial ports...")
        for port in ports:
            available_ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
            
        return available_ports
    
    def select_port(self):
        """Interactive port selection"""
        ports = self.scan_ports()
        
        if not ports:
            print("[ERROR] No serial ports found!")
            return None
        
        print("\nAvailable serial ports:")
        for i, port in enumerate(ports):
            print(f"{i+1}. {port['device']} - {port['description']}")
        
        while True:
            try:
                choice = input(f"\nSelect port (1-{len(ports)}) or press Enter for auto-detect: ").strip()
                
                if not choice:
                    # Auto-detect: try each port
                    print("\n[INFO] Auto-detecting NanoVNA...")
                    for port in ports:
                        print(f"[INFO] Trying {port['device']}...")
                        if self.test_connection(port['device']):
                            self.port = port['device']
                            print(f"[INFO] NanoVNA found on {port['device']}")
                            return port['device']
                    print("[ERROR] No NanoVNA found on any port")
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(ports):
                    selected_port = ports[choice_idx]['device']
                    if self.test_connection(selected_port):
                        self.port = selected_port
                        return selected_port
                    else:
                        print(f"[ERROR] No NanoVNA found on {selected_port}")
                        return None
                else:
                    print("Invalid selection. Please try again.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n[INFO] Port selection cancelled")
                return None
    
    def test_connection(self, port):
        """Test if a NanoVNA is connected to the specified port"""
        try:
            test_ser = serial.Serial(port, self.baud, timeout=1)
            time.sleep(0.5)
            test_ser.reset_input_buffer()
            test_ser.reset_output_buffer()
            
            # Send carriage return to get prompt
            test_ser.write(b"\r\n")
            time.sleep(0.5)
            
            # Try info command
            test_ser.write(b"info\r\n")
            time.sleep(0.5)
            
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2:
                if test_ser.in_waiting:
                    data = test_ser.read(test_ser.in_waiting).decode(errors='ignore')
                    response += data
                else:
                    time.sleep(0.1)
            
            test_ser.close()
            
            # Check for NanoVNA indicators
            response_lower = response.lower()
            indicators = ['nanovna', 'ch>', 'ch0>', 'ch1>', 'board_id', 'version']
            
            return any(indicator in response_lower for indicator in indicators)
            
        except Exception:
            return False
    
    def connect(self):
        """Connect to NanoVNA with improved detection"""
        if not self.port:
            self.port = self.select_port()
            if not self.port:
                return False
        
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(1)  # Give device time to initialize
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Send carriage return to get prompt
            self.ser.write(b"\r\n")
            time.sleep(0.5)
            
            # Read any initial response and banners
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2:
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting).decode(errors='ignore')
                    response += data
                else:
                    break
            
            print(f"[INFO] Initial response: '{response.strip()}'")
            
            # Try to get info to confirm connection
            info_response = self.send_cmd("info")
            
            # Check for prompt patterns or successful info response
            has_prompt = self.prompt_pattern.search(response)
            has_info = bool(info_response)
            
            if has_prompt or has_info:
                print(f"[INFO] NanoVNA connection established on {self.port}")
                return True
            else:
                print("[WARNING] Could not confirm NanoVNA connection")
                return False
                
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from NanoVNA"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[INFO] Disconnected from NanoVNA")
    
    def send_cmd(self, cmd):
        """Send a command and return the response lines"""
        if not self.ser or not self.ser.is_open:
            print("[ERROR] Serial connection not open")
            return []
        
        try:
            # Clear input buffer
            self.ser.reset_input_buffer()
            
            # Send command
            self.ser.write((cmd + "\r\n").encode())
            time.sleep(0.2)  # Wait for command processing
            
            lines = []
            start_time = time.time()
            
            # Read response until timeout or prompt
            while time.time() - start_time < 3:  # 3 second timeout
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        # Skip echo of the command
                        if line == cmd:
                            continue
                        # Check for prompt indicating end of response
                        if self.prompt_pattern.search(line):
                            break
                        lines.append(line)
                else:
                    time.sleep(0.1)
            
            print(f"[DEBUG] Command '{cmd}' returned {len(lines)} lines")
            return lines
            
        except Exception as e:
            print(f"[ERROR] Failed to send command '{cmd}': {e}")
            return []
    
    def get_data(self, data_type):
        """Get measurement data from NanoVNA"""
        if not self.ser or not self.ser.is_open:
            print("[ERROR] Serial connection not open")
            return []
        
        cmd = f"data {data_type}"
        
        try:
            # Clear buffers
            self.ser.reset_input_buffer()
            
            # Send data command
            self.ser.write((cmd + "\r\n").encode())
            time.sleep(0.2)
            
            lines = []
            start_time = time.time()
            echo_skipped = False
            
            # Read data until prompt or timeout
            while time.time() - start_time < 10:  # 10 second timeout for data
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    
                    if not line:
                        continue
                    
                    # Skip command echo
                    if not echo_skipped and line == cmd:
                        echo_skipped = True
                        continue
                    
                    # Check for prompt (end of data)
                    if self.prompt_pattern.search(line):
                        break
                    
                    # Check if line looks like data (two numbers)
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            lines.append(line)
                        except ValueError:
                            # Not a data line, skip
                            pass
                else:
                    time.sleep(0.1)
            
            print(f"[DEBUG] Retrieved {len(lines)} data points for data type {data_type}")
            return lines
        
        except Exception as e:
            print(f"[ERROR] Failed to get data: {e}")
            return []
    
    def parse_s11_data(self, lines):
        """Parse data lines into complex S11 values"""
        s11_list = []
        
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    re = float(parts[0])
                    im = float(parts[1])
                    s11_list.append(complex(re, im))
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Could not parse S11 line: '{line}' - {e}")
                continue
        
        print(f"[INFO] Parsed {len(s11_list)} S11 values")
        return s11_list
    
    def parse_s21_data(self, lines):
        """Parse data lines into complex S21 values"""
        s21_list = []
        
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    re = float(parts[0])
                    im = float(parts[1])
                    s21_list.append(complex(re, im))
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Could not parse S21 line: '{line}' - {e}")
                continue
        
        print(f"[INFO] Parsed {len(s21_list)} S21 values")
        return s21_list
    
    def s11_to_impedance(self, s11):
        """Convert S11 to impedance"""
        if abs(1 - s11) < 1e-10:  # Avoid division by zero
            return complex(float('inf'), float('inf'))
        return self.z0 * (1 + s11) / (1 - s11)
    
    def get_frequencies(self, num_points=None):
        """Get frequency array based on current sweep settings"""
        if num_points is None:
            num_points = self.current_points
        
        return np.linspace(self.last_start_freq, self.last_stop_freq, num_points)
    
    def interactive_sweep_setup(self):
        """Interactive sweep parameter setup"""
        print("\n=== Sweep Configuration ===")
        print(f"Current settings:")
        print(f"  Start frequency: {self.last_start_freq:,} Hz ({self.last_start_freq/1e6:.3f} MHz)")
        print(f"  Stop frequency: {self.last_stop_freq:,} Hz ({self.last_stop_freq/1e6:.3f} MHz)")
        print(f"  Points: {self.current_points}")
        
        try:
            # Start frequency
            start_input = input(f"\nStart frequency (Hz) [current: {self.last_start_freq}]: ").strip()
            if start_input:
                # Handle MHz input
                if start_input.lower().endswith('mhz'):
                    start_freq = int(float(start_input[:-3]) * 1e6)
                elif start_input.lower().endswith('khz'):
                    start_freq = int(float(start_input[:-3]) * 1e3)
                else:
                    start_freq = int(float(start_input))
            else:
                start_freq = self.last_start_freq
            
            # Stop frequency
            stop_input = input(f"Stop frequency (Hz) [current: {self.last_stop_freq}]: ").strip()
            if stop_input:
                # Handle MHz input
                if stop_input.lower().endswith('mhz'):
                    stop_freq = int(float(stop_input[:-3]) * 1e6)
                elif stop_input.lower().endswith('khz'):
                    stop_freq = int(float(stop_input[:-3]) * 1e3)
                else:
                    stop_freq = int(float(stop_input))
            else:
                stop_freq = self.last_stop_freq
            
            # Number of points
            points_input = input(f"Number of points [current: {self.current_points}]: ").strip()
            if points_input:
                points = int(points_input)
            else:
                points = self.current_points
            
            # Validate inputs
            if start_freq >= stop_freq:
                print("[ERROR] Start frequency must be less than stop frequency")
                return False
            
            if points < 2 or points > 1000:
                print("[ERROR] Points must be between 2 and 1000")
                return False
            
            return self.sweep(start_freq, stop_freq, points)
            
        except ValueError as e:
            print(f"[ERROR] Invalid input: {e}")
            return False
        except KeyboardInterrupt:
            print("\n[INFO] Sweep setup cancelled")
            return False
    
    def sweep(self, start_freq=None, stop_freq=None, points=None):
        """Set up frequency sweep parameters"""
        if start_freq is None:
            start_freq = self.last_start_freq
        if stop_freq is None:
            stop_freq = self.last_stop_freq
        if points is None:
            points = self.current_points
            
        try:
            # Send sweep command
            response = self.send_cmd(f"sweep {start_freq} {stop_freq} {points}")
            
            # Update stored values
            self.last_start_freq = start_freq
            self.last_stop_freq = stop_freq
            self.current_points = points
            
            print(f"[INFO] Sweep set: {start_freq:,} Hz to {stop_freq:,} Hz, {points} points")
            print(f"[INFO] Frequency range: {start_freq/1e6:.3f} MHz to {stop_freq/1e6:.3f} MHz")
            time.sleep(2)  # Wait for sweep to complete
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to set sweep: {e}")
            return False
    
    def acquire(self, get_s21=False):
        """Acquire S11 data and optionally S21 data, convert S11 to impedance"""
        try:
            # Get S11 data (data 0)
            raw_lines_s11 = self.get_data(0)
            
            if not raw_lines_s11:
                print("[ERROR] No S11 data received")
                return None, None, None, None, None, None, None, None, None
            
            # Parse S11 values
            s11_values = self.parse_s11_data(raw_lines_s11)
            
            if not s11_values:
                print("[ERROR] No valid S11 values parsed")
                return None, None, None, None, None, None, None, None, None
            
            # Generate frequency array
            freqs = self.get_frequencies(len(s11_values))
            
            # Convert S11 to impedances
            impedances = np.array([self.s11_to_impedance(s) for s in s11_values])
            resistance = impedances.real
            reactance = impedances.imag
            magnitude = np.abs(impedances)
            phase = np.angle(impedances, deg=True)
            
            print(f"[INFO] Acquired {len(s11_values)} S11 measurements")
            print(f"[INFO] Frequency range: {freqs[0]/1e6:.3f} MHz to {freqs[-1]/1e6:.3f} MHz")
            
            # Optionally get S21 data
            s21_values = None
            freqs_s21 = None
            if get_s21:
                raw_lines_s21 = self.get_data(1)
                if raw_lines_s21:
                    s21_values = self.parse_s21_data(raw_lines_s21)
                    if s21_values:
                        freqs_s21 = self.get_frequencies(len(s21_values))
                        print(f"[INFO] Acquired {len(s21_values)} S21 measurements")
                    else:
                        print("[WARNING] No valid S21 values parsed")
                else:
                    print("[WARNING] No S21 data received")
            
            return freqs, s11_values, impedances, resistance, reactance, magnitude, phase, s21_values, freqs_s21
            
        except Exception as e:
            print(f"[ERROR] Failed to acquire data: {e}")
            return None, None, None, None, None, None, None, None, None
    
    def get_info(self):
        """Get device information"""
        return self.send_cmd("info")
    
    def reset(self):
        """Reset the device"""
        return self.send_cmd("reset")
    
    def quick_measurement(self):
        """Quick measurement with current settings"""
        print("\n=== Quick Measurement ===")
        result = self.acquire(get_s21=True)  # Get both S11 and S21
        
        if result[0] is not None:
            freqs, s11, impedances, resistance, reactance, magnitude, phase, s21, freqs_s21 = result
            
            print(f"\nMeasurement Summary:")
            print(f"S11 frequency points: {len(freqs)}")
            print(f"S11 frequency range: {freqs[0]/1e6:.3f} - {freqs[-1]/1e6:.3f} MHz")
            
            if s21 is not None:
                print(f"S21 frequency points: {len(s21)}")
                print(f"S21 frequency range: {freqs_s21[0]/1e6:.3f} - {freqs_s21[-1]/1e6:.3f} MHz")
            
            # Show some sample values
            mid_idx = len(impedances) // 2
            print(f"\nSample values at {freqs[mid_idx]/1e6:.3f} MHz:")
            print(f"  S11: {s11[mid_idx]}")
            print(f"  Impedance: {resistance[mid_idx]:.1f} + j{reactance[mid_idx]:.1f} Ω")
            print(f"  Magnitude: {magnitude[mid_idx]:.1f} Ω")
            print(f"  Phase: {phase[mid_idx]:.1f}°")
            
            if s21 is not None:
                print(f"  S21: {s21[mid_idx]}")
            
            return result
        else:
            print("[ERROR] Measurement failed")
            return None

def main():
    """Main interactive program"""
    print("=== Enhanced NanoVNA Data Acquisition ===")
    print("Compatible with NanoVNA-H, NanoVNA-H4, NanoVNA-F, and clones")
    
    # Create NanoVNA instance (no port specified - will auto-detect)
    vna = NanoVNA()
    
    try:
        # Connect to device
        print("\n=== Connection Setup ===")
        if not vna.connect():
            print("[ERROR] Failed to connect to NanoVNA!")
            return
        
        # Get device info
        print("\n=== Device Information ===")
        info = vna.get_info()
        if info:
            for line in info:
                print(f"  {line}")
        else:
            print("  Could not retrieve device information")
        
        # Interactive menu
        while True:
            print("\n=== Main Menu ===")
            print("1. Configure sweep parameters")
            print("2. Quick measurement (current settings)")
            print("3. Show current settings")
            print("4. Reset device")
            print("5. Exit")
            
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    vna.interactive_sweep_setup()
                    
                elif choice == '2':
                    vna.quick_measurement()
                    
                elif choice == '3':
                    print(f"\nCurrent Settings:")
                    print(f"  Port: {vna.port}")
                    print(f"  Start frequency: {vna.last_start_freq:,} Hz ({vna.last_start_freq/1e6:.3f} MHz)")
                    print(f"  Stop frequency: {vna.last_stop_freq:,} Hz ({vna.last_stop_freq/1e6:.3f} MHz)")
                    print(f"  Points: {vna.current_points}")
                    print(f"  Reference impedance: {vna.z0} Ω")
                    
                elif choice == '4':
                    print("Resetting device...")
                    vna.reset()
                    time.sleep(2)
                    
                elif choice == '5':
                    break
                    
                else:
                    print("Invalid selection. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n[INFO] Exiting...")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
        
    finally:
        vna.disconnect()
        print("Program terminated.")

# Example usage for scripted operation:
if __name__ == "__main__":
    # Run interactive main program
    main()