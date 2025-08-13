# Acquire_data.py – Code Documentation

This script provides an interface for communicating with NanoVNA devices over a serial connection.  
It supports device detection, connection, sweep configuration, and S11 measurement acquisition.

---

## 1. Global Initialization
```python
def __init__(self, port=None, baud=115200, z0=50)
```
**Purpose:** Set up initial parameters for the NanoVNA connection.  
**Key Attributes:**
- `port`: Serial port (e.g., `"COM3"` or `"/dev/ttyUSB0"`), default `None`.
- `baud`: Baud rate for serial communication (default 115200).
- `z0`: Reference impedance, default 50 Ω.
- `current_points`, `last_start_freq`, `last_stop_freq`: Default sweep parameters.
- `prompt_pattern`: Regex for detecting NanoVNA command prompts (`ch>`, `ch0>`, etc.).

---

## 2. scan_ports()
```python
def scan_ports(self)
```
**Purpose:** Detect all available serial ports.  
**Returns:** List of dictionaries, each with:
- `device` – Port name (e.g., `"COM5"`)
- `description` – Device description
- `hwid` – Hardware ID  

**Why:** Needed so the user can choose the correct COM port or for auto-detection.

---

## 3. select_port()
```python
def select_port(self)
```
**Purpose:** Let the user choose a port or auto-detect the NanoVNA.  
**Behavior:**
- Lists all available ports.
- If user presses Enter, attempts auto-detection by testing each port with `test_connection()`.  

**Returns:** Selected port name (`str`) or `None` if not found.

---

## 4. test_connection()
```python
def test_connection(self, port)
```
**Purpose:** Check if a NanoVNA is connected to the given port.  
**Process:**
1. Open serial port.
2. Reset input/output buffers.
3. Send carriage return to “wake up” device.
4. Send `"info"` command.
5. Read response for up to 2 seconds.
6. Search for known NanoVNA keywords (`nanovna`, `ch>`, `board_id`, etc.).  

**Returns:** `True` if device matches, otherwise `False`.

---

## 5. connect()
```python
def connect(self)
```
**Purpose:** Establish a confirmed connection to the NanoVNA.  
**Steps:**
1. If `self.port` is not set, call `select_port()`.
2. Open serial connection, reset buffers.
3. Send wake-up command (`\r\n`).
4. Read initial banners or prompts.
5. Call `send_cmd("info")` to confirm device type.  

**Returns:** `True` if connection confirmed, else `False`.

---

## 6. disconnect()
```python
def disconnect(self)
```
**Purpose:** Close the serial connection.  
**Why:** Prevents port locking issues.

---

## 7. send_cmd()
```python
def send_cmd(self, cmd)
```
**Purpose:** Send a command and collect the response.  
**Steps:**
- Clear input buffer.
- Send command followed by `\r\n`.
- Read lines until prompt is detected or timeout occurs.
- Skip command echo in responses.  

**Returns:** List of response lines (`list[str]`).

---

## 8. get_data()
```python
def get_data(self, data_type=0)
```
**Purpose:** Request measurement data from the NanoVNA.  
**Steps:**
- Send `"data {data_type}"`.
- Read lines until prompt is detected or timeout.
- Only keep lines containing **two numeric values**.  

**Returns:** List of strings, each `"real imag"`.

---

## 9. parse_data()
```python
def parse_data(self, lines)
```
**Purpose:** Convert text data lines into complex S11 values.  
**Input:** List of strings with two floats.  
**Returns:** List of `complex` numbers.

---

## 10. s11_to_impedance()
```python
def s11_to_impedance(self, s11)
```
**Purpose:** Convert an S11 reflection coefficient into impedance.  
**Formula:**
```math
Z = Z_0 rac{1+S_{11}}{1-S_{11}}
```
**Returns:** Complex impedance.

---

## 11. get_frequencies()
```python
def get_frequencies(self, num_points=None)
```
**Purpose:** Generate frequency array based on stored sweep settings.  
**Returns:** NumPy array from `last_start_freq` to `last_stop_freq`.

---

## 12. interactive_sweep_setup()
```python
def interactive_sweep_setup(self)
```
**Purpose:** Let the user configure sweep start/stop frequencies and number of points.  
**Validations:**
- Start < Stop
- Points between 2 and 1000  

**Updates:** Stored sweep parameters via `sweep()`.

---

## 13. sweep()
```python
def sweep(self, start_freq, stop_freq, points)
```
**Purpose:** Update sweep parameters on the device.  
**Process:**
- Send `"sweep start stop points"`.
- Update internal tracking variables.  

**Returns:** `True` if successful.

---

## 14. acquire()
```python
def acquire(self)
```
**Purpose:** Complete acquisition of S11 data and convert to impedance-related metrics.  
**Steps:**
- Get raw S11 from `get_data(0)`.
- Parse to complex S11 values.
- Generate frequencies.
- Convert to impedance, resistance, reactance, magnitude, phase.  

**Returns:** Tuple:
```python
(freqs, s11_values, impedances, resistance, reactance, magnitude, phase)
```

---

## 15. get_info()
```python
def get_info(self)
```
**Purpose:** Retrieve NanoVNA device info.  
**Returns:** List of info strings.

---

## 16. reset()
```python
def reset(self)
```
**Purpose:** Reset the NanoVNA.  
**Returns:** Device’s response.

---

## 17. quick_measurement()
```python
def quick_measurement(self)
```
**Purpose:** Run a fast measurement with current sweep settings and print a summary.  
**Outputs:** Displays mid-point measurement details.

---

## 18. main()
```python
def main()
```
**Purpose:** Interactive CLI for connecting to a NanoVNA and performing operations.  
**Menu Options:**
1. Configure sweep parameters
2. Quick measurement
3. Show current settings
4. Reset device
5. Exit
