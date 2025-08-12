# diag_nano.py
import serial
import time

PORT = "COM12"   # change if needed
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Send Enter to get any banner
    ser.write(b"\r")
    time.sleep(0.2)
    print("After CR:", ser.read_all().decode(errors="ignore"))

    # Try 'ver'
    ser.write(b"ver\r")
    time.sleep(0.5)
    print("After 'ver':", ser.read_all().decode(errors="ignore"))

    # Try 'info'
    ser.write(b"info\r")
    time.sleep(0.5)
    print("After 'info':", ser.read_all().decode(errors="ignore"))

    # Try 'version'
    ser.write(b"version\r")
    time.sleep(0.5)
    print("After 'version':", ser.read_all().decode(errors="ignore"))
