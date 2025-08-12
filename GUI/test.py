# test_unified.py
from Acquire_unified import scan_serial_ports_for_vna, classify_device_from_probe

print("🔍 Scanning serial ports for NanoVNA / compatible devices...\n")
devices_found = False

for dev, hint, banner, fwinfo in scan_serial_ports_for_vna():
    devices_found = True
    print(f"Port: {dev}")
    print(f"  VID/PID hint: {hint}")
    print(f"  Banner: {banner.strip()}")
    print(f"  Firmware info:\n{fwinfo}")
    model = classify_device_from_probe(banner, fwinfo, hint)
    print(f"  📡 Classified as: {model}")
    print("-" * 50)

if not devices_found:
    print("⚠ No devices found. Check cable connection and ensure device is powered on.")
