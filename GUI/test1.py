from Acquire_unified import UnifiedNanoVNA

# Create the VNA object (None = auto-detect)
vna = UnifiedNanoVNA(port=None, notify_callback=print)

print("Connecting to NanoVNA...")
if vna.connect():
    print("Connected!")
    try:
        # Test scan
        start_freq = 1e6      # 1 MHz
        stop_freq = 10e6      # 10 MHz
        points = 21           # fewer points for faster test
        freqs, s11 = vna.scan(start_freq, stop_freq, points)
        print(f"Frequencies: {freqs[:5]} ...")
        print(f"S11 first 5 points: {s11[:5]}")
    except Exception as e:
        print("Scan failed:", e)
else:
    print("Failed to connect.")
