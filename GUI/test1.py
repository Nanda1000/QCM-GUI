# test_connect.py
from Acquire_data import NanoVNA
from Acquire_unified import UnifiedNanoVNA

print("=== Testing direct ASCII driver (Acquire_data.NanoVNA) ===")
vna_ascii = NanoVNA(port=None, notify_callback=lambda l,t,m: print(f"[{l}] {t}: {m}"))
if vna_ascii.connect():
    print("ASCII driver connected successfully!")
    try:
        freqs, s11 = vna_ascii.scan(1e6, 2e6, 5)
        print("Freqs:", freqs)
        print("S11:", s11)
    except Exception as e:
        print("Scan failed:", e)
    vna_ascii.disconnect()
else:
    print("ASCII driver connection failed.")

print("\n=== Testing Unified driver, skipping ASCII (Acquire_unified.UnifiedNanoVNA) ===")
vna_unified = UnifiedNanoVNA(port=None, notify_callback=lambda l,t,m: print(f"[{l}] {t}: {m}"), prefer_ascii_first=False)
if vna_unified.connect():
    print("Unified driver connected successfully!")
    try:
        freqs, s11 = vna_unified.scan(1e6, 2e6, 5)
        print("Freqs:", freqs)
        print("S11:", s11)
    except Exception as e:
        print("Scan failed:", e)
    vna_unified.disconnect()
else:
    print("Unified driver connection failed.")
