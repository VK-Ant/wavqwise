"""WavqWise EEG Signal Analysis Demo"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wavqwise import SignalPipeline

# Load EEG data
sig = SignalPipeline()
sig.load("demos/sample_data/eeg_sample.csv", channels=["Fp1", "Fp2", "C3", "C4"], sample_rate=256)

# Filter (band-pass 1-50 Hz, notch 50 Hz)
sig.filter(low=1, high=50, notch=50)

# Extract frequency bands
print("=== Band Power Extraction ===")
bands = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
for name, power in bands.bands.items():
    print(f"  {name}: {power:.4f} uV^2")

# Detect events
print("\n=== Event Detection ===")
events = sig.detect_events(threshold=3.0)
print(f"  Events detected: {len(events.events)}")
