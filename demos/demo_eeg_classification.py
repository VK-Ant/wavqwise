"""
WavqWise Demo: EEG Signal Classification
=========================================
Classify EEG signals into mental states using frequency band power ratios.

Data: Synthetic multi-class EEG (simulates real clinical patterns)
  - Class 0: Relaxed (high alpha, low beta)
  - Class 1: Focused (low alpha, high beta)
  - Class 2: Drowsy (high theta, high delta)

This demo shows:
  1. Load multi-channel EEG data
  2. Band-pass filter + notch filter
  3. Extract frequency band power features
  4. Classify mental state using sklearn
  5. Visualize results
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavqwise import SignalPipeline


# === 1. Generate realistic multi-class EEG dataset ===
def generate_eeg_dataset(n_samples=300, duration=4, sample_rate=256):
    """Generate labeled EEG epochs simulating 3 mental states."""
    n_points = duration * sample_rate
    t = np.arange(n_points) / sample_rate
    data = []
    labels = []

    for i in range(n_samples):
        label = i % 3
        noise = np.random.normal(0, 1.5, n_points)

        if label == 0:  # Relaxed: dominant alpha (8-13 Hz)
            signal = (
                8 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
                + 2 * np.sin(2 * np.pi * 6 * t)
                + 1 * np.sin(2 * np.pi * 20 * t)
                + noise
            )
        elif label == 1:  # Focused: dominant beta (13-30 Hz)
            signal = (
                2 * np.sin(2 * np.pi * 10 * t)
                + 7 * np.sin(2 * np.pi * 22 * t + np.random.uniform(0, 2*np.pi))
                + 3 * np.sin(2 * np.pi * 18 * t)
                + noise
            )
        else:  # Drowsy: dominant theta+delta (0.5-8 Hz)
            signal = (
                6 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
                + 5 * np.sin(2 * np.pi * 2 * t)
                + 1 * np.sin(2 * np.pi * 10 * t)
                + noise
            )

        data.append(signal)
        labels.append(label)

    return np.array(data), np.array(labels), t


# === 2. Extract band power features using WavqWise ===
def extract_features(signals, sample_rate=256):
    """Extract frequency band power for each epoch."""
    features = []
    band_defs = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for signal in signals:
        # Band-pass filter
        b, a = butter(4, [0.5, 50], btype="band", fs=sample_rate)
        filtered = filtfilt(b, a, signal)

        # Power spectral density
        freqs, psd = welch(filtered, fs=sample_rate, nperseg=min(256, len(filtered)))

        # Band power extraction
        band_powers = {}
        total_power = 0
        for name, (lo, hi) in band_defs.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if hasattr(np, 'trapezoid'):
                power = np.trapezoid(psd[mask], freqs[mask])
            else:
                power = float(np.sum(psd[mask]) * (freqs[mask][-1] - freqs[mask][0]) / max(len(freqs[mask]) - 1, 1))
            band_powers[name] = power
            total_power += power

        # Relative power (normalize)
        for name in band_defs:
            band_powers[f"{name}_rel"] = band_powers[name] / max(total_power, 1e-8)

        # Ratios (clinically meaningful)
        band_powers["alpha_beta_ratio"] = band_powers["alpha"] / max(band_powers["beta"], 1e-8)
        band_powers["theta_alpha_ratio"] = band_powers["theta"] / max(band_powers["alpha"], 1e-8)
        band_powers["theta_beta_ratio"] = band_powers["theta"] / max(band_powers["beta"], 1e-8)

        features.append(band_powers)

    return pd.DataFrame(features)


# === 3. Run the demo ===
print("=" * 60)
print("WavqWise EEG Signal Classification Demo")
print("Sense. Forecast. Alert.")
print("=" * 60)

# Generate data
print("\n[1/5] Generating synthetic EEG dataset...")
signals, labels, t = generate_eeg_dataset(n_samples=300, duration=4, sample_rate=256)
label_names = {0: "Relaxed", 1: "Focused", 2: "Drowsy"}
print(f"  Epochs: {len(signals)}")
print(f"  Duration: 4s per epoch @ 256 Hz")
print(f"  Classes: {', '.join(label_names.values())}")
for cls in range(3):
    print(f"    {label_names[cls]}: {np.sum(labels == cls)} epochs")

# WavqWise SignalPipeline for quick analysis
print("\n[2/5] WavqWise SignalPipeline band analysis (single epoch)...")
sig = SignalPipeline()
sig._data = signals[0]
sig._sample_rate = 256
sig.filter(low=1, high=50, notch=50)
bands = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
print("  Sample epoch (Relaxed) band power:")
for name, power in bands.bands.items():
    print(f"    {name}: {power:.4f} uV^2")

# Extract features for all epochs
print("\n[3/5] Extracting band power features for all epochs...")
features = extract_features(signals, sample_rate=256)
print(f"  Feature matrix: {features.shape}")
print(f"  Features: {list(features.columns)}")

# Train classifier
print("\n[4/5] Training RandomForest classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {accuracy:.1%}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_names.values())))

# Feature importance
print("[5/5] Top features by importance:")
importances = pd.Series(clf.feature_importances_, index=features.columns)
top_features = importances.nlargest(5)
for feat, imp in top_features.items():
    print(f"    {feat}: {imp:.4f}")

# === 4. Visualizations ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("WavqWise EEG Classification Results", fontsize=14, fontweight="bold")

# Plot sample signals
ax = axes[0, 0]
for cls, color, label in [(0, "#059669", "Relaxed"), (1, "#2563eb", "Focused"), (2, "#d97706", "Drowsy")]:
    idx = np.where(labels == cls)[0][0]
    ax.plot(t[:512], signals[idx][:512], color=color, alpha=0.8, linewidth=0.8, label=label)
ax.set_title("Sample EEG Epochs (2s)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (uV)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Band power comparison
ax = axes[0, 1]
band_names = ["delta", "theta", "alpha", "beta", "gamma"]
x_pos = np.arange(len(band_names))
width = 0.25
for i, (cls, color, label) in enumerate([(0, "#059669", "Relaxed"), (1, "#2563eb", "Focused"), (2, "#d97706", "Drowsy")]):
    mask = labels == cls
    mean_powers = features.loc[mask, band_names].mean()
    ax.bar(x_pos + i*width, mean_powers, width, color=color, alpha=0.8, label=label)
ax.set_title("Mean Band Power by Class")
ax.set_xticks(x_pos + width)
ax.set_xticklabels([b.capitalize() for b in band_names])
ax.set_ylabel("Power (uV^2)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Feature importance
ax = axes[1, 0]
top_10 = importances.nlargest(10)
colors = ["#dc2626" if "ratio" in f else "#2563eb" for f in top_10.index]
ax.barh(range(len(top_10)), top_10.values, color=colors, alpha=0.8)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10.index, fontsize=9)
ax.set_title("Feature Importance (Top 10)")
ax.set_xlabel("Importance")
ax.grid(True, alpha=0.3, axis="x")

# PSD for each class
ax = axes[1, 1]
for cls, color, label in [(0, "#059669", "Relaxed"), (1, "#2563eb", "Focused"), (2, "#d97706", "Drowsy")]:
    idx = np.where(labels == cls)[0][0]
    freqs, psd = welch(signals[idx], fs=256, nperseg=256)
    ax.semilogy(freqs[:60], psd[:60], color=color, alpha=0.8, linewidth=1.5, label=label)
ax.axvspan(8, 13, alpha=0.1, color="green", label="Alpha band")
ax.axvspan(13, 30, alpha=0.1, color="blue", label="Beta band")
ax.set_title("Power Spectral Density")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (uV^2/Hz)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

plt.tight_layout()
plt.savefig("demos/eeg_classification_results.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved: demos/eeg_classification_results.png")
print(f"\nAccuracy: {accuracy:.1%} on 3-class EEG classification")
print("Done.")
