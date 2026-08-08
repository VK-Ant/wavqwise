"""
WavqWise Demo: EEG Signal Classification (REAL DATA)
=====================================================
Dataset: MNE Sample Dataset - Real auditory/visual MEG+EEG recording
Subject: Real human participant, auditory+visual stimuli
Source: https://mne.tools/stable/overview/datasets_index.html

Classification: Auditory vs Visual evoked responses
Pipeline: WavqWise SignalPipeline -> Band Power Features -> RandomForest

Requirements: pip install wavqwise[signals] mne scikit-learn
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from wavqwise import SignalPipeline


def load_real_eeg_data():
    """Load REAL EEG data from MNE sample dataset."""
    try:
        import mne
        print("  Loading MNE sample dataset (real EEG recording)...")
        print("  Source: Auditory/Visual evoked response experiment")
        print("  (First run downloads ~1.5GB, cached after that)\n")

        # Load sample dataset
        data_path = mne.datasets.sample.data_path()
        raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

        # Pick EEG channels only
        raw.pick_types(meg=False, eeg=True, eog=False, stim=False, verbose=False)
        print(f"  EEG channels: {len(raw.ch_names)}")
        print(f"  Channels: {', '.join(raw.ch_names[:8])}...")
        print(f"  Sample rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")

        # Find events
        events_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw-eve.fif")
        events = mne.read_events(events_fname, verbose=False)

        # Event IDs: 1=auditory/left, 2=auditory/right, 3=visual/left, 4=visual/right
        # Simplify: auditory (1,2) vs visual (3,4)
        event_id = {"auditory": 1, "visual": 3}

        # Filter events to only auditory/left and visual/left
        mask = np.isin(events[:, 2], [1, 3])
        events_filtered = events[mask]

        # Create epochs
        epochs = mne.Epochs(
            raw, events_filtered, event_id=event_id,
            tmin=-0.2, tmax=0.5, baseline=(None, 0),
            preload=True, verbose=False
        )
        epochs.drop_bad(verbose=False)

        print(f"  Epochs extracted: {len(epochs)}")
        print(f"  Auditory epochs: {len(epochs['auditory'])}")
        print(f"  Visual epochs: {len(epochs['visual'])}")
        print(f"  Epoch duration: {epochs.tmin}s to {epochs.tmax}s")

        # Get data as numpy arrays
        X_auditory = epochs["auditory"].get_data()  # (n_epochs, n_channels, n_times)
        X_visual = epochs["visual"].get_data()

        labels = np.concatenate([
            np.zeros(len(X_auditory)),
            np.ones(len(X_visual))
        ])
        data = np.concatenate([X_auditory, X_visual], axis=0)

        return data, labels, epochs.info["sfreq"], raw.ch_names, epochs

    except ImportError:
        print("  MNE not installed. Install: pip install mne")
        print("  Falling back to PhysioNet BCI dataset...")
        return load_physionet_eeg()
    except Exception as e:
        print(f"  MNE dataset error: {e}")
        print("  Falling back to PhysioNet BCI dataset...")
        return load_physionet_eeg()


def load_physionet_eeg():
    """Fallback: Load PhysioNet EEG Motor Imagery dataset via MNE."""
    try:
        import mne
        from mne.datasets import eegbci
        from mne.io import concatenate_raws, read_raw_edf

        print("  Loading PhysioNet EEG Motor Imagery (real BCI data)...")
        print("  Source: https://physionet.org/content/eegmmidb/1.0.0/")

        subject = 1
        runs = [6, 10, 14]  # Motor imagery: left fist vs right fist

        raw_fnames = eegbci.load_data(subject, runs, verbose=False)
        raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = concatenate_raws(raws)

        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        raw.filter(1, 40, verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # T1=left fist, T2=right fist
        epochs = mne.Epochs(
            raw, events, event_id=dict(T1=event_id.get("T1", 2), T2=event_id.get("T2", 3)),
            tmin=0, tmax=4.0, baseline=None, preload=True, verbose=False
        )
        epochs.drop_bad(verbose=False)

        data = epochs.get_data()
        labels = epochs.events[:, 2]
        labels = (labels == labels.max()).astype(int)  # Binary: 0/1

        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Epochs: {len(epochs)}")
        print(f"  Class 0: {np.sum(labels==0)}, Class 1: {np.sum(labels==1)}")

        return data, labels, raw.info["sfreq"], raw.ch_names[:len(data[0])], None

    except Exception as e:
        print(f"  PhysioNet also failed: {e}")
        raise RuntimeError("No real EEG data available. Install mne: pip install mne")


def extract_band_features(data, sfreq, use_wavqwise=True):
    """Extract frequency band power features from EEG epochs."""
    from scipy.signal import welch

    band_defs = {
        "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 50),
    }

    all_features = []
    for epoch in data:
        epoch_features = {}

        # Average across channels or process per channel
        if epoch.ndim == 2:
            n_channels = epoch.shape[0]
            for ch_idx in range(min(n_channels, 8)):  # Limit to 8 channels
                signal = epoch[ch_idx]
                freqs, psd = welch(signal, fs=sfreq, nperseg=min(int(sfreq), len(signal)))

                for band_name, (lo, hi) in band_defs.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    if mask.any():
                        if hasattr(np, "trapezoid"):
                            power = np.trapezoid(psd[mask], freqs[mask])
                        else:
                            power = float(np.sum(psd[mask]) * (freqs[mask][-1] - freqs[mask][0]) / max(len(freqs[mask]) - 1, 1))
                        epoch_features[f"ch{ch_idx}_{band_name}"] = power

            # Global features (averaged across channels)
            avg_signal = epoch.mean(axis=0)
            freqs, psd = welch(avg_signal, fs=sfreq, nperseg=min(int(sfreq), len(avg_signal)))
            total_power = 0
            for band_name, (lo, hi) in band_defs.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    if hasattr(np, "trapezoid"):
                        power = np.trapezoid(psd[mask], freqs[mask])
                    else:
                        power = float(np.sum(psd[mask]) * (freqs[mask][-1] - freqs[mask][0]) / max(len(freqs[mask]) - 1, 1))
                    epoch_features[f"avg_{band_name}"] = power
                    total_power += power

            # Relative power
            for band_name in band_defs:
                key = f"avg_{band_name}"
                if key in epoch_features:
                    epoch_features[f"avg_{band_name}_rel"] = epoch_features[key] / max(total_power, 1e-10)

            # Ratios
            alpha = epoch_features.get("avg_alpha", 1e-10)
            beta = epoch_features.get("avg_beta", 1e-10)
            theta = epoch_features.get("avg_theta", 1e-10)
            epoch_features["alpha_beta_ratio"] = alpha / max(beta, 1e-10)
            epoch_features["theta_alpha_ratio"] = theta / max(alpha, 1e-10)

        else:
            # 1D signal
            freqs, psd = welch(epoch, fs=sfreq, nperseg=min(int(sfreq), len(epoch)))
            for band_name, (lo, hi) in band_defs.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    if hasattr(np, "trapezoid"):
                        epoch_features[band_name] = np.trapezoid(psd[mask], freqs[mask])
                    else:
                        epoch_features[band_name] = float(np.sum(psd[mask]))

        all_features.append(epoch_features)

    return pd.DataFrame(all_features).fillna(0)


# === Run Demo ===
print("=" * 60)
print("WavqWise EEG Classification - REAL DATA")
print("Sense. Forecast. Alert.")
print("=" * 60)

# Load real data
print("\n[1/5] Loading real EEG dataset...")
data, labels, sfreq, ch_names, epochs_obj = load_real_eeg_data()

label_names = {0: "Auditory", 1: "Visual"}
print(f"\n  Total epochs: {len(data)}")
print(f"  Sample rate: {sfreq} Hz")
for cls in [0, 1]:
    print(f"  {label_names[cls]}: {np.sum(labels == cls)} epochs")

# WavqWise analysis on single epoch
print("\n[2/5] WavqWise SignalPipeline analysis (single epoch)...")
sig = SignalPipeline()
if data[0].ndim == 2:
    sig._data = data[0]  # (channels, timepoints)
    sig._channels = ch_names[:data[0].shape[0]]
else:
    sig._data = data[0]
sig._sample_rate = int(sfreq)
bands = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
for name, power in bands.bands.items():
    print(f"  {name}: {power:.6f}")

# Extract features
print("\n[3/5] Extracting band power features...")
features = extract_band_features(data, sfreq)
print(f"  Feature matrix: {features.shape}")

# Classification
print("\n[4/5] Training classifiers...")
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
}

results = {}
for name, clf in classifiers.items():
    if name == "SVM":
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "predictions": y_pred}
    print(f"  {name}: {acc:.1%}")

# Best model report
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_pred = results[best_name]["predictions"]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=list(label_names.values())))

# Visualization
print("[5/5] Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("WavqWise EEG Classification - Real Data", fontsize=14, fontweight="bold")

# Sample epochs
ax = axes[0, 0]
for cls, color in [(0, "#2563eb"), (1, "#dc2626")]:
    idx = np.where(labels == cls)[0][0]
    if data[idx].ndim == 2:
        signal = data[idx].mean(axis=0)
    else:
        signal = data[idx]
    t = np.arange(len(signal)) / sfreq
    ax.plot(t, signal, color=color, alpha=0.7, linewidth=0.8, label=label_names[cls])
ax.set_title("Sample Epochs (avg across channels)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True, alpha=0.3)

# Band power comparison
ax = axes[0, 1]
band_cols = [c for c in features.columns if c.startswith("avg_") and "_rel" not in c and "ratio" not in c]
if band_cols:
    x_pos = np.arange(len(band_cols))
    width = 0.35
    for i, (cls, color) in enumerate([(0, "#2563eb"), (1, "#dc2626")]):
        mask = labels == cls
        means = features.loc[mask, band_cols].mean()
        ax.bar(x_pos + i * width, means, width, color=color, alpha=0.8, label=label_names[cls])
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([c.replace("avg_", "").capitalize() for c in band_cols], fontsize=9)
    ax.set_title("Mean Band Power by Class")
    ax.set_ylabel("Power")
    ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Model comparison
ax = axes[1, 0]
model_names = list(results.keys())
accs = [results[m]["accuracy"] for m in model_names]
colors = ["#059669" if m == best_name else "#94a3b8" for m in model_names]
ax.barh(model_names, accs, color=colors)
ax.set_xlim(0, 1)
ax.set_title("Model Comparison (Accuracy)")
for i, acc in enumerate(accs):
    ax.text(acc + 0.01, i, f"{acc:.1%}", va="center")
ax.grid(True, alpha=0.3, axis="x")

# Confusion matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, best_pred)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(list(label_names.values()))
ax.set_yticklabels(list(label_names.values()))
ax.set_title(f"Confusion Matrix ({best_name})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("demos/eeg_real_classification_results.png", dpi=150, bbox_inches="tight")
print(f"  Plot saved: demos/eeg_real_classification_results.png")
print("\nDone.")
