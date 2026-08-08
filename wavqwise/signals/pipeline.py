"""Signal processing pipeline for EEG and time-series signals."""
import numpy as np
import pandas as pd

class SignalResult:
    def __init__(self, data, bands=None, events=None, sample_rate=256):
        self.data = data
        self.bands = bands or {}
        self.events = events or []
        self.sample_rate = sample_rate

    def plot_bands(self):
        import matplotlib.pyplot as plt
        if self.bands:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(self.bands.keys(), self.bands.values(), color=["#7c3aed","#2563eb","#059669","#d97706","#dc2626"])
            ax.set_title("Band Power"); ax.set_ylabel("Power (uV^2)")
            plt.tight_layout(); plt.show(); return fig

    def plot_spectrogram(self):
        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(self.data, fs=self.sample_rate)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.pcolormesh(t, f, 10*np.log10(Sxx+1e-10), shading="gouraud", cmap="viridis")
        ax.set_ylabel("Frequency (Hz)"); ax.set_xlabel("Time (s)")
        ax.set_title("Spectrogram"); plt.tight_layout(); plt.show(); return fig

    def plot_psd(self):
        import matplotlib.pyplot as plt
        from scipy.signal import welch
        f, psd = welch(self.data, fs=self.sample_rate)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(f, psd, color="#2563eb")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
        ax.set_title("Power Spectral Density"); plt.tight_layout(); plt.show(); return fig


class SignalPipeline:
    def __init__(self):
        self._data = None
        self._channels = None
        self._sample_rate = 256

    def load(self, source, channels=None, sample_rate=256, **kwargs):
        if isinstance(source, np.ndarray):
            self._data = source
        elif isinstance(source, str):
            df = pd.read_csv(source)
            self._channels = channels or [c for c in df.columns if c not in ["time","timestamp"]]
            self._data = df[self._channels].values.T if len(self._channels) > 1 else df[self._channels[0]].values
        self._sample_rate = sample_rate
        return self

    def filter(self, low=1, high=50, notch=None):
        from scipy.signal import butter, filtfilt, iirnotch
        b, a = butter(4, [low, high], btype="band", fs=self._sample_rate)
        if self._data.ndim == 1:
            self._data = filtfilt(b, a, self._data)
        else:
            self._data = np.array([filtfilt(b, a, ch) for ch in self._data])
        if notch:
            b, a = iirnotch(notch, 30, self._sample_rate)
            if self._data.ndim == 1:
                self._data = filtfilt(b, a, self._data)
            else:
                self._data = np.array([filtfilt(b, a, ch) for ch in self._data])
        return self

    def extract_bands(self, bands=None):
        from scipy.signal import welch
        default_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
        bands = {b: default_bands[b] for b in (bands or default_bands.keys())}
        signal = self._data if self._data.ndim == 1 else self._data.mean(axis=0)
        f, psd = welch(signal, fs=self._sample_rate)
        result = {}
        for name, (lo, hi) in bands.items():
            mask = (f >= lo) & (f <= hi)
            result[name] = np.trapezoid(psd[mask], f[mask]) if hasattr(np, 'trapezoid') else float(np.sum(psd[mask]) * (f[mask][-1] - f[mask][0]) / max(len(f[mask]) - 1, 1))
        return SignalResult(signal, bands=result, sample_rate=self._sample_rate)

    def detect_events(self, method="auto", threshold=3.0):
        signal = self._data if self._data.ndim == 1 else self._data.mean(axis=0)
        mean, std = np.mean(signal), np.std(signal)
        events = np.where(np.abs(signal - mean) > threshold * std)[0]
        return SignalResult(signal, events=events.tolist(), sample_rate=self._sample_rate)

    def plot_channels(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(max(1, self._data.shape[0] if self._data.ndim > 1 else 1), 1, figsize=(12, 8), sharex=True)
        if self._data.ndim == 1:
            axes = [axes]
            data = [self._data]
        else:
            data = self._data
        t = np.arange(len(data[0])) / self._sample_rate
        for i, (ax, ch) in enumerate(zip(axes if hasattr(axes, '__iter__') else [axes], data)):
            ax.plot(t, ch, linewidth=0.5, color="#2563eb")
            label = self._channels[i] if self._channels and i < len(self._channels) else f"Ch {i}"
            ax.set_ylabel(label)
        axes[-1].set_xlabel("Time (s)") if hasattr(axes, '__iter__') else axes.set_xlabel("Time (s)")
        plt.suptitle("EEG Channels"); plt.tight_layout(); plt.show(); return fig
