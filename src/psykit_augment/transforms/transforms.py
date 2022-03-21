from typing import Optional, List

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


class Transform(BaseEstimator, TransformerMixin):
    def fit(self, batch: np.ndarray, labels: Optional[np.ndarray] = None):
        return self


class ButterWorthFilter(Transform):
    def __init__(self, sampling_rate: int, order: int, highpass: int, lowpass: int):
        self.sampling_rate = sampling_rate
        self.order = order
        self.highpass = highpass
        self.lowpass = lowpass

        normal_cutoff = tuple(
            a / (0.5 * self.sampling_rate) for a in (self.highpass, self.lowpass)
        )
        self.filter = signal.butter(self.order, normal_cutoff, btype="bandpass")

    def transform(self, batch: np.ndarray) -> List[np.ndarray]:
        out = np.empty_like(batch)
        out[:] = [signal.filtfilt(*self.filter, item) for item in batch]
        return out


class SignalDecimator(Transform):
    def __init__(self, factor: int):
        self.factor = factor

    def transform(self, batch: np.ndarray) -> List[np.ndarray]:
        out = np.empty(len(batch), dtype=np.object)
        out[:] = [signal.decimate(item, self.factor) for item in batch]
        return out


class ChannelScaler(Transform):
    def __init__(self, scaler: Transform):
        self.scaler = scaler

    def fit(self, batch: np.ndarray, labels: Optional[np.ndarray] = None):
        for signals in batch:
            self.scaler.partial_fit(signals.T)
        return self

    def transform(self, batch: np.ndarray) -> List[np.ndarray]:
        scaled = np.empty_like(batch)
        for i, signals in enumerate(batch):
            scaled[i] = self.scaler.transform(signals.T).T
        return scaled


class MarkersTransformer(Transform):
    def __init__(
        self, labels_mapping: dict, decimation_factor: int = 1, empty_label: float = 0.0
    ):
        self.labels_mapping = labels_mapping
        self.decimation_factor = decimation_factor
        self.empty_label = empty_label

    def transform(self, batch: np.ndarray) -> List[np.ndarray]:
        res = []

        for markers in batch:
            index_label = []
            for index, label in enumerate(markers):
                if label == self.empty_label:
                    continue

                new_index = index // self.decimation_factor
                new_label = self.labels_mapping[label]
                index_label.append((new_index, new_label))

            res.append(np.array(index_label, dtype=np.int))

        return res
