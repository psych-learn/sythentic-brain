from typing import Callable, Union, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

DatasetType = Union[np.ndarray, tf.data.Dataset]
LossType = Union[str, Callable, Loss]
AcceptableMetric = Union[str, Callable, Metric]
MetricsType = Union[
    List[AcceptableMetric], List[List[AcceptableMetric]], Dict[str, AcceptableMetric]
]


