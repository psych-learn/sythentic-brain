from typing import Optional, Union

import tensorflow as tf
from keras.layers import SeparableConv2D, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Dropout, SpatialDropout2D, Activation, \
    AveragePooling2D


class SpatialEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            num_filters: int,
            depth_multiplier: int,
            kernel_length: int,
            nb_channels: int,
            nb_samples: int,
            dropout_layer: Union[Dropout, SpatialDropout2D],
            dropout_rate: float = 0.2
    ):
        """

        :param num_filters:
        :param depth_multiplier:
        :param kernel_length:
        :param nb_channels:
        :param nb_samples:
        :param dropout_layer:
        :param dropout_rate:
        """
        super().__init__()

        self.conv2d = Conv2D(num_filters, (1, kernel_length), padding='same', input_shape=(nb_channels, nb_samples, 1),
                             use_bias=False)
        self.batch_norm1 = BatchNormalization()
        self.depthwise_conv = DepthwiseConv2D((nb_channels, 1), use_bias=False,
                                              depth_multiplier=depth_multiplier,
                                              depthwise_constraint=max_norm(1.))
        self.batch_norm2 = BatchNormalization()
        self.activation = Activation('elu')
        self.pooling = AveragePooling2D((1, 4))

        self.dropout = dropout_layer(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.conv2d(inputs)
        x = self.batch_norm1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x


class TemporalEncoder(tf.keras):
    def __init__(
            self,
            num_filters: int,
            dropout_layer: Union[Dropout, SpatialDropout2D],
            dropout_rate: float = 0.2
    ):
        """

        :param num_filters:
        :param dropout_layer:
        :param dropout_rate:
        """
        super().__init__()
        self.separable_conv = SeparableConv2D(num_filters, (1, 16), use_bias=False, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.activation = Activation('elu')
        self.pooling = AveragePooling2D((1, 8))
        self.dropout = dropout_layer(dropout_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.separable_conv(inputs)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)

        return x


class EEGNetPlus(tf.keras.layers.Layer):
    def __init__(
            self,
            nb_classes: int,
            nb_channels: int = 64,
            nb_samples: int = 128,
            dropout_rate: float = 0.5,
            nb_spatial_filters: int = 8,
            nb_temporal_filters: Optional[int] = None,
            depth_multiplier: int = 2,
            dropout_type: str = "Dropout",
            kernel_length: Optional[int] = None,
            fs: Optional[float] = None,
            *args,
            **kwargs
    ) -> None:
        """
        EEGNet+ Implementation

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.

        :param nb_classes: number of classes to classify
        :param nb_channels: number of channels in the EEG data
        :param nb_samples: number of time points in the EEG data
        :param dropout_rate: percentage of dropout
        :param fs, kernel_length: length of temporal convolution in first layer. We found
                                that setting this to be half the sampling rate worked
                                well in practice. For the SMR dataset in particular
                                since the data was high-passed at 4Hz we used a kernel
                                length of 32.
                                If sample frequency is passed, then kernel_length = fs // 2
        :param nb_spatial_filters, nb_temporal_filters: number of temporal filters and number of pointwise filters to learn. Default: F1 = 8, F2 = F1 * D.
        :param depth_multiplier: number of spatial filters to learn within each temporal convolution. Default: D = 2
        :param dropout_type: Either SpatialDropout2D or Dropout, passed as a string

        References:
        ---
        [1]. https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

        """
        super.__init__(*args, **kwargs)
        if dropout_type == 'SpatialDropout2D':
            dropout_layer = SpatialDropout2D
        elif dropout_type == 'Dropout':
            dropout_layer = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        kernel_length = kernel_length if kernel_length else fs // 2
        self.nb_temporal_filters = depth_multiplier * nb_spatial_filters if nb_temporal_filters is None else nb_temporal_filters

        self.spatial_encoder = SpatialEncoder(nb_spatial_filters, depth_multiplier, kernel_length, nb_channels,
                                              nb_samples, dropout_layer, dropout_rate)
        self.temporal_encoder = TemporalEncoder(nb_temporal_filters, dropout_layer, dropout_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.spatial_encoder(inputs)
        x = self.temporal_encoder(x)

        return x


class EEGNetPlusClassifier(tf.keras.Model):
    def __init__(
            self,
            nb_classes: int,
            nb_channels: int = 64,
            nb_samples: int = 128,
            dropout_rate: float = 0.5,
            nb_spatial_filters: int = 8,
            nb_temporal_filters: Optional[int] = None,
            depth_multiplier: int = 2,
            dropout_type: str = "Dropout",
            kernel_length: Optional[int] = None,
            fs: Optional[float] = None,
            *args,
            **kwargs
    ):
        super().__init__()
        self.eegnet_plus = EEGNetPlus()
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(nb_classes, name='dense')
        self.classification_head = Activation('softmax', name="softmax")

    def call(self, inputs, *args, **kwargs):
        x = self.eegnet_plus(inputs)
        x = self.flatten(x)
        x = self.dense(x)

        return self.classification_head(x)


class EEGNetPlusRegressor(tf.keras.Model):
    def __init__(
            self,
            nb_classes: int,
            nb_channels: int = 64,
            nb_samples: int = 128,
            dropout_rate: float = 0.5,
            nb_spatial_filters: int = 8,
            nb_temporal_filters: Optional[int] = None,
            depth_multiplier: int = 2,
            dropout_type: str = "Dropout",
            kernel_length: Optional[int] = None,
            fs: Optional[float] = None,
            num_output_units: int = 100,
            *args,
            **kwargs
    ):
        super().__init__()
        self.eegnet_plus = EEGNetPlus()
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(activation='linear', units=num_output_units)
        self.regression_head = Dense(activation='linear', units = 1, name="regression_head")

    def call(self, inputs, *args, **kwargs):
        x = self.eegnet_plus(inputs)
        x = self.flatten(x)
        x = self.dense(x)

        return self.regression_head(x)
