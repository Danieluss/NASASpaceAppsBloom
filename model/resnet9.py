import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    BatchNormalization,
    Activation,
    Add,
    MaxPool2D,
    GlobalAvgPool2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.keras.optimizers import Adam
import os


def bottleneck_residual_block(X, kernel_size, filters, reduce=False, s=2):
    F1, F2 = filters

    X_shortcut = X

    if reduce:
        X_shortcut = Conv2D(filters=F2,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding="same")(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        X = Conv2D(filters=F1,
                   kernel_size=(3, 3),
                   strides=(s, s),
                   padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

    else:
        X = Conv2D(filters=F1,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("relu")(X)

    X = Conv2D(filters=F2,
               kernel_size=kernel_size,
               strides=(1, 1),
               padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def ResNet9(x_size, y_size, features):
    X_input = Input((x_size, y_size, features))

    X = Conv2D(32, (5, 5), strides=(2, 2), name="conv1",
               padding="same")(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPool2D((3, 3), strides=(2, 2), padding="same")(X)

    X = bottleneck_residual_block(X, 2, [32, 32])

    X = bottleneck_residual_block(X, 2, [64, 64], reduce=True, s=2)

    X = GlobalAvgPool2D()(X)

    X = Flatten()(X)  # sigmoid/softmax
    X = Dense(2, activation="sigmoid", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet9")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.0001),
        metrics=["binary_accuracy"],
    )
    return model
