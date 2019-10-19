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

    X = Flatten()(X)
    X = Dense(2, activation="sigmoid", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet9")

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=0.001),
        metrics=["binary_accuracy"],
    )
    return model


try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()

import numpy as np

from sklearn.model_selection import train_test_split
from pprint import pprint
from tqdm import tqdm

print("[MODEL]")

THRESHOLD = 0.8  # FIXME: in future linear
NUM_ROUNDS = 100 * 50000
SHAPE = (9, 9, 7)
MODEL_PATH = "treeboost.txt"
# FIXME: rotate in all directions
# FIXME: zapisywanie datasetu


def get_dataset():
    global THRESHOLD, SHAPE

    kanapki = []

    from dataset import Data

    d = Data(2018)

    a = d.load_dataset()
    print(a.shape)

    for x, y in a:
        kanapki.append(
            Kanapka(features=x[:, :, 0:], label=np.nanmean(y[:, :, 0])))

    return Dataset(kanapki=kanapki)


class Kanapka:
    # FIXME: TO MODYFIKUJE KAMIL
    def __init__(self, features=None, label=None):
        self.features = features
        self.label = label

    def fake(self):
        # 9x9 (3 features)
        self.features = np.random.rand(9, 9, 3)
        self.label = np.random.randint(0, 10)
        self.features[0, 0, 0] *= self.label
        self.features[0, 0, 1] -= self.label
        self.features[0, 0, 2] += self.label

    def get(self):
        x = self.features
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)

        x = (x - x_min) / (x_max - x_min)
        norm = x
        # pprint(norm)
        # FIXME: nadal SA NANY????????????/
        where_are_NaNs = np.isnan(norm)
        norm[where_are_NaNs] = 0

        # FIXME: tutaj manipulujemy wejsciem/wyjsciem
        # FIXME: mhmh, a moze by tak bez chlorofilu? (autoencoding)
        return norm, self.label


class Dataset:
    X = []
    y = []

    def __init__(self, kanapki=None):
        if kanapki is None:
            self.fake()  # FIXME: lista kanapek
        else:
            for a in kanapki:
                f, l = a.get()
                self.X.append(f)
                self.y.append(l)
        self.normalize()

    def fake(self, N=2000):
        for _ in range(N):
            a = Kanapka()
            a.fake()
            f, l = a.get()
            self.X.append(f)
            self.y.append(l)

    def normalize(self):
        global THRESHOLD

        y_onehot = []
        for e in self.y:
            if e > THRESHOLD:
                y_onehot.append([1, 0])
            else:
                y_onehot.append([0, 1])
        self.y = y_onehot
        # pprint(self.y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        # binaryzujemy wynik
        # self.y = np.where(self.y > THRESHOLD, 0, 1)

    def train_and_test(self):
        return train_test_split(self.X,
                                self.y,
                                test_size=0.33,
                                random_state=42)


class Model2d:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.train_and_test()
        batch_size = 128
        epochs = 100
        model = ResNet9(*SHAPE)
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
        )
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        model.save("keras.h5")
        self.load()

    def load(self):
        from tensorflow.keras.models import load_model

        self.model = load_model("keras.h5")

    def predict(self, arr):
        return self.model.predict(arr)


if __name__ == "__main__":
    # dataset = Dataset(kanapki=get_dataset())
    dataset = get_dataset()
    model = Model2d(dataset=dataset)
    model.train()

    for i in range(100):
        pred = model.predict([dataset.X[i]])
        print(f"PRED {pred} | {dataset.y[i]}")

    lgb.plot_importance(model.pst)
