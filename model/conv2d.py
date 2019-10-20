import numpy as np
from tensorflow.keras.models import load_model
from model.resnet9 import ResNet9


class ModelConv2d:
    SHAPE = (9, 9, 7)
    MODEL_PATH = "covn2dkeras.txt"
    BATCH_SIZE = 128
    EPOCHS = 100

    def __init__(self, dataset=None):
        self.dataset = dataset

    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.train_and_test()
        model = ResNet9(*self.SHAPE)
        model.fit(
            X_train,
            y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            verbose=1,
            validation_data=(X_test, y_test),
        )
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        model.save(self.MODEL_PATH)
        self.load()

    def load(self):
        self.model = load_model(self.MODEL_PATH)

    def predict(self, arr):
        return self.model.predict(arr)

    @staticmethod
    def get_input(x):
        return x[:, :, 0:]

    @staticmethod
    def get_output(y):
        return np.nanmean(y[:, :, 0])

    @staticmethod
    def get(cls):
        x = cls.features  # FIXME (1, 2)
        x_min = x.min(axis=(2), keepdims=True)
        x_max = x.max(axis=(2), keepdims=True)

        x = (x - x_min) / (x_max - x_min)
        norm = x
        where_are_NaNs = np.isnan(norm)
        norm[where_are_NaNs] = 0
        return norm, cls.label

    @staticmethod
    def normalize(cls):
        y_onehot = []
        for e in cls.y:
            if e > cls.threshold:
                y_onehot.append([1, 0])
            else:
                y_onehot.append([0, 1])
        cls.y = y_onehot
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
