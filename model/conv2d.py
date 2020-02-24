import numpy as np
from tensorflow.keras.models import load_model
from model.resnet9 import ResNet9
import pickle


class ModelConv2d:
    SHAPE = (9, 9, 7)
    MODEL_PATH = "conv2dkeras"
    BATCH_SIZE = 512
    EPOCHS = 20

    def __init__(self):
        pass

    def __init__(self, dataset=None):
        self.dataset = dataset

    def save_vector(self, name, value):
        with open(self.MODEL_PATH + "/" + name + ".pickle", "wb") as f:
            pickle.dump(value, f)

    def load_vector(self, name):
        with open(self.MODEL_PATH + "/" + name + ".pickle", "rb") as f:
            value = pickle.load(f)
        return value

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
        self.save_vector("mi", self.mi)
        self.save_vector("sigma", self.sigma)
        self.load()

    def load(self):
        self.model = load_model(self.MODEL_PATH)
        self.mi = self.load_vector("mi")
        self.sigma = self.load_vector("sigma")

    def predict(self, arr):
        return self.model.predict(arr)

    def predict_01(self, arr):
        pred_01 = []
        pred = self.model.predict(arr)
        for x1, x2 in pred:
            if x1 > x2:
                pred_01.append(True)
            else:
                pred_01.append(False)
        return pred_01

    @staticmethod
    def get_input(x):
        return x[:, :, 0:]

    @staticmethod
    def get_output(y):
        return y[4, 4, 0]

    def get(self, cls):
        x = cls.features  # FIXME (1, 2)
        # print(x.shape)
        # print(np.nanmean(x[:,:,0]), cls.label)
        # print(x.shape)
        x = (x-self.mi)/self.sigma
        # print(np.nanmean(x[:,:,0]))
        # x_min = x.min(axis=(2), keepdims=True)
        # x_max = x.max(axis=(2), keepdims=True)

        # x = (x - x_min) / (x_max - x_min)
        # print(x.shape)
        norm = x
        where_are_NaNs = np.isnan(norm)
        norm[where_are_NaNs] = 0
        return norm, cls.label

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def normalize(cls):
        y_onehot = []
        for e in cls.y:
            y0 = ModelConv2d.sigmoid(3*(e-cls.threshold))
            # if e > cls.threshold:
            #     y_onehot.append([1, 0])
            # else:
            #     y_onehot.append([0, 1])
            y_onehot.append([y0, 1-y0])
        cls.y = y_onehot
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
