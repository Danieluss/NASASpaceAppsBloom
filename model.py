try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from pprint import pprint
from tqdm import tqdm

print("[MODEL]")

THRESHOLD = 1  # FIXME: in future linear
NUM_ROUNDS = 100 * 50000
MODEL_PATH = "treeboost.txt"
PARAMS = {
    "boosting_type": "gbdt",
    # "objective": "regression",
    # "metric": "l2",
    "objective": "binary",
    "metric": ["binary_error", "binary_logloss"],
    "is_unbalance": True,
    "feature_fraction": 0.85,
    "learning_rate": 0.005,
    "verbose": -1,
    "min_split_gain": 0.1,
    "reg_alpha": 0.3,
    "max_bin": 512,  # 512*10 FIXME/RESEARCH
    "num_leaves": 32,  # 32*10 FIXME/RESEARCH
    "max_depth": 9,
    "min_child_weight": 0.5,
    "is_training_metric": "True",
}


def get_dataset():
    kanapki = []

    from features import FeaturesExtractor

    f = FeaturesExtractor(200910)

    for _ in tqdm(range(1000)):
        a = f.get_grid(np.random.randint(-50, 50), np.random.randint(-50, 50))
        if not np.isnan(a[:, :, 0].mean()):
            print("-->", a[:, :, 0].mean())
            kanapki.append(
                Kanapka(features=a[:, :, 1:], label=a[:, :, 0].mean()))

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
        # FIXME: tutaj manipulujemy wejsciem/wyjsciem
        # FIXME: mhmh, a moze by tak bez chlorofilu? (autoencoding)
        return self.features.flatten(), self.label


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
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        # binaryzujemy wynik
        self.y = np.where(self.y > THRESHOLD, 0, 1)

    def train_and_test(self):
        return train_test_split(self.X,
                                self.y,
                                test_size=0.33,
                                random_state=42)


class Model:
    def __init__(self, dataset=None):
        X_train, X_test, y_train, y_test = dataset.train_and_test()
        self.lgb_train = lgb.Dataset(X_train, y_train)
        self.lgb_test = lgb.Dataset(X_test, y_test)

    def train(self):
        global MODEL_PATH, PARAMS, NUM_ROUNDS
        gbm = lgb.train(
            PARAMS,
            self.lgb_train,
            num_boost_round=NUM_ROUNDS,
            valid_sets=self.lgb_test,
            early_stopping_rounds=NUM_ROUNDS / 10000,
        )
        gbm.save_model(MODEL_PATH)
        self.load()

    def load(self):
        global MODEL_PATH
        self.pst = lgb.Booster(model_file=MODEL_PATH)

    def predict(self, arr):
        return self.pst.predict(arr)


if __name__ == "__main__":
    """
    """
    dataset = Dataset()
    model = Model(dataset=dataset)
    model.train()

    for i in range(10):
        pred = model.predict([dataset.X[i]])
        print(f"PRED {pred} | {dataset.y[i]}")
