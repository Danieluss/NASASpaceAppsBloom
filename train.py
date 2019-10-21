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

from dataset import Data
from model.tree import ModelTree
from model.treemean import ModelTreeMean
from model.conv2d import ModelConv2d

print("[MODEL]")

THRESHOLD = 0.8  # FIXME: in future linear


def get_dataset(model_block, year=2018):
    global THRESHOLD

    kanapki = []
    d = Data(year)

    a = d.load_dataset()
    print(a.shape)

    for x, y in a:
        x_map = model_block.get_input(x)
        kanapki.append(
            Kanapka(
                model_block=model_block,
                features=x_map,
                label=model_block.get_output(y),
            ))
        # FIXME: add [[augment]]
        try:
            for _ in range(4):
                x_copy = x_map
                x_map = np.swapaxes(x_map, 0, 1)
                if np.array_equal(x_copy, x_map):
                    break
                kanapki.append(
                    Kanapka(
                        model_block=model_block,
                        features=x_map,
                        label=model_block.get_output(y),
                    ))
        except:
            pass

    return Dataset(model_block=model_block,
                   kanapki=kanapki,
                   threshold=THRESHOLD)


class Kanapka:
    # FIXME: TO MODYFIKUJE KAMIL
    def __init__(self, model_block=None, features=None, label=None):
        self.model_block = model_block
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
        return self.model_block.get(self)


class Dataset:
    X = []
    y = []

    def __init__(self, model_block=None, kanapki=None, threshold=None):
        self.model_block = model_block
        self.threshold = threshold
        if kanapki is None:
            self.fake()  # FIXME: lista kanapek
        else:
            for a in kanapki:
                f, l = a.get()
                self.X.append(f)
                self.y.append(l)
        self.model_block.normalize(self)

    def fake(self, N=2000):
        for _ in range(N):
            a = Kanapka()
            a.fake()
            f, l = a.get()
            self.X.append(f)
            self.y.append(l)

    def train_and_test(self):
        return train_test_split(self.X,
                                self.y,
                                test_size=0.33,
                                random_state=42)


if __name__ == "__main__":
    # model_block = ModelConv2d  # FIXME: ModelConv2d
    # model_block = ModelConv2d
    model_block = ModelTree

    dataset = get_dataset(model_block)
    model = model_block(dataset=dataset)
    # model.load()
    model.train()

    import lightgbm as lgb
    import matplotlib.pyplot as plt

    print("Feature importances:", list(model.pst.feature_importance()))

    ax = lgb.plot_tree(model.pst)
    plt.show()

    ax = lgb.plot_importance(model.pst,
                             importance_type="gain",
                             max_num_features=30)
    plt.show()

    # model.train()

    for i in range(100):
        pred = model.predict([dataset.X[i]])
        print(f"PRED {pred} | {dataset.y[i]}")
