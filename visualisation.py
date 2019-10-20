import numpy as np
import matplotlib.pyplot as plt
from features import FeaturesExtractor
import pickle
from tqdm import tqdm
from model.tree import ModelTree
from pprint import pprint


class Visalisation:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.path = "data/" + str(year) + str(month) + "v.pickle"

    def prepare_dataset(self):
        dx, dy = 9, 9
        a = FeaturesExtractor(self.year, self.month)
        if self.month == 12:
            year = self.year + 1
            month = 1
        else:
            year = self.year
            month = self.month + 1
        b = FeaturesExtractor(year, month)
        self.res = []
        for i in tqdm(range(0, a.map_shape[0], 10 * dx)):
            res2 = []
            for j in range(0, a.map_shape[1], 10 * dy):
                inp, out, land = self.get_input_and_real_output(
                    a, b, i, j, dx, dy)
                res2.append([inp, out, land])
            self.res.append(res2)
        self.save_dataset()

    def save_dataset(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.res, f)

    def load_dataset(self):
        with open(self.path, "rb") as f:
            self.res = pickle.load(f)

    def get_input_and_real_output(self, a, b, x, y, dx, dy):
        inp = a.get_grid_mod(x, y)[:, :, 0:]
        out = b.get_feature_mod(0, x, y)[:, :]
        land = np.mean(a.land_mask[x:x + dx, y:y + dy]) > 0.5
        return inp, out, land

    def visualise(self, model, threshold=0.8):
        from train import Kanapka

        real = []
        predicted = []
        for row in tqdm(self.res):
            real2 = []
            inp = [
                Kanapka(
                    model_block=type(model),
                    features=type(model).get_input(col[0]),
                ).get()[0] for col in row
            ]
            print(np.array(inp).shape)
            # [pprint(np.array(x)) for x in inp]
            # exit(0)
            predicted2 = model.predict(inp)
            ids = np.isnan(predicted2) == False
            predicted2[ids] = predicted2[ids] <= 0.5
            j = 0
            for col in row:
                r = col[1]
                if np.count_nonzero(np.isnan(r)) == r.size:
                    r = np.nan
                else:
                    r = np.nanmean(r) > threshold
                if col[2]:
                    predicted2[j], r = np.nan, np.nan
                real2.append(r)
                j += 1
            real.append(real2)
            predicted.append(predicted2)

        real = np.array(real)
        predicted = np.array(predicted)
        # print(np.unique(real), np.unique(predicted))
        _, ax = plt.subplots(2, 1)
        # ax[0].hist(real)
        # ax[1].hist(predicted)
        ax[0].imshow(real)
        ax[1].imshow(predicted)
        plt.show()


if __name__ == "__main__":
    v = Visalisation(2018, 3)
    v.prepare_dataset()
    v.load_dataset()
    t = ModelTree()
    t.load()
    v.visualise(t)
    # a = FeaturesExtractor(2018, 1)
    # b = FeaturesExtractor(2018, 2)
    # print(a.map_shape)
    # get_input_and_real_output(a, b, 1000, 2000)
