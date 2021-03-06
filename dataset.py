from features import FeaturesExtractor
import numpy as np
from tqdm import tqdm
import pickle


class Data:
    def __init__(self, year):
        self.year = year

    def add_to_res(self, val, res, n):
        for i in tqdm(range(n)):
            r = np.random.randint(0, len(val))
            v = val[r]
            arr = []
            for fe in self.fe:
                arr.append(fe.get_grid_mod(v[0], v[1]))
            for i in range(len(arr) - 1):
                a = arr[i + 1][:, :, 0]
                if np.count_nonzero(np.isnan(a)) > a.size / 2 or np.isnan(arr[i+1][4,4,0]):
                    continue
                res.append([arr[i], arr[i + 1]])

    def create_dataset(self, n=1000, threshold=0.8):
        self.fe = []
        for i in range(1, 13):
            self.fe.append(FeaturesExtractor(self.year, i))
        n //= 2
        res = []
        dx = self.fe[0].dx
        dy = self.fe[0].dy
        val = np.argwhere(self.fe[0].arrays[0][2][:-dx, :-dy] > threshold)
        print(val.size)
        self.add_to_res(val, res, n)
        val = np.argwhere(self.fe[0].arrays[0][2][:-dx, :-dy] <= threshold)
        print(val.size)
        self.add_to_res(val, res, n)
        self.res = np.array(res)

    def save_dataset(self):
        filename = "data/" + str(self.year) + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(self.res, f)

    def load_dataset(self):
        filename = "data/" + str(self.year) + ".pickle"
        with open(filename, "rb") as f:
            self.res = pickle.load(f)
        return self.res  # (number_of_examples, 2, x, y, number_of_features)

    def update_dataset(self, d):
        self.res = np.concatenate((self.res, d.res))

    def get_dataset(self):
        return self.res


if __name__ == "__main__":
    d = Data(2018)
    # d.load_dataset()
    d.create_dataset(1000)
    # d.save_dataset()
    print(d.get_dataset().shape)
    # e = Data(2017)
    # e.create_dataset(1000)
    # print(e.get_dataset().shape)
    # d.update_dataset(e)
    print(d.get_dataset().shape)
    d.save_dataset()
    
    # c = d.load_dataset()
    # a = c[:, 0, :, :, 0]
    # a = np.nanmean(a, axis=(1, 2))
    # print(np.count_nonzero(a > 0.7)/a.size)
    # n = c.shape[-1]
    # import matplotlib.pyplot as plt
    # plt.hist(a)
    # plt.show()

    # fig, ax = plt.subplots(2, n)
    # for i in range(n):
    #     ax[0][i].imshow(c[200, 0, :, :, i])
    #     ax[1][i].imshow(c[200, 1, :, :, i])
    # plt.show()
    
