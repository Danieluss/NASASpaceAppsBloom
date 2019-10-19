import numpy as np
import xarray as xr
import scipy.interpolate

class SF:
    def __init__(self, filename, key):
        self.filename = filename
        self.key = key

class FeaturesExtractor():
    def __init__(self, month, size_x=9, size_y=9, distance=40):
        self.month = month
        self.size_x = size_x
        self.size_y = size_y
        self.distance = distance
        self.features = [SF('a', 'chl_ocx'), SF('b', 'sst')]
        self.arrays = []
        for sf in self.features:
            self.arrays.append(self.get_array(sf))
        self.waters = self.get_array(SF('waters', 'sst'))
        print(self.waters[2].shape)

    def get_array(self, feature):
        if feature.filename == "waters":
            f = 'data/' + feature.filename + '.nc'
        else:
            f = 'data/' + feature.filename + str(self.month) + ".nc"
        d = xr.open_dataset(f)
        return (np.array(d['lat']), np.array(d['lon']), np.array(d[feature.key]))

    def get_grid(self, lat=0.0, lon=0.0):
        res = []
        for i in range(len(self.features)):
            res.append(self.get_feature(i, lat, lon))
        res = np.array(res)
        res = np.moveaxis(res, 0, -1)
        return res
        
    def get_feature(self, i, lat, lon):
        lat*=-1
        x = int((lat+90.0)*self.arrays[i][0].size/180.0)
        y = int((lon+180.0)*self.arrays[i][1].size/360.0)
        x_s = x-self.size_x//2
        y_s = y-self.size_y//2
        res = self.arrays[i][2][x_s:x_s+self.size_x:1, y_s:y_s+self.size_y:1]
        # if np.count_nonzero(np.isnan(res)==False) > 5:
        #     res = self.interpolate(res)
        return res

    def get_grid_mod(self, x, y):
        res = []
        for i in range(len(self.features)):
            res.append(self.get_feature_mod(i, x, y))
        res = np.array(res)
        res = np.moveaxis(res, 0, -1)
        return res
    
    def get_feature_mod(self, i, x, y):
        x_s = x-self.size_x//2
        y_s = y-self.size_y//2
        res = self.arrays[i][2][x_s:x_s+self.size_x:1, y_s:y_s+self.size_y:1]
        if np.count_nonzero(np.isnan(res)==False) > self.size_x:
            res = self.interpolate(res)
        return res
    
    def interpolate(self, arr):
        x = np.arange(0, arr.shape[1])
        y = np.arange(0, arr.shape[0])
        arr = np.ma.masked_invalid(arr)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        arr = arr[~arr.mask]
        return scipy.interpolate.griddata((x1, y1), arr.ravel(), (xx, yy), method='cubic')
    
    def get_waters(self):
        return self.waters[2]

    def get_dataset(self, n=1000, threshold=0.8):
        n = n//2
        val = np.argwhere(self.arrays[0][2] > threshold)
        res = []
        for i in range(n):
            r = np.random.randint(0, len(val))
            v = val[r]
            res.append(self.get_grid_mod(v[0], v[1]))
        
        val = np.argwhere(self.arrays[0][2] <= threshold)
        for i in range(n):
            r = np.random.randint(0, len(val))
            v = val[r]
            res.append(self.get_grid_mod(v[0], v[1]))
        return np.array(res)

if __name__ == "__main__":
    f = FeaturesExtractor(20187, 9, 9)
    a = f.get_grid(lat=0.0, lon=0.0)[:,:,0]
    # a = a/np.max(a[np.isnan(a) == False])

    a[a < 0.7] = np.nan

    c = f.get_dataset(10)
    print(c.shape)

    # b = f.get_waters()

    import matplotlib.pyplot as plt
    # # print(np.count_nonzero(a > 0.7)/a.size)
    # # ax[1].imshow(f.interpolate(a))
    plt.imshow(c[0,:,:,0])
    plt.show()