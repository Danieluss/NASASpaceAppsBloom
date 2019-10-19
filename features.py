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

    def get_array(self, feature):
        f = feature.filename + str(self.month) + ".nc"
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
        x = int((lat+90.0)*self.arrays[i][0].size/180.0)
        y = int((lon+180.0)*self.arrays[i][1].size/360.0)
        x_s = x-self.size_x//2
        y_s = y-self.size_y//2
        res = self.arrays[i][2][x_s:x_s+self.size_x:1, y_s:y_s+self.size_y:1]
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

if __name__ == "__main__":
    f = FeaturesExtractor(200910, 100, 100)
    a = f.get_grid(lat=50.0, lon=-20.0)[:,:,0].reshape(100, 100)
    a = a/np.max(a[np.isnan(a) == False])

    import matplotlib.pyplot as plt
    plt.imshow(a)
    # ax[1].imshow(f.interpolate(a))
    plt.show()