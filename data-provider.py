import xarray as xr
from calendar import monthrange
import wget


def daysInMonth(year, month):
    return monthrange(year, month)[1]


class Resource:
    def __init__(self, url=None, filename=None):
        self.data = None
        self.url = url
        self.filename = filename

    def fetch(self):
        wget.download(self.url, self.location())
        self.load()
        return self

    def load(self):
        self.data = xr.open_dataset(self.location())
        return self

    def save(self):
        self.data.to_netcdf(self.location())
        return self

    def location(self):
        return 'data/' + self.filename


class DataProvider:
    def __init__(self):
        self.resources = {}
        self.base = 'http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A{}{}{}{}.L3m_{}_9km.nc'

    def fetch(self, year, month):
        year = str(year)
        month = str(month)
        self.fetch_for('DAY_CHL_chlor_a', year, month)
        self.fetch_for('MO_FLH_nflh', year, month)
        self.fetch_for('MO_FLH_ipar', year, month)
        self.fetch_for('MO_NSST_sst', year, month)
        self.fetch_for('MO_PIC_pic', year, month)
        self.fetch_for('MO_POC_poc', year, month)

    def fetch_for(self, name, year, month):
        if self.resources[name] is None:
            self.resources[name] = {}
        self.resources[name][str(year) + str(month)] = self.cachedFetch(
            self.base.format(year, '1', year, daysInMonth(year, month),
                             name), name)

    @staticmethod
    def cached_fetch(url, filename):
        try:
            resource = Resource(filename=filename)
            return resource.load().data
        except:
            resource = Resource(url, filename)
            return resource.fetch().data

    def get(self):
        return self.resources
