from calendar import monthrange

import xarray as xr
import urllib.request
import shutil
import datetime


def daysInMonth(year, month):
    return monthrange(int(year), int(month))[1]


class Resource:
    def __init__(self, url=None, filename=None):
        self.data = None
        self.url = url
        self.filename = filename

    def fetch(self):
        with urllib.request.urlopen(self.url) as response, open(self.location(), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
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
        self.names = ['MO_CHL_chlor_a', 'MO_FLH_nflh', 'MO_FLH_ipar', 'MO_NSST_sst', 'MO_PIC_pic', 'MO_POC_poc']

    def fetch(self, year, month):
        year = year
        month = month
        for name in self.names:
            self.fetch_for(name, year, month)

    def to_day_of_year(self, date):
        return date.strftime('%j')

    def fetch_for(self, name, year, month):
        if self.resources.get(name) is None:
            self.resources[name] = {}
        self.resources[name][str(year) + str(month)] = self.cached_fetch(
            self.base.format(str(year), self.to_day_of_year(datetime.datetime(year=year, month=month, day=1)), year,
                             self.to_day_of_year(
                                 datetime.datetime(year=year, month=month, day=daysInMonth(year, month))),
                             name), name + str(year) + str(month))

    def cached_fetch(self, url, filename):
        print(url)
        try:
            resource = Resource(filename=filename)
            return resource.load()
        except:
            resource = Resource(url, filename)
        return resource.fetch()

    def get(self):
        return self.resources

if __name__ == '__main__':
    DataProvider().fetch(2019, 5).save()
