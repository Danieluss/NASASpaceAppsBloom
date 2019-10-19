from calendar import monthrange

import xarray as xr
import urllib.request
import datetime
import gzip
import shutil
import os.path
from os import path


def daysInMonth(year, month):
    return monthrange(int(year), int(month))[1]


class Resource:
    def __init__(self, url=None, filename=None):
        self.data = None
        self.url = url
        self.filename = filename

    def fetch(self):
        with urllib.request.urlopen(self.url) as response, open(self.location(), 'wb') as out_file:
            print('saving ' + self.location())
            shutil.copyfileobj(response, out_file)
        return self

    def fetch_load(self):
        self.fetch()
        self.load()
        return self

    def load(self):
        self.data = xr.open_dataset(self.location())
        return self

    def save(self):
        print('saving ' + self.location())
        self.data.to_netcdf(self.location())
        return self

    def location(self):
        return 'data/' + self.filename


class DataProvider:
    def __init__(self):
        self.resources = {}
        self.base = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A{}{}{}{}.L3m_{}_9km.nc'
        self.base_oscar = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/oscar/preview/L4/oscar_third_deg//oscar_vel{}.nc.gz'
        self._names = ['MO_CHL_chlor_a', 'MO_FLH_nflh', 'MO_FLH_ipar', 'MO_NSST_sst', 'MO_PIC_pic', 'MO_POC_poc']

    def fetch(self, year, month):
        if isinstance(year, list):
            for y in year:
                self.fetch(y, month)
            return
        if isinstance(month, list):
            for m in month:
                self.fetch(year, m)
            return
        for name in self._names:
            self.fetch_for(name, year, month)
        name = 'OSCAR_VEL'
        if self.resources.get(name) is None:
            self.resources[name] = {}
        self.resources[name][str(year) + str(month)] = self.fetch_oscar('OSCAR_VEL', year, month)

    def to_day_of_year(self, date):
        return date.strftime('%j')

    def make_url(self, name, year, month):
        return self.base.format(str(year), self.to_day_of_year(datetime.datetime(year=year, month=month, day=1)), year,
                                self.to_day_of_year(
                                    datetime.datetime(year=year, month=month, day=daysInMonth(year, month))),
                                name)

    def fetch_oscar(self, name, year, month):
        oscar_startdate = datetime.datetime(year=2011, day=6, month=12)
        scan_days = (datetime.datetime(year=year, month=month, day=1) - oscar_startdate).days
        if not path.exists(Resource(filename=name + str(year) + str(month)).location()):
            for i in range(10):
                try:
                    return self.cached_fetch(self.base_oscar.format(7001 + scan_days - i),
                                             name + str(year) + str(month))
                    break
                except:
                    continue

    def fetch_for(self, name, year, month):
        if self.resources.get(name) is None:
            self.resources[name] = {}
        if self.resources.get(name).get(str(year) + str(month)) is None:
            self.resources[name][str(year) + str(month)] = self.cached_fetch(self.make_url(name, year, month),
                                                                             name + str(year) + str(month))

    def cached_fetch(self, url, filename):
        print(url)
        try:
            resource = Resource(filename=filename)
            return resource.load()
        except:
            resource = Resource(url, filename)
            return resource.fetch_load()

    def get(self):
        return self.resources


if __name__ == '__main__':
    data_provider = DataProvider()
    data_provider.fetch(2019, [5, 6])
    print('finished')
