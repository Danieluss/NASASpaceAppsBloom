import datetime
import os
import shutil
import urllib.request
from calendar import monthrange
import xarray as xr
from ecmwfapi.api import ECMWFDataServer
import cfgrib

ecmwf = ECMWFDataServer(url="https://api.ecmwf.int/v1", key="8e65cd061cb66a38a26c3e394d989444",
                        email="maciejanthonyczyzewski@gmail.com")


def days_in_month(year, month):
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


class GRIBResource:
    def __init__(self, filename=None):
        self.data = None
        self.filename = filename

    def load(self):
        self.data = xr.open_dataset(self.filename, engine='cfgrib')
        return self

    def location(self):
        return 'data/' + self.filename


class DataProvider:
    def __init__(self):
        self.resources = {}
        self.base = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A{}{}{}{}.L3m_{}_9km.nc'
        self.nasa_datasets = ['MO_CHL_chlor_a', 'MO_FLH_nflh', 'MO_FLH_ipar', 'MO_NSST_sst', 'MO_PIC_pic', 'MO_POC_poc']
        self.ecmwf_datasets = ['SALT', 'CO', 'SO2']
        self.ecmwf_params = ['2.210/3.210', '123.210', '122.210']

    def fetch(self, year, month):
        if isinstance(year, list):
            for y in year:
                self.fetch(y, month)
            return
        if isinstance(month, list):
            for m in month:
                self.fetch(year, m)
            return
        for name in self.nasa_datasets:
            self.fetch_for(name, year, month)
        self.fetch_ecmwf(year, month)

    def fetch_ecmwf(self, year, month):
        if month < 10:
            date = "{}-{}-01".format(year, '0' + str(month))
        else:
            date = "{}-{}-01".format(year, '0' + str(month))
        postfix = str(year) + str(month)

        for name, param in zip(self.ecmwf_datasets, self.ecmwf_params):
            self.fetch_single_ecmwf(name + '_' + postfix, date, param, postfix)

    def fetch_single_ecmwf(self, name, date, param, postfix):
        filename = 'data/' + name
        if not os.path.exists(filename):
            ecmwf.retrieve({
                "class": "mc",
                "dataset": "macc",
                "date": date,
                "expver": "rean",
                "grid": "0.75/0.75",
                "levelist": "60",
                "levtype": "ml",
                "param": param,
                "step": "0",
                "stream": "oper",
                "time": "00:00:00",
                "type": "an",
                "target": filename,
            })
        if self.resources.get(name) is None:
            self.resources[name] = {}
        if self.resources.get(name).get(postfix) is None:
            self.resources[name][postfix] = GRIBResource(filename).load()

    def to_day_of_year(self, date):
        return date.strftime('%j')

    def fetch_for(self, name, year, month):
        if self.resources.get(name) is None:
            self.resources[name] = {}
        if self.resources.get(name).get(str(year) + str(month)) is None:
            self.resources[name][str(year) + str(month)] = self.cached_fetch(
                self.base.format(str(year), self.to_day_of_year(datetime.datetime(year=year, month=month, day=1)), year,
                                 self.to_day_of_year(
                                     datetime.datetime(year=year, month=month, day=days_in_month(year, month))),
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
    DataProvider().fetch(2011, [5, 6])
