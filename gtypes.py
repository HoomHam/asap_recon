# globals and enums
from enum import IntEnum

class imgtype(IntEnum):
    GPDYN = 0
    DPDYN = 1
    GPREF = 2
    DPREF = 3

class graddir(IntEnum):
    X = 0
    Y = 1
    Z = 2

class bintype(IntEnum):
    PNEUMOTACH = 0
    DIAPHRAGM = 1
    SIGNAL = 2

class species(IntEnum):
    HUMAN = 0
    RAT = 1
    RABBIT = 2
    NONE = 3

class gvar:
    basefolder = '/Users/kento/dev/data/'
    MS = 240
    IS = 100
    nbins = 16
    bindt = 1
    griddx = 1
    usegpu = 1
    gplb = 300
    dplb = 40
    freqfilter = 0
    def get_MScenter(self): return(int(self.MS / 2 + .1))
    def ISLL(self): return(int(self.get_MScenter() - self.IS / 2 + .1))
    def ISUL(self): return(self.ISLL() + self.IS)

class gdir:
    datadirname = ''
    dyndatasets = []
    dyndatasetslen = []
    pneumodatasets = []
    pneumodatasetslen = []
    trajdirname = ''
    gptrajfilename = ''
    dpgrajfilename = ''
    fileformat = 'siemens'
