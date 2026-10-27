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
    MIX = 3          # inspiration on the absolute clock after EE, exhale on navigator rank (fork, 2026-10-26)

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
    binsigma = 1.0   # soft-bin kernel width in bins: exp(-d^2 / (2 sigma^2)); 1.0 == Steve's bindist0sq = 2
    binpou = 0       # 1: normalise each interleave's bin weights to sum 1 (partition of unity)
    pneumoclock = 0  # 1: map the pneumotach clock onto the gas interleaves from the MDH timestamps (fork 2026-10-27)
    pneumo_offset_s = 0.0  # added to the pneumotach clock before mapping (record-start offset, e.g. -5.8 s in the 2024-07..09 window)
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
