import subprocess
import sys
from tkinter import * #Tk, StringVar, IntVar, Button, Radiobutton, Label, Entry, OptionMenu, Checkbutton, filedialog, _setit
from tkinter import _setit, filedialog
from PIL import ImageTk, Image
from dateutil.parser import parse
from os import listdir, path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import numpy as np
from numba import cuda
from gtypes import gvar, gdir, imgtype, bintype, graddir, species
from raw import raw, traj
from results import results
from scipy.signal import savgol_filter
from scipy.io import savemat
from enum import IntEnum


# ---------------------------------------------------------------------------
# GLOBAL STATE OBJECTS
# ---------------------------------------------------------------------------
g     = gvar()
g_raw = raw()
g_traj = traj()
g_res = results()
g_dir = gdir()

class urimgtype(IntEnum):
    NONE = 0
    COLORMAP = 1
    BWMAP = 2
    COLORMAPDIAPHRAGM = 3
    COMPLEXPLOT = 4

# Widget globals — set by setup_* functions, referenced by callbacks
root = None
dyn_unreg_frames = []

calcLV_method    = None
calcLV_pneumo    = None
showLV_var       = None

MS_entry         = None
IS_entry         = None
nbins_entry      = None
xyz_kernel_entry = None
t_kernel_entry   = None
gplb_entry       = None
dplb_entry       = None
freqfilter_entry = None

excluderanges_cb_disable = False
excluderanges_str        = None
excluderanges_entry      = None

dyn_recon_button  = None
calc_T1RF_button  = None

dataset_figure = None
dataset_plot   = None
dataset_canvas = None
ur_figure      = None
ur_figuretype  = None
ur_plot        = None
ur_canvas      = None
ur_plot_colors = ['b', 'k', 'r', 'm']

gp_traj_label        = None
dp_traj_label        = None
datasets_label       = None
pneumotach_found_label = None
rootdir_label        = None
dyn_image_label      = None

data_menu_var = None
date_menu_var = None
ID_menu_var   = None
data_menu     = None
date_menu     = None
ID_menu       = None

usegpu = False


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def rangeoverlap(f1, f2, rg):
    # returns True if the range from (float) f1 to f2 interesects the range rg
    return((f1 > rg.start and f1 < rg.stop) or (f2 > rg.start and f2 < rg.stop) or (f1 < rg.start and f2 > rg.stop))

def stringtorangelist(s):
    import re
    a = list(map(int, re.findall(r'\d+', s)))
    if(len(a) % 2 == 1):
        return([])
    rangelist = []
    for j in range(0, len(a), 2):
        if(a[j+1] > a[j]):
            rangelist.append(range(a[j], a[j+1]))
    return(rangelist)

def rangelisttostring(rl):
    s = ''
    for j in range(0, len(rl)):
        s += str(rl[j].start) + '-' + str(rl[j].stop) + 's'
        if(j < len(rl) - 1):
            s += ', '
    return(s)

def isdt(x):
    try:
        parse(x)
    except:
        return(False)
    return(True)

def getdtlist(x):
    return([i for i in x if isdt(i)])

def refresh_menu(thismenu, selectedstring, newoptions):
    selectedstring.set('')
    thismenu['menu'].delete(0, 'end')
    for choice in newoptions:
        thismenu['menu'].add_command(label = choice, command = _setit(selectedstring, choice))
    if(len(newoptions) > 0):
        selectedstring.set(newoptions[0])

def getbintype():
    bt = bintype.SIGNAL if calcLV_method.get() == 'Signal' else bintype.PNEUMOTACH
    return(bintype.DIAPHRAGM if calcLV_method.get() == 'Diaphragm' else bt)


# ---------------------------------------------------------------------------
# UI UPDATE FUNCTIONS
# ---------------------------------------------------------------------------

def ur_fig_update(type, thing):
    global ur_plot
    global ur_canvas
    ur_plot.cla()
    match type:
        case urimgtype.NONE:
            ur_plot.cla()
        case urimgtype.BWMAP:
            ur_plot.imshow(thing, cmap = 'gray')
        case urimgtype.COLORMAP:
            ur_plot.imshow(thing)
        case urimgtype.COLORMAPDIAPHRAGM:
            ur_plot.imshow(thing[0])
            ur_plot.plot([0, g.IS - 1], [thing[1], thing[1]], 'r')
            ur_plot.text(10, 10, '%d' % thing[2], color='w')
        case urimgtype.COMPLEXPLOT:
            for pltidx in range(len(thing)):
                ur_plot.plot(thing[pltidx], ur_plot_colors[pltidx])
    ur_canvas.draw()

def binning_plot_update():
    dataset_plot.cla()
    if(len(g_raw.ilvtime) == 0):
        return
    gpdyn_duration = 0.0
    if(g_raw.hasimg(imgtype.GPDYN)):
        dataset_plot.plot(g_raw.volmeastime[bintype.SIGNAL], g_raw.volmeasvol[bintype.SIGNAL], ':y')
        gpdyn_duration = g_raw.volmeastime[bintype.SIGNAL][-1]
    if(g_raw.hasimg(imgtype.GPREF)):
         sigref = np.sum(np.abs(g_raw.getimg(imgtype.GPREF)[0:10, :, :]), axis = (0, 1))
         dataset_plot.plot(np.array(range(len(sigref))) * g_raw.TR + gpdyn_duration, sigref / np.max(sigref), ':y')
    colors = ['r', 'g', 'b']
    bt = getbintype()
    if(showLV_var.get() == 'Bin' and len(g_raw.ilvbin[bt]) == len(g_raw.ilvtime)):
        dataset_plot.plot(g_raw.ilvtime[g_raw.ilvbin[bt] >= 0], g_raw.ilvbin[bt][g_raw.ilvbin[bt] >= 0], colors[int(bt)])
        for rg in g_raw.excluderanges:
            dataset_plot.plot(g_raw.ilvtime[int(rg.start / g_raw.TR):int(rg.stop / g_raw.TR)], g_raw.ilvbin[bt][int(rg.start / g_raw.TR):int(rg.stop / g_raw.TR)], '--w')
    elif(showLV_var.get() == 'Volume' and len(g_raw.ilvbin[bt]) == len(g_raw.ilvtime)):
        dataset_plot.plot(g_raw.ilvtime, g_raw.ilvvol[bt], colors[int(bt)])
        for rg in g_raw.excluderanges:
            dataset_plot.plot(g_raw.ilvtime[int(rg.start / g_raw.TR):int(rg.stop / g_raw.TR)], g_raw.ilvvol[bt][int(rg.start / g_raw.TR):int(rg.stop / g_raw.TR)], '--w')
    for j in range(0, len(g_raw.volmeasEEtime[bt])):
        dataset_plot.plot([g_raw.volmeasEEtime[bt][j], g_raw.volmeasEEtime[bt][j]], [0, -.1], colors[int(bt)])
    dataset_plot.plot(g_raw.ilvtime, g_raw.ilvtime * 0, 'k')
    dataset_plot.set_yticks([])
    dataset_plot.set_ylim(-.1, 1)
    dataset_canvas.draw()

def buttons_update():
    global excluderanges_str
    global excluderanges_cb_disable
    calc_T1RF_button['state'] = 'normal' if g_raw.hasimg(imgtype.GPREF) else 'disabled'
    dyn_recon_button['state'] = 'normal'# if not type(g_res.b) == list and len(g_raw.ilvbin[getbintype()]) > 0 else 'disabled'
    if(not calcLV_pneumo['state'] == 'normal' and getbintype() == bintype.PNEUMOTACH):
        calcLV_method.set('Signal')
    excluderanges_cb_disable = True
    excluderanges_entry.delete(0, END)
    excluderanges_entry.insert(0, rangelisttostring(g_raw.excluderanges))
    excluderanges_cb_disable = False

def movie_update(ind):
    global dyn_image_label
    if(ind >= len(dyn_unreg_frames)):
        root.after(100, movie_update, 0)
        return
    frame = dyn_unreg_frames[ind]
    dyn_image_label.configure(image = frame)
    dyn_image_label.image = frame
    root.after(100, movie_update, (ind + 1) % len(dyn_unreg_frames))


# ---------------------------------------------------------------------------
# PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------

def check_recon_entries():
    t = int(IS_entry.get())
    if(t >= 8 and t <= 512):
        g.IS = t
    t = int(MS_entry.get())
    if(t >= 8 and t <= 512):
        g.MS = t
    t = int(nbins_entry.get())
    if(t >= 1 and t <= 128):
        g.nbins = t
    dt = float(t_kernel_entry.get())
    if(dt >= .1 and dt <= g.nbins / 2):
        g.bindt = dt
    dx = float(xyz_kernel_entry.get())
    if(dx >= .1 and dx <= 5):
        g.griddx = dx
    i = int(gplb_entry.get())
    if(i >= 0):
        g.gplb = i
    i = int(dplb_entry.get())
    if(i >= 0):
        g.dplb = i
    i = int(freqfilter_entry.get())
    if(i >= 0):
        g.freqfilter = i
    g.usegpu = cuda.is_available()

def convert_and_recon():
    """Convert the selected Siemens .dat directory to MRD (with GUI parameters
    as header user parameters), then run tyger_recon.py on it as a subprocess.
    Both the raw and reconstructed MRD files land in the data directory."""
    check_recon_entries()
    raw_mrd   = path.join(g_dir.datadirname, 'xe_dyn_raw.mrd')
    recon_mrd = path.join(g_dir.datadirname, 'xe_dyn_recon.mrd')
    pneumo = g_dir.pneumodatasets[0] if len(g_dir.pneumodatasets) == 1 else None
    binning = 'PNEUMOTACH' if (getbintype() == bintype.PNEUMOTACH and pneumo) else 'SIGNAL'
    params = dict(MS=g.MS, IS=g.IS, nbins=g.nbins, griddx=g.griddx, bindt=g.bindt,
                  gplb=g.gplb, dplb=g.dplb, freqfilter=g.freqfilter, binning=binning)
    try:
        from convert_siemens_to_mrd import convert_siemens_to_mrd
        convert_siemens_to_mrd(g_dir.datadirname, raw_mrd, g_dir.gptrajfilename, g_dir.dptrajfilename,
                 pneumotach_file=pneumo, params=params, dat_files=g_dir.dyndatasets)
    except Exception as e:
        print('conversion FAILED:', e)
        datasets_label.config(text='conversion FAILED: %s' % e, fg='red')
        return
    datasets_label.config(text='converted, running recon...', fg='green')
    root.update()
    cmd = [sys.executable, path.join(path.dirname(path.abspath(__file__)), 'tyger_recon.py'),
           '--input', raw_mrd, '--output', recon_mrd]
    proc = subprocess.run(cmd, cwd=g_dir.datadirname)  # child output streams to terminal
    if proc.returncode == 0:
        print('recon done:', recon_mrd)
        datasets_label.config(text='recon done: ' + recon_mrd, fg='green')
    else:
        print('recon FAILED (exit %d)' % proc.returncode)
        datasets_label.config(text='recon FAILED (exit %d)' % proc.returncode, fg='red')

def calcLVcb():
    bt = getbintype()
    if(bt == bintype.DIAPHRAGM and not len(g_raw.ilvbin[bt])):
        print('calculating low-res b')
        saveMS = g.MS
        g.MS = g.IS + 4
        g_res.calcb(g, g_raw, g_traj, True)
        nusimg = int(g_raw.ntotalilvs * g_raw.npts / g_traj.nsmpperusimg)
        tempvolmeasvol = []
        for iusimg in range(0, nusimg):
            res = g_res.dyn_usimg_recon(g, g_raw, g_traj, g.usegpu, iusimg)
            if(not res):
                continue
            urfig = np.sum(np.abs(g_res.getimg(imgtype.GPDYN)[:, int(g.IS/4):int(3*g.IS/4), :]), 1)
            dropoff = np.sum(np.abs(g_res.getimg(imgtype.GPDYN)[:, int(g.IS/4):int(3*g.IS/4), :]), (1, 2))
            # fit from 25% to 75% signal
            for p1 in range(g.IS - 1, 0, -1):
                if(dropoff[p1] > np.min(dropoff) + 0.25 * (np.max(dropoff) - np.min(dropoff))):
                    break
            for p2 in range(p1, 0, -1):
                if(dropoff[p2] > np.min(dropoff) + 0.75 * (np.max(dropoff) - np.min(dropoff))):
                    break
            if(p1 - p2 >= 2):
                p = np.polyfit(np.array(range(p2, p1)), dropoff[p2:p1], 2)
                # fit curve is where y = p[0] x^2 + p[1] x + p[2], need to choose x where y = avg of min and max
                # (max - min) / 2 = p[0] x^2 + p[1] x + p[2]
                # p[0] x^2 + p[1] x + (p[2] - (max - min) / 2) = 0
                det = np.sqrt(p[1]**2 - 4 * p[0] * (p[2] - (np.max(dropoff) + np.min(dropoff)) / 2))
                m1 = (-p[1] + det) / (2 * p[0])
                m2 = (-p[1] - det) / (2 * p[0])
                ilvperusimg = int(g_traj.nsmpperusimg / g_raw.npts + .1)
                if(m1 > p2 and m1 < p1):
                    tempvolmeasvol.append(m1)
                    g_raw.volmeastime[bt].append(np.mean(g_raw.ilvtime[(iusimg * ilvperusimg):((iusimg + 1) * ilvperusimg)]))
                    urfigline = m1
                elif(m2 > p2 and m2 < p1):
                    tempvolmeasvol.append(m2)
                    g_raw.volmeastime[bt].append(np.mean(g_raw.ilvtime[(iusimg * ilvperusimg):((iusimg + 1) * ilvperusimg)]))
                    urfigline = m2
            else:
                print('p2 - p1 < 2')
            if(len(g_raw.volmeastime[bt]) > 5):
                ftvol = savgol_filter(tempvolmeasvol, 5, 2)
                minftvol = np.min(ftvol)
                g_raw.ilvvol[bt] = np.interp(g_raw.ilvtime, g_raw.volmeastime[bt], \
                        (ftvol - minftvol) / (max(ftvol) - minftvol), left=np.nan, right=np.nan)
                g_raw.rescale(g_raw.ilvvol[bt])
                g_raw.ilvbin[bt] = g_raw.bin(g_raw.ilvvol[bt])
                binning_plot_update()
            ur_fig_update(urimgtype.COLORMAPDIAPHRAGM, [urfig, urfigline, iusimg])
            root.update()
        ur_fig_update(urimgtype.COLORMAP, g_res.bmag[:, int(g.IS / 2), :])
        g.MS = saveMS
    binning_plot_update()
    buttons_update()
    root.update()

def calc_T1RF_function():
    # first non-killed point of each acquisition should have the same phase
    sigref = g_raw.getimg(imgtype.GPREF)[0, :, :].copy()
    for ich in range(0, g_raw.nch):
        sigref[ich, :] *= np.conj(np.mean(sigref[ich, :]))
    sigref = np.real(np.sum(sigref, axis = (0)))
    # throw away points after signal has decayed by a factor of 10
    for ipt in range((len(sigref) - 1), 0, -1):
        print(ipt, sigref[ipt])
        if(sigref[ipt] > np.max(sigref) / 10):
            sigref = sigref[:(ipt + 1)]
            break
    sigref = np.log(sigref)
    idx = [i for i in range(0, len(sigref))]
    refilvtime = g_raw.ilvtime[idx]
    p = np.polyfit(refilvtime[idx], sigref[idx], 1)
    # remove points until correlation coefficient is > 0.99
    while(idx):
        cc = np.min(np.abs(np.corrcoef(refilvtime[idx], sigref[idx])))
        p = np.polyfit(refilvtime[idx], sigref[idx], 1)
        avgdecayrate = -p[0]
        if(cc > 0.99):
            break
        idx.remove(idx[np.argmax(np.abs(sigref[idx] - np.polyval(p, refilvtime[idx])))])
    # recalculate b matrix based on full dynamic dataset
    g_res.calcb(g, g_raw, g_traj, usegpu)
    bumpup = np.zeros(max(idx) + 1)
    sig = np.zeros((g_raw.npts, g_raw.nch, max(idx) + 1), dtype='complex64')
    for iilv in idx:
        bumpup[iilv] = np.exp(avgdecayrate * refilvtime[iilv])
        sig[:, :, iilv] = g_raw.getimg(imgtype.GPREF)[:, :, iilv] * bumpup[iilv]
    g_res.T1RFimg = np.zeros((2, g.IS, g.IS, g.IS))
    g_res.T1RFimgtime = np.zeros(2)
    for iter in range(0, 2):
        sigcopy = sig.copy()
        useilv = np.zeros(max(idx) + 1, dtype = 'bool')
        killilv = np.zeros(max(idx) + 1, dtype = 'bool')
        for iilvidx in range(len(idx)):
            iilv = idx[iilvidx]
            useilv[iilv] = ((iter == 0 and iilvidx < int(len(idx) * 2 / 3)) or (iter == 1 and iilvidx > int(len(idx) / 3)))
            killilv[iilv] = not useilv[iilv]
        sigcopy[:, :, killilv] = 0.0
        g_res.T1RFimgtime[iter] = np.mean(refilvtime[useilv])
        g_res.T1RF_recon(g, g_raw, g_traj, sigcopy, usegpu)
        g_res.T1RFimg[iter, :, :, :] = g_res.getimg(imgtype.GPREF) / np.mean(bumpup[useilv])
    savemat(g_dir.datadirname + 'T1RF', {'T1RFimg': g_res.T1RFimg, 'T1RFtime': g_res.T1RFimgtime})
    urfig = g_res.T1RFimg[0, :, int(g.IS / 2), :] / g_res.T1RFimg[1, :, int(g.IS / 2), :]
    urfig[urfig < 0] = 0
    urfig[urfig > 3] = 3
    ur_fig_update(urimgtype.COLORMAP, urfig)


# ---------------------------------------------------------------------------
# DIRECTORY / MENU CALLBACKS
# ---------------------------------------------------------------------------

def ID_callback(*args):
    global g_dir
    global gp_traj_label
    global dp_traj_label
    global datasets_label
    datatype = data_menu_var.get()
    g_dir.trajdirname = g.basefolder + '/' + datatype + '/traj/'

    g_dir.datadirname = g.basefolder + '/' + datatype + '/' + date_menu_var.get() + '/' + ID_menu_var.get() + '/'
    allrawdir = [f for f in listdir(g_dir.datadirname) if not f.startswith('.')]
    g_dir.dyndatasets = [g_dir.datadirname + i for i in allrawdir if i.find('rawdata.job0') != -1]
    g_dir.fileformat = 'bruker' if len(g_dir.dyndatasets) > 0 else 'siemens'
    try:
        f = open(g_dir.trajdirname + 'seqnames.txt')
    except:
        return
    for x in f.read().splitlines():
        [fileformat, seqname] = x.split()
        if(fileformat == 'bruker' and g_dir.fileformat == 'bruker'):
            break
        g_dir.dyndatasets = [g_dir.datadirname + i for i in allrawdir if i.find(seqname) != -1]
        if(len(g_dir.dyndatasets) > 0):
            break
    f.close()
    g_dir.dyndatasetslen = [path.getsize(i) for i in g_dir.dyndatasets]
    if len(g_dir.dyndatasetslen) == 0:
        datasets_label.config(text = 'No dynamic datasets found', fg = 'red')
    else:
        dyntext = str(len(g_dir.dyndatasetslen)) + ' dynamic dataset' + ('s' if len(g_dir.dyndatasetslen) > 1 else '') + ': '
        for i in range(0, len(g_dir.dyndatasetslen)):
            dyntext += str(int(g_dir.dyndatasetslen[i] / 1000000 + 0.5))
            if(i < len(g_dir.dyndatasetslen) - 1):
                dyntext += ', '
        dyntext += ' MB'
        datasets_label.config(text = dyntext, fg = 'green')
    g_dir.pneumodatasets = [g_dir.datadirname + i for i in allrawdir if i.find('pneumotach') != -1]
    g_dir.pneumodatasetslen = [path.getsize(i) for i in g_dir.pneumodatasets]
    if(len(g_dir.pneumodatasets) == 0):
        calcLV_pneumo["state"] = 'disabled'
        pneumotach_found_label.config(text = 'No pneumotach file found', fg = 'red')
    elif(len(g_dir.pneumodatasets) == 1):
        calcLV_pneumo["state"] = 'normal'
        pneumotach_found_label.config(text = 'Pneumotach file found', fg = 'green')
    else:
        calcLV_pneumo["state"] = 'normal'
        pneumotach_found_label.config(text = 'Multiple pneumotach files found', fg = 'red')
    g_dir.gptrajfilename = g_dir.trajdirname + seqname + '_' + 'gp' + '.npy'
    if(path.exists(g_dir.gptrajfilename)):
        gp_traj_label.config(fg = 'green')
    else:
        gp_traj_label.config(fg = 'red')
    g_dir.dptrajfilename = g_dir.trajdirname + seqname + '_' + 'dp' + '.npy'
    if(path.exists(g_dir.dptrajfilename)):
        dp_traj_label.config(fg = 'green')
    else:
        dp_traj_label.config(fg = 'red')

def date_callback(*args):
    set_ID_menu()

def data_callback(*args):
    if(data_menu_var.get() == 'human'):
        g.species = species.HUMAN
    elif(data_menu_var.get() == 'rat'):
        g.speices = species.RAT
    elif(data_menu_var.get() == 'rabbit'):
        g.species = species.RABBIT
    else:
        g.species = species.NONE
    set_date_menu()

def set_ID_menu():
    options = [f for f in listdir(g.basefolder + '/' + data_menu_var.get() + '/' + date_menu_var.get()) if not f.startswith('.')]
    refresh_menu(ID_menu, ID_menu_var, options)

def set_date_menu():
    options = [f for f in listdir(g.basefolder + '/' + data_menu_var.get()) if not f.startswith('.')]
    options = getdtlist(options)
    refresh_menu(date_menu, date_menu_var, options)
    set_ID_menu()

def set_data_menu():
    options = [f for f in listdir(g.basefolder) if not f.startswith('.')]
    refresh_menu(data_menu, data_menu_var, options)
    set_date_menu()


# ---------------------------------------------------------------------------
# SETUP FUNCTIONS
# ---------------------------------------------------------------------------

def setup_window():
    global root
    root = Tk()
    root.title('HPX dynamic data analysis')
    root.geometry('1300x600')

def setup_figures():
    global dataset_figure, dataset_plot, dataset_canvas
    global ur_figure, ur_figuretype, ur_plot, ur_canvas

    dataset_figure = Figure(figsize = (6.5, 2), dpi = 100)
    dataset_figure.tight_layout()
    dataset_plot = dataset_figure.add_subplot(111)
    dataset_plot.set_yticks([])
    dataset_canvas = FigureCanvasTkAgg(dataset_figure, master = root)
    dataset_canvas.get_tk_widget().place(x = 470, y = 0)

    ur_figure = Figure(figsize = (2, 2), dpi = 100)
    ur_figuretype = urimgtype.NONE
    ur_figure.tight_layout()
    ur_plot = ur_figure.add_subplot(111)
    ur_plot.set_xticks([])
    ur_plot.set_yticks([])
    ur_canvas = FigureCanvasTkAgg(ur_figure, master = root)
    ur_canvas.get_tk_widget().place(x = 1070, y = 0)

def setup_binning_controls():
    global calcLV_method, calcLV_pneumo, showLV_var
 
    calcLV_method = StringVar(None, 'Pneumotach')
    calcLV_pneumo = Radiobutton(root, text = 'Pneumotach', value = 'Pneumotach', variable = calcLV_method, command=calcLVcb)
    calcLV_pneumo.place(x = 360, y = 185)
    calcLV_signal = Radiobutton(root, text = 'Signal', value = 'Signal', variable = calcLV_method, command=calcLVcb)
    calcLV_signal.place(x = 360, y = 203)
    calcLV_diaphragm = Radiobutton(root, text = 'Diaphragm', value = 'Diaphragm', variable = calcLV_method, command=calcLVcb)
    calcLV_diaphragm.place(x = 360, y = 221)

    def showLVcb(): binning_plot_update()
    showLV_var = StringVar(None, 'Volume')
    Radiobutton(root, text='Volume', value='Volume', variable=showLV_var, command=showLVcb).place(x = 360, y = 60)
    Radiobutton(root, text='Bin',    value='Bin',    variable=showLV_var, command=showLVcb).place(x = 360, y = 78)

def setup_param_entries():
    global MS_entry, IS_entry, nbins_entry, xyz_kernel_entry, t_kernel_entry
    global gplb_entry, dplb_entry, freqfilter_entry

    Label(root, text = 'Matrix size =').place(x = 10, y = 230)
    MS_entry = Entry(root, width = 3)
    MS_entry.insert(0, str(g.MS))
    MS_entry.place(x = 95, y = 230)

    Label(root, text = 'Image size  =').place(x = 10, y = 250)
    IS_entry = Entry(root, width = 3)
    IS_entry.insert(0, str(g.IS))
    IS_entry.place(x = 95, y = 250)

    Label(root, text = '# of bins   =').place(x = 10, y = 270)
    nbins_entry = Entry(root, width = 3)
    nbins_entry.insert(0, str(g.nbins))
    nbins_entry.place(x = 95, y = 270)

    Label(root, text = 'Kernel dx  =').place(x = 10, y = 290)
    xyz_kernel_entry = Entry(root, width = 3)
    xyz_kernel_entry.insert(0, '1')
    xyz_kernel_entry.place(x = 95, y = 290)

    Label(root, text = 'Kernel dt  =').place(x = 10, y = 310)
    t_kernel_entry = Entry(root, width = 3)
    t_kernel_entry.insert(0, '1')
    t_kernel_entry.place(x = 95, y = 310)

    Label(root, text = 'GP LB =').place(x = 10, y = 330)
    gplb_entry = Entry(root, width = 3)
    gplb_entry.insert(0, '300')
    gplb_entry.place(x = 95, y = 330)

    Label(root, text = 'DP LB =').place(x = 10, y = 350)
    dplb_entry = Entry(root, width = 3)
    dplb_entry.insert(0, '40')
    dplb_entry.place(x = 95, y = 350)

    Label(root, text = 'Freq filter =').place(x = 10, y = 370)
    freqfilter_entry = Entry(root, width = 3)
    freqfilter_entry.insert(0, '0')
    freqfilter_entry.place(x = 95, y = 370)

def setup_exclude_ranges():
    global excluderanges_cb_disable, excluderanges_str, excluderanges_entry

    def excluderanges_cb(var, index, mode):
        global excluderanges_cb_disable
        if(excluderanges_cb_disable):
            return
        global g_raw
        g_raw.excluderanges = stringtorangelist(excluderanges_str.get())
        binning_plot_update()

    Label(root, text = 'Exclude ranges').place(x = 320, y = 120)
    excluderanges_str = StringVar()
    excluderanges_str.trace_add('write', excluderanges_cb)
    excluderanges_entry = Entry(root, width = 20, textvariable = excluderanges_str)
    excluderanges_entry.place(x = 320, y = 140)

def setup_action_buttons():
    global dyn_recon_button, calc_T1RF_button

    dyn_recon_button = Button(root, text = 'Convert + Recon', fg = 'green', command = convert_and_recon)
    dyn_recon_button.place(x = 250, y = 150)

    calc_T1RF_button = Button(root, text = 'Calc T1RF', fg = 'green', command = lambda: calc_T1RF_function())
    calc_T1RF_button.place(x = 250, y = 200)

def setup_status_labels():
    global gp_traj_label, dp_traj_label, datasets_label, pneumotach_found_label, dyn_image_label

    Label(root, text = 'Trajectories:').place(x = 60, y = 150)
    gp_traj_label = Label(root, text = 'GP,')
    gp_traj_label.place(x = 130, y = 150)
    dp_traj_label = Label(root, text = 'DP')
    dp_traj_label.place(x = 150, y = 150)
    datasets_label = Label(root, text = 'No dynamic datasets found', fg = 'red')
    datasets_label.place(x = 60, y = 170)
    pneumotach_found_label = Label(root, text = 'No pneumotach file found', fg = 'red')
    pneumotach_found_label.place(x = 60, y = 190)

    dyn_image_photo_img = ImageTk.PhotoImage(Image.fromarray(np.zeros((1,1,3), dtype = np.uint8)))
    dyn_image_label = Label(root, image = dyn_image_photo_img)
    dyn_image_label.place(x = 470, y = 200)

def setup_menus():
    global data_menu_var, date_menu_var, ID_menu_var
    global data_menu, date_menu, ID_menu
    global rootdir_label, usegpu

    options = [f for f in listdir(g.basefolder) if not f.startswith('.')]

    ID_menu_var = StringVar()
    ID_menu_var.trace('w', ID_callback)
    ID_menu = OptionMenu(root, ID_menu_var, *options)
    ID_menu.place(x = 60, y = 115)

    date_menu_var = StringVar()
    date_menu_var.trace('w', date_callback)
    date_menu = OptionMenu(root, date_menu_var, *options, command = set_ID_menu)
    date_menu.place(x = 40, y = 85)

    data_menu_var = StringVar()
    data_menu_var.trace('w', data_callback)
    data_menu = OptionMenu(root, data_menu_var, *options, command = set_date_menu)
    data_menu.place(x = 20, y = 55)

    Label(root, text = "Data root dir:").place(x = 0, y = 30)
    rootdir_label = Label(root, text = g.basefolder)
    rootdir_label.place(x = 150, y = 30)

    def select_file():
        global g
        selectedfolder = filedialog.askdirectory(title = 'dynamic data directory', initialdir = '/')
        if(len(selectedfolder) > 1):
            g.basefolder = selectedfolder
            rootdir_label.config(text = g.basefolder)
            set_data_menu()
    Button(root, text = 'change', command = select_file).place(x = 80, y = 27)

    usegpu = cuda.is_available()
    if(usegpu):
        Label(root, text = cuda.get_current_device()).place(x = 40, y = 3)
        usegpu_cb = Checkbutton(text = 'Use')
        usegpu_cb.select()
        usegpu_cb.place(x = 0, y = 0)
    else:
        Label(root, text = 'No GPU available').place(x = 0, y = 0)

    set_data_menu()


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    setup_window()              # setup TK window
    setup_figures()             # canva draw
    setup_binning_controls()    # set volume bins
    setup_param_entries()       # setup parameters to be passed to mrd file header
    setup_exclude_ranges()
    setup_action_buttons()
    setup_status_labels()       # must come before setup_menus (ID_callback uses these labels)
    setup_menus()               # load function triggers ID_callback via StringVar trace

    binning_plot_update()
    ur_fig_update(urimgtype.NONE, [])
    buttons_update()
    root.after(100, movie_update, 0)
    root.mainloop()


if __name__ == '__main__':
    main()