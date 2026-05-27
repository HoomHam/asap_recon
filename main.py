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


# CREATE GLOBAL VARIABLE CLASSES
g = gvar()
g_raw = raw()
g_traj = traj()
g_res = results()
g_dir = gdir()

# MAIN WINDOW OBJECT
root = Tk()
root.title('HPX dynamic data analysis')
root.geometry('1300x600')
# DISPLAYABLE ELEMENTS IN THE MAIN WINDOW
dyn_unreg_frames = []

# BUTTONS AND CONTROLS
# Calc LV button
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

calcLV_method = StringVar(None, 'Pneumotach')
calcLV_pneumo = Radiobutton(root, text = 'Pneumotach', value = 'Pneumotach', variable = calcLV_method, command=calcLVcb)
calcLV_pneumo.place(x = 360, y = 185)
calcLV_signal = Radiobutton(root, text = 'Signal', value = 'Signal', variable = calcLV_method, command=calcLVcb)
calcLV_signal.place(x = 360, y = 203)
calcLV_diaphragm = Radiobutton(root, text = 'Diaphragm', value = 'Diaphragm', variable = calcLV_method, command=calcLVcb)
calcLV_diaphragm.place(x = 360, y = 221)
def getbintype():
    bt = bintype.SIGNAL if calcLV_method.get() == 'Signal' else bintype.PNEUMOTACH
    return(bintype.DIAPHRAGM if calcLV_method.get() == 'Diaphragm' else bt)

# Show Bin or Volume radio buttons
def showLVcb(): binning_plot_update()
showLV_var = StringVar(None, 'Volume')
showLV_volume = Radiobutton(root, text='Volume', value='Volume', variable=showLV_var, command=showLVcb)
showLV_volume.place(x = 360, y = 60)
showLV_bin = Radiobutton(root, text='Bin', value='Bin', variable=showLV_var, command=showLVcb)
showLV_bin.place(x = 360, y = 78)

# Matrix size entry
Label(root, text = 'Matrix size =').place(x = 10, y = 230)
MS_entry = Entry(root, width = 3)
MS_entry.insert(0, str(g.MS))
MS_entry.place(x = 95, y = 230)

# Image size entry
Label(root, text = 'Image size  =').place(x = 10, y = 250)
IS_entry = Entry(root, width = 3)
IS_entry.insert(0, str(g.IS))
IS_entry.place(x = 95, y = 250)

# Number of bins entry
Label(root, text = '# of bins   =').place(x = 10, y = 270)
nbins_entry = Entry(root, width = 3)
nbins_entry.insert(0, str(g.nbins))
nbins_entry.place(x = 95, y = 270)

# x,y,z kernel width
Label(root, text = 'Kernel dx  =').place(x = 10, y = 290)
xyz_kernel_entry = Entry(root, width = 3)
xyz_kernel_entry.insert(0, '1')
xyz_kernel_entry.place(x = 95, y = 290)

# t kernel width
Label(root, text = 'Kernel dt  =').place(x = 10, y = 310)
t_kernel_entry = Entry(root, width = 3)
t_kernel_entry.insert(0, '1')
t_kernel_entry.place(x = 95, y = 310)

# gas phase line broadening
Label(root, text = 'GP LB =').place(x = 10, y = 330)
gplb_entry = Entry(root, width = 3)
gplb_entry.insert(0, '300')
gplb_entry.place(x = 95, y = 330)

# dissolved phase line broadening
Label(root, text = 'DP LB =').place(x = 10, y = 350)
dplb_entry = Entry(root, width = 3)
dplb_entry.insert(0, '40')
dplb_entry.place(x = 95, y = 350)

# filter point
Label(root, text = 'Freq filter =').place(x = 10, y = 370)
freqfilter_entry = Entry(root, width = 3)
freqfilter_entry.insert(0, '0')
freqfilter_entry.place(x = 95, y = 370)

# exclude ranges entry
excludranges_cb_disable = False
def excluderanges_cb(var, index, mode):
    global exclude_ranges_disable
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

def dyn_recon_cb(): dyn_recon_function()
dyn_recon_button = Button(root, text = 'Recon', fg = 'green', command = dyn_recon_cb)
dyn_recon_button.place(x = 250, y = 150)

def calc_T1RF_cb(): calc_T1RF_function()
calc_T1RF_button = Button(root, text = 'Calc T1RF', fg = 'green', command = calc_T1RF_cb)
calc_T1RF_button.place(x = 250, y = 200)


# HELPER FUNCTIONS
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



# functions for selecting only date/time compatible entries in a directory list
def isdt(x):
    try:
        parse(x)
    except:
        return(False)
    return(True)

def getdtlist(x):
    return([i for i in x if isdt(i)])

# reset the choices in a popup menu
def refresh_menu(thismenu, selectedstring, newoptions):
    # reset var and delete all old options
    selectedstring.set('')
    thismenu['menu'].delete(0, 'end')
    for choice in newoptions:
        thismenu['menu'].add_command(label = choice, command = _setit(selectedstring, choice))
    if(len(newoptions) > 0):
        selectedstring.set(newoptions[0])



# DATA DISPLAY FIGURES
# raw data plot figure
dataset_figure = Figure(figsize = (6.5, 2), dpi = 100)
dataset_figure.tight_layout()
dataset_plot = dataset_figure.add_subplot(111)
dataset_plot.set_yticks([])
dataset_canvas = FigureCanvasTkAgg(dataset_figure, master = root)  
dataset_canvas.get_tk_widget().place(x = 470, y = 0)

# ur (upper right) plot figure
class urimgtype(IntEnum):
    NONE = 0
    COLORMAP = 1
    BWMAP = 2
    COLORMAPDIAPHRAGM = 3
    COMPLEXPLOT = 4
ur_figure = Figure(figsize = (2, 2), dpi = 100)
ur_figuretype = urimgtype.NONE
ur_figure.tight_layout()
ur_plot = ur_figure.add_subplot(111)
ur_plot.set_xticks([])
ur_plot.set_yticks([])
ur_canvas = FigureCanvasTkAgg(ur_figure, master = root)
ur_canvas.get_tk_widget().place(x = 1070, y = 0)

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

def dyn_recon_function():
    global g_res
    global g_raw
    global g_traj
    global dyn_unreg_frames
    global g

    check_recon_entries()

    print('calculating full-res b')
    g_res.calcb(g, g_raw, g_traj, True)
    ur_fig_update(urimgtype.COLORMAP, g_res.bmag[int(g.IS/2), :, :])

    bt = getbintype()
    if(not len(g_raw.ilvbin[bt]) == len(g_raw.ilvtime)):
        print('bin with selected type first')
        return
    # do recon
    g_res.dyn_recon(g, g_raw, g_traj, g_raw.ilvbin[bt], True)

    if(g_res.hasimg(imgtype.DPDYN)):
        print('reconstructed dissolved ratio = ', np.sum(np.real(g_res.getimg(imgtype.DPDYN))) / np.sum(np.imag(g_res.getimg(imgtype.DPDYN))))
        print('spectrum dissolved ratio = ', g_raw.RBCTPratio[0])
    # save
    savemat(g_dir.datadirname + 'gpdyn.mat', {'gpdyn': g_res.getimg(imgtype.GPDYN)})
    savemat(g_dir.datadirname + 'dpdyn.mat', {'dpdyn': g_res.getimg(imgtype.DPDYN)})
    # make movie
    gp_scaling = np.max(g_res.getimg(imgtype.GPDYN)) * .8
    gp_noise = np.mean(g_res.getimg(imgtype.GPDYN)[:, 0:5, 0:5, 0:5])
    if(g_res.hasimg(imgtype.DPDYN)):
        dp_scaling = np.max(np.abs(g_res.getimg(imgtype.DPDYN))) * .9
    totalS = np.sum(g_res.getimg(imgtype.GPDYN))
    ctridx = [np.sum(np.sum(g_res.getimg(imgtype.GPDYN), axis = (0, 2, 3)) * np.array(range(0, g.IS))) / totalS, \
            np.sum(np.sum(g_res.getimg(imgtype.GPDYN), axis = (0, 1, 3)) * np.array(range(0, g.IS))) / totalS, \
            np.sum(np.sum(g_res.getimg(imgtype.GPDYN), axis = (0, 1, 2)) * np.array(range(0, g.IS))) / totalS]
    ctridx = [x if x > g.IS / 3 else int(g.IS / 3) + 1 for x in ctridx]
    ctridx = [x if x < g.IS * 2 / 3 else int(g.IS * 2 / 3) for x in ctridx]
    fig = plt.figure(frameon = False)
    fig.set_size_inches(16, 6, forward = False)
    pic = np.zeros((3 * g.IS, 8 * g.IS))
    im = plt.imshow(pic, interpolation = 'none', vmin = 0, vmax = 1, cmap = 'jet', aspect = 1)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    step = int(g.IS / 30)
    def animate_func(ibin):
        print('in animate ', ibin)
        global dyn_unreg_frames
        for icol in range(0, 8):
            if(ibin < g.nbins):
                gpimg = g_res.getimg(imgtype.GPDYN)[ibin, :, int(ctridx[1] + (icol - 3.5) * step), :]
                pic[0:g.IS, (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(gpimg / gp_scaling, 0)
                gpimg = g_res.getimg(imgtype.GPDYN)[ibin, int(ctridx[0] + (icol - 3.5) * step), :, :]
                pic[g.IS:(2*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(gpimg / gp_scaling, 0)
                gpimg = g_res.getimg(imgtype.GPDYN)[ibin, :, :, int(ctridx[2] + (icol - 3.5) * step)]
                pic[(2*g.IS):(3*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(gpimg / gp_scaling, 0)
            elif(ibin < g.nbins * 2):
                dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - g.nbins, :, int(ctridx[1] + (icol - 3.5) * step), :])
                pic[0:g.IS, (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
                dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - g.nbins, int(ctridx[0] + (icol - 3.5) * step), :, :])
                pic[g.IS:(2*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
                dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - g.nbins, :, :, int(ctridx[2] + (icol - 3.5) * step)])
                pic[(2*g.IS):(3*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
            else:
                if(g.species == species.RABBIT):
                    dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - 2 * g.nbins, :, int(ctridx[1] + (icol - 3.5) * step), :]).copy()
                    gpimg = np.real(g_res.getimg(imgtype.GPDYN)[ibin - 2 * g.nbins, :, int(ctridx[1] + (icol - 3.5) * step), :]).copy() + 1E-9
                    gpimg[gpimg<(gp_scaling / 5)] = 1E+9
                    pic[0:g.IS, (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling / gpimg * gp_scaling / 3, 0)
                    dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - 2 * g.nbins, int(ctridx[0] + (icol - 3.5) * step), :, :]).copy()
                    gpimg = np.real(g_res.getimg(imgtype.GPDYN)[ibin - 2 * g.nbins, int(ctridx[0] + (icol - 3.5) * step), :, :]).copy() + 1E-9
                    gpimg[gpimg<(gp_scaling / 5)] = 1E+9
                    pic[g.IS:(2*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling / gpimg * gp_scaling / 3, 0)
                    dpimg = np.real(g_res.getimg(imgtype.DPDYN)[ibin - 2 * g.nbins, :, :, int(ctridx[2] + (icol - 3.5) * step)]).copy()
                    gpimg = np.real(g_res.getimg(imgtype.GPDYN)[ibin - 2 * g.nbins, :, :, int(ctridx[2] + (icol - 3.5) * step)]).copy() + 1E-9
                    gpimg[gpimg<(gp_scaling / 5)] = 1E+9
                    pic[(2*g.IS):(3*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling / gpimg * gp_scaling / 3, 0)
                else:
                    dpimg = np.imag(g_res.getimg(imgtype.DPDYN)[ibin - 2*g.nbins, :, int(ctridx[1] + (icol - 3.5) * step), :]) 
                    pic[0:g.IS, (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
                    dpimg = np.imag(g_res.getimg(imgtype.DPDYN)[ibin - 2*g.nbins, int(ctridx[0] + (icol - 3.5) * step), :, :])
                    pic[g.IS:(2*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
                    dpimg = np.imag(g_res.getimg(imgtype.DPDYN)[ibin - 2*g.nbins, :, :, int(ctridx[2] + (icol - 3.5) * step)])
                    pic[(2*g.IS):(3*g.IS), (icol * g.IS):((icol + 1) * g.IS)] = np.rot90(dpimg / dp_scaling, 0)
        pic[pic < 0] = 0
        pic[pic >= 1] = .999
        thisimg = np.zeros((pic.shape[0], pic.shape[1], 3), dtype = np.uint8)
        thisimg[:, :, 0] = (pic * 255).astype(np.uint8)
        thisimg[:, :, 1] = (pic * 255).astype(np.uint8)
        thisimg[:, :, 2] = (pic * 255).astype(np.uint8)
        if(g.species == species.RABBIT and ibin >= g.nbins * 2):
            for j in range(0, pic.shape[0]):
                for k in range(0, pic.shape[1]):
                    f = pic[j,k] * 255
                    if(f < 0):
                        idx = 0
                    elif(f > 255):
                        idx = 255
                    else:
                        idx = np.uint8(f)
                    carray = cm.jet(range(256))[idx][0:3]
                    thisimg[j,k,:] = (carray * 255).astype(np.uint8)
        newthisimg = np.resize(thisimg, (thisimg.shape[0] * 2, thisimg.shape[1] * 2, 3))
        for j in range(0, newthisimg.shape[0]):
            for k in range(0, newthisimg.shape[1]):
                newthisimg[j,k,:] = thisimg[int(j/2), int(k/2),:]
        img = Image.fromarray(newthisimg)
        photo_img = ImageTk.PhotoImage(img)
        dyn_unreg_frames.append(photo_img)
        im.set_array(pic)
        return [im]
    dyn_unreg_frames = []
    anim = animation.FuncAnimation(fig, animate_func, frames = g.nbins * (3 if g_res.hasimg(imgtype.DPDYN) else 1), interval = 100) # 10 fps
    writergif = animation.PillowWriter(fps=1)
    anim.save(g_dir.datadirname + 'dynamic_unreg.gif', writer = writergif)
    buttons_update()

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

ur_plot_colors = ['b', 'k', 'r', 'm']
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

dyn_image_photo_img = ImageTk.PhotoImage(Image.fromarray(np.zeros((1,1,3), dtype = np.uint8)))
dyn_image_label = Label(root, image = dyn_image_photo_img)
dyn_image_label.place(x = 470, y = 200)

    
Label(root, text = 'Trajectories:').place(x = 60, y = 150)
gp_traj_found = 0
gp_traj_label = Label(root, text = 'GP,')
gp_traj_label.place(x = 130, y = 150)
dp_traj_found = 0
dp_traj_label = Label(root, text = 'DP')
dp_traj_label.place(x = 150, y = 150)
datasets_label = Label(root, text = 'No dynamic datasets found', fg = 'red')
datasets_label.place(x = 60, y = 170)
pneumotach_found_label = Label(root, text = 'No pneumotach file found', fg = 'red')
pneumotach_found_label.place(x = 60, y = 190)


def ID_callback(*args):
    global g_dir
    global gp_traj_label
    global dp_traj_label
    global datasets_label
    datatype = data_menu_var.get()
    g_dir.trajdirname = g.basefolder + datatype + '/traj/'

    g_dir.datadirname = g.basefolder + datatype + '/' + date_menu_var.get() + '/' + ID_menu_var.get() + '/'
    allrawdir = listdir(g_dir.datadirname)
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

# dataset ID selector
def set_ID_menu():
    options = listdir(g.basefolder + '/' + data_menu_var.get() + '/' + date_menu_var.get())
    refresh_menu(ID_menu, ID_menu_var, options)

# dataset date selector
def set_date_menu():
    options = listdir(g.basefolder + '/' + data_menu_var.get())
    options = getdtlist(options)
    refresh_menu(date_menu, date_menu_var, options)
    set_ID_menu()
    
# dataset data type selector
def set_data_menu():
    options = listdir(g.basefolder)
    refresh_menu(data_menu, data_menu_var, options)
    set_date_menu()

options = listdir(g.basefolder) 

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
set_data_menu()

# add rootdir selector
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
rootdir_selector = Button(root, text = 'change', command = select_file).place(x = 80, y = 27)

# GPU selector
usegpu = cuda.is_available()
if(usegpu):
    Label(root, text = cuda.get_current_device()).place(x = 40, y = 3)
    usegpu_cb = Checkbutton(text = 'Use')
    usegpu_cb.select()
    usegpu_cb.place(x = 0, y = 0)
else:
    Label(root, text = 'No GPU available').place(x = 0, y = 0)


# load dataset and plot
def loadfunction():
    global g_raw
    global g_traj
    check_recon_entries()
    g_traj.load(g_dir.gptrajfilename, g_dir.dptrajfilename, 32)
    g_raw.load(g, g_traj, g_dir.dyndatasets, g_dir.dyndatasetslen, g_dir.fileformat, g_dir.pneumodatasets, g_dir.pneumodatasetslen)
    if(len(g_raw.sspect)):
        ur_fig_update(urimgtype.COMPLEXPLOT, [np.real(g_raw.sspect[0]), np.real(g_raw.sspectfit[0]), np.imag(g_raw.sspect[0]), np.imag(g_raw.sspectfit[0])])
    g_res = results()
    binning_plot_update()
    buttons_update()

load_button = Button(root, text = '  Load  ', fg = 'green', command = loadfunction)
load_button.place(x = 250, y = 95)

# Execute tkinter
binning_plot_update()
ur_fig_update(urimgtype.NONE, [])
buttons_update()
root.after(100, movie_update, 0)
root.mainloop()
