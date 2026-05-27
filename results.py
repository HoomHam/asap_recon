import numpy as np
from gtypes import gvar, imgtype, graddir, species
from numba import cuda
from recon import cudarezero, cudarenorm, cudarecon
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# HELPER FUNCTIONS
def rangeoverlap(f1, f2, rg):
    # returns True if the range from (float) f1 to f2 interesects the range rg
    return((f1 > rg.start and f1 < rg.stop) or (f2 > rg.start and f2 < rg.stop) or (f1 < rg.start and f2 > rg.stop))

def bin(val):
    v = val.copy()
    breathdir = np.zeros(len(v))
    if(v[1] < v[0]): breathdir[0] = 1
    for iilv in range(1, len(v)):
        if(v[iilv] < v[iilv - 1]):
            breathdir[iilv] = 1
    iv = np.sort(v[breathdir == 1])
    ev = np.sort(v[breathdir == 0])
    v[breathdir == 1] = max(ev) + max(iv) - v[breathdir == 1]
    vsort = np.sort(v)
    bins = np.zeros(len(v))
    for iilv in range(0, len(v)):
        bins[iilv] = np.searchsorted(vsort, v[iilv])
    bins /= len(v)
    return(bins)

class results:
    def __init__(self):
        self.all = [[], [], [], []]
        self.b = []
        self.bmag = []
        self.T1RFimg = []
        self.T1RFimgtime = []
        # allocated memory blocks to keep between calls
        self.kspace = []
        self.kspacenorm = []
        self.d_kspace = []
        self.d_kspacenorm = []

    def getimg(self, itype): return(self.all[itype.value])
    def setimg(self, itype, img): self.all[itype.value] = img
    def hasimg(self, itype): return(len(self.all[itype.value]) > 0)

    def check_arrays(self, matsize, nch, usegpu):
        if(matsize == 0):
            self.kspace = []
            self.kspacenorm = []
        elif(not (len(self.kspace) == matsize**3)):
            self.kspace = np.ascontiguousarray(np.zeros(matsize**3), dtype = 'complex64')
            self.kspacenorm = np.ascontiguousarray(np.zeros(matsize**3), dtype = 'float32')
        else:
            return
        self.d_kspace = cuda.to_device(self.kspace) if(usegpu) else self.kspace
        self.d_kspacenorm = cuda.to_device(self.kspacenorm) if(usegpu) else self.kspacenorm

    def calcb(self, g, g_raw, g_traj, usegpu):
        self.check_arrays(g.MS, g_raw.nch, usegpu) 
        g_traj.rescale_to_MS(g.MS, g.IS)
        # params array is MS, MS center idx, numiquesmp, npts, nbins, ibin, idxoff, smoothing
        reconparams = np.array([g.MS, g.get_MScenter(), g_traj.nuniquesmp, g_raw.npts, 1, 0, 0, g.gplb])

        # send arrays to gpu. kspace arrays are already there
        d_reconparams = cuda.to_device(np.ascontiguousarray(reconparams)) if(usegpu) else params
        d_renormparams = cuda.to_device(np.array([g.get_MScenter() * (g.MS**2 + g.MS + 1)], dtype = 'int'))
        d_trajx = cuda.to_device(g_traj.gettraj(imgtype.GPDYN, graddir.X)) if(usegpu) else g_traj.gettraj(imgtype.GPDYN, graddir.X)
        d_trajy = cuda.to_device(g_traj.gettraj(imgtype.GPDYN, graddir.Y)) if(usegpu) else g_traj.gettraj(imgtype.GPDYN, graddir.Y)
        d_trajz = cuda.to_device(g_traj.gettraj(imgtype.GPDYN, graddir.Z)) if(usegpu) else g_traj.gettraj(imgtype.GPDYN, graddir.Z)
        d_binarr = cuda.to_device(np.zeros(1)) if(usegpu) else np.zeros(1)
        # mask out excluded points
        mask = np.ones(g_raw.npts * g_raw.ntotalilvs)
        for iilv in range(0, g_raw.ntotalilvs):
            for el in g_raw.excluderanges:
                if(iilv * g_raw.TR >= el.start and iilv * g_raw.TR <= el.stop):
                    mask[(iilv * g_raw.npts):((iilv + 1) * g_raw.npts)] = 0
        # calc b
        threadsperblock = 256
        self.b = np.zeros((g_raw.nch, g.MS, g.MS, g.MS), dtype = 'complex64')
        print('calcb channel ', end = '', flush = True)
        for ich in range(0, g_raw.nch):
            print('%d... ' % ich, end = '', flush = True)
            rawdata = np.ascontiguousarray(np.reshape(g_raw.getimg(imgtype.GPDYN)[:, ich, :], \
                (g_raw.npts * g_raw.ntotalilvs), order = 'F').copy() * mask)
            if(usegpu):
                d_rawdata = cuda.to_device(rawdata)
                cudarezero[int(g.MS**3 / threadsperblock) + 1, threadsperblock](self.d_kspace, self.d_kspacenorm)
                cudarecon[int(len(rawdata) / threadsperblock) + 1, threadsperblock](d_reconparams, d_trajx, d_trajy, \
                        d_trajz, d_binarr, d_rawdata, self.d_kspace, self.d_kspacenorm)
                cudarenorm[int(g.MS**3 / threadsperblock) + 1, threadsperblock](d_renormparams, self.d_kspace, self.d_kspacenorm)
                kspace3d = np.reshape(self.d_kspace.copy_to_host(), (g.MS, g.MS, g.MS), order = 'F')
            else:
                cudarezero(self.kspace, self.kspacenorm)
                cudarecon(params, g_traj.gettraj(imgtype.GPDYN, graddir.X), g_traj.gettraj(imgtype.GPDYN, graddir.Y), \
                        g_traj.gettraj(imgtype.GPDYN, graddir.Z), binarr, rawdata, self.kspace, self.kspacenorm, np.zeros(1, dtype = 'complex64'))
                cudarenorm(self.kspace, self.kspacenorm)
                kspace3d = np.reshape(self.kspace, (g.MS, g.MS, g.MS), order = 'F')
            rspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(kspace3d), axes=(0,1,2)))
            avgnoise = np.mean(np.abs(rspace[0:10,0:10,0:10]), (0,1,2))
            self.b[ich, :, :, :] = rspace / avgnoise
        # Choose region of image to display
        self.b = np.conj(self.b[:, g.ISLL():g.ISUL(), g.ISLL():g.ISUL(), g.ISLL():g.ISUL()])
        self.bmag = np.sqrt(np.sum(np.abs(self.b)**2, 0))
        # fit to get a reasonable angle where the magnitude of b was low
        x, y, z = np.meshgrid(np.linspace(-1, 1, g.IS), np.linspace(-1, 1, g.IS), np.linspace(-1, 1, g.IS))
        u = np.ones((g.IS, g.IS, g.IS))
        x2 = x * x
        y2 = y * y
        z2 = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        fu = np.reshape(u, -1)
        fx = np.reshape(x, -1)
        fy = np.reshape(y, -1)
        fz = np.reshape(z, -1)
        fx2 = np.reshape(x2, -1)
        fy2 = np.reshape(y2, -1)
        fz2 = np.reshape(z2, -1)
        fxy = np.reshape(xy, -1)
        fxz = np.reshape(xz, -1)
        fyz = np.reshape(yz, -1)
        for ich in range(g_raw.nch):
            bmask = (np.abs(self.b[ich, :, :, :]) > np.mean(np.abs(self.b[ich, 0:10,0:10,0:10]) * 10)).astype('int')
            invbmask = 1 - bmask
            fb = np.reshape(self.b[ich, :, :, :], -1)
            fbmask = np.reshape(bmask, -1)
            fbp = fb[fbmask == 1]
            fup = fu[fbmask == 1]
            fxp = fx[fbmask == 1]
            fyp = fy[fbmask == 1]
            fzp = fz[fbmask == 1]
            fx2p = fx2[fbmask == 1]
            fy2p = fy2[fbmask == 1]
            fz2p = fz2[fbmask == 1]
            fxyp = fxy[fbmask == 1]
            fxzp = fxz[fbmask == 1]
            fyzp = fyz[fbmask == 1]
            # rotate flatb to be centered at an angle of zero so that angle doesn't wrap
            meanbangle = np.mean(fbp)
            meanbangle /= np.abs(meanbangle)
            fbp *= np.conj(meanbangle)
            # linear regression
            def fitfun(x, cu, cx, cy, cz, cx2, cy2, cz2, cxy, cyz, cxz):
                return(cu*fup + cx*fxp + cy*fyp + cz*fzp + cx2*fx2p + cy2*fy2p + cz2*fz2p + cxy*fxyp + cyz*fyzp + cxz*fxzp)
            p, popt = curve_fit(fitfun, fup, np.angle(fbp), np.zeros((10)))
            self.b[ich, :, :, :] = bmask * self.b[ich, :, :, :] + invbmask * meanbangle * np.abs(self.b[ich, :, :, :]) * \
                    np.exp(1j*(p[0]*u + p[1]*x + p[2]*y + p[3]*z + p[4]*x2 + p[5]*y2 + p[6]*z2 + p[7]*xy + p[8]*yz + p[9]*xz))
            plt.imshow(np.angle(self.b[ich, :,50,:]))
            plt.show()
        self.b /= np.max(np.abs(self.b), 0)
        print('done')
   
    def T1RF_recon(self, g, g_raw, g_traj, rawdata, usegpu):
        ### calculation of T1RF map
        reconparams = np.array([g.MS, g.get_MScenter(), g_traj.nuniquesmp, g_raw.npts, 1, 0, 0, 100])
        d_reconparams = cuda.to_device(np.ascontiguousarray(reconparams)) if(usegpu) else params
        d_renormparams = cuda.to_device(np.array([g.get_MScenter() * (g.MS**2 + g.MS + 1)], dtype = 'int'))
        d_binarr = cuda.to_device(np.zeros(1)) if(usegpu) else np.zeros(1)
        # send trajectories to gpu if it's available
        trajx = np.array([])
        trajy = np.array([])
        trajz = np.array([])
        while(rawdata.shape[0] * rawdata.shape[2] > len(trajx)):
            trajx = np.append(trajx, g_traj.gettraj(imgtype.GPREF, graddir.X))
            trajy = np.append(trajy, g_traj.gettraj(imgtype.GPREF, graddir.Y))
            trajz = np.append(trajz, g_traj.gettraj(imgtype.GPREF, graddir.Z))
        d_trajx = cuda.to_device(np.ascontiguousarray(trajx[:(rawdata.shape[0] * rawdata.shape[2])])) if(usegpu) else trajx[:(rawdata.shape[0] * rawdata.shape[2])]
        d_trajy = cuda.to_device(np.ascontiguousarray(trajy[:(rawdata.shape[0] * rawdata.shape[2])])) if(usegpu) else trajy[:(rawdata.shape[0] * rawdata.shape[2])]
        d_trajz = cuda.to_device(np.ascontiguousarray(trajz[:(rawdata.shape[0] * rawdata.shape[2])])) if(usegpu) else trajz[:(rawdata.shape[0] * rawdata.shape[2])]
        rspace = np.zeros((g.IS, g.IS, g.IS))
        threadsperblock = 16
        for ich in range(0, g_raw.nch):
            chrawdata = np.ascontiguousarray(np.reshape(rawdata[:, ich, :], rawdata.shape[0] * rawdata.shape[2], order = 'F'))
            if(usegpu):
                d_rawdata = cuda.to_device(chrawdata)
                cudarezero[int(g.MS**3 /  threadsperblock) + 1, threadsperblock](self.d_kspace, self.d_kspacenorm)
                cudarecon[int(len(chrawdata) / threadsperblock) + 1, threadsperblock](d_reconparams, d_trajx, d_trajy, \
                        d_trajz, d_binarr, d_rawdata, self.d_kspace, self.d_kspacenorm)
                cudarenorm[int(g.MS**3 / threadsperblock) + 1, threadsperblock](d_renormparams, self.d_kspace, self.d_kspacenorm)
                kspace3d = np.reshape(self.d_kspace.copy_to_host(), (g.MS, g.MS, g.MS), order = 'F')
            else:
                barf
            thisrspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(kspace3d), axes=(0, 1, 2)))
            #rspace = np.abs(thisrspace[g.ISLL():g.ISUL(), g.ISLL():g.ISUL(), g.ISLL():g.ISUL()])
            rspace += np.real(thisrspace[g.ISLL():g.ISUL(), g.ISLL():g.ISUL(), g.ISLL():g.ISUL()] * self.b[ich, :, :, :])
        self.setimg(imgtype.GPREF, rspace)
        return(True)

    def dyn_usimg_recon(self, g, g_raw, g_traj, usegpu, iusimg):
        ### recon of undersampled images and extraction of diaphragm position
        if(iusimg == 0):
            self.check_arrays(g.MS, g_raw.nch, usegpu)
        ilvperusimg = int(g_traj.nsmpperusimg / g_raw.npts + .1)
        for el in g_raw.excluderanges:
            if(rangeoverlap(iusimg * ilvperusimg * g_raw.TR, (iusimg + 1) * ilvperusimg * g_raw.TR, el)):
                print('skipping iter', iusimg, ilvperusimg * g_raw.TR, el)
                return(False)
        # reconparams array is MS, numiquesmp, npts, nbins, ibin, idxoff, smoothing
        reconparams = np.array([g.MS, g.get_MScenter(), g_traj.nuniquesmp, g_raw.npts, 1, 0, 0, 50])
        d_reconparams = cuda.to_device(np.ascontiguousarray(reconparams)) if(usegpu) else params
        d_renormparams = cuda.to_device(np.array([g.get_MScenter() * (g.MS**2 + g.MS + 1)], dtype = 'int'))
        d_binarr = cuda.to_device(np.zeros(1)) if(usegpu) else np.zeros(1)

        # send trajectories to gpu if it's available
        trajx = g_traj.gettraj(imgtype.GPDYN, graddir.X)
        trajy = g_traj.gettraj(imgtype.GPDYN, graddir.Y)
        trajz = g_traj.gettraj(imgtype.GPDYN, graddir.Z)
        trajstartpt = (iusimg * g_traj.nsmpperusimg) % g_traj.nuniquesmp
        d_trajx = cuda.to_device(np.ascontiguousarray(trajx[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)])) if(usegpu) \
                else trajx[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)]
        d_trajy = cuda.to_device(np.ascontiguousarray(trajy[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)])) if(usegpu) \
                else trajy[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)]
        d_trajz = cuda.to_device(np.ascontiguousarray(trajz[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)])) if(usegpu) \
                else trajz[trajstartpt:(trajstartpt + g_traj.nsmpperusimg)]
        rspace = np.zeros((g.IS, g.IS, g.IS))
        threadsperblock = 16
        for ich in range(0, g_raw.nch):
            rawdata = np.ascontiguousarray(np.reshape(g_raw.getimg(imgtype.GPDYN)[:, ich, \
                    (iusimg * ilvperusimg):((iusimg + 1) * ilvperusimg)], g_traj.nsmpperusimg, order = 'F'))
            if(np.abs(rawdata[10]) == 0.0):
                print('skipping')
                continue
            if(usegpu):
                d_rawdata = cuda.to_device(rawdata)
                cudarezero[int(g.MS**3 /  threadsperblock) + 1, threadsperblock](self.d_kspace, self.d_kspacenorm)
                cudarecon[int(len(rawdata) / threadsperblock) + 1, threadsperblock](d_reconparams, d_trajx, d_trajy, \
                        d_trajz, d_binarr, d_rawdata, self.d_kspace, self.d_kspacenorm)
                cudarenorm[int(g.MS**3 / threadsperblock) + 1, threadsperblock](d_renormparams, self.d_kspace, self.d_kspacenorm)
                kspace3d = np.reshape(self.d_kspace.copy_to_host(), (g.MS, g.MS, g.MS), order = 'F')
            else:
                barf
            thisrspace = np.fft.fftshift(np.fft.fftn(kspace3d, axes=(0, 1, 2)))
            rspace += np.real(thisrspace[g.ISLL():g.ISUL(), g.ISLL():g.ISUL(), g.ISLL():g.ISUL()] * self.b[ich, :, :, :])
        self.setimg(imgtype.GPDYN, np.flip(rspace, axis=0))
        return(True)

    def dyn_recon(self, g, g_raw, g_traj, bins, usegpu):
        ### Binned reconstruction
        self.check_arrays(g.MS, g_raw.nch, usegpu)
        g_traj.rescale_to_MS(g.MS, g.IS)
        reconparams = np.array([g.MS, g.get_MScenter(), g_traj.nuniquesmp, g_raw.npts, g.nbins, 0, 0, 0])
        binarr = np.ascontiguousarray(bins * g.nbins)
        mask = np.ones(g_raw.npts * g_raw.ntotalilvs)
        for iilv in range(0, g_raw.ntotalilvs):
            for el in g_raw.excluderanges:
                if(iilv * g_raw.TR >= el.start and iilv * g_raw.TR <= el.stop):
                    mask[(iilv * g_raw.npts):((iilv + 1) * g_raw.npts)] = 0
        # send arrays to gpu. kspace arrays are already there
        d_renormparams = cuda.to_device(np.array([g.get_MScenter() * (g.MS**2 + g.MS + 1)], dtype = 'int'))
        d_binarr = cuda.to_device(binarr) if(usegpu) else binarr
        threadsperblock = 256
        for itype in [imgtype.GPDYN, imgtype.DPDYN]:
            if(not g_raw.hasimg(itype)):
                continue
            if(itype==imgtype.GPDYN):
                np.save('trajx', g_traj.gettraj(itype, graddir.X))
                np.save('trajy', g_traj.gettraj(itype, graddir.Y))
                np.save('trajz', g_traj.gettraj(itype, graddir.Z))
                np.save('acq', np.reshape(g_raw.getimg(itype)[:,0,:], g_raw.npts*g_raw.ntotalilvs, order='F')*mask)
                np.save('bins', binarr)
            d_trajx = cuda.to_device(g_traj.gettraj(itype, graddir.X)) if(usegpu) else g_traj.gettraj(itype, graddir.X)
            d_trajy = cuda.to_device(g_traj.gettraj(itype, graddir.Y)) if(usegpu) else g_traj.gettraj(itype, graddir.Y)
            d_trajz = cuda.to_device(g_traj.gettraj(itype, graddir.Z)) if(usegpu) else g_traj.gettraj(itype, graddir.Z)
            # bin
            rspace = np.zeros((g.nbins, g.IS, g.IS, g.IS), dtype = 'complex' if itype == imgtype.DPDYN else 'double')
            for ich in range(0, g_raw.nch):
                rawdata = np.ascontiguousarray(np.reshape(g_raw.getimg(itype)[:, ich, :].copy(), \
                        (g_raw.npts * g_raw.ntotalilvs), order = 'F') * mask)
                d_rawdata = cuda.to_device(rawdata) if(usegpu) else rawdata
                print('bin channel %d,' % ich, 'bin 0', end = '', flush = True)
                for ibin in range(0, g.nbins):
                    if(ibin > 0):
                        print(',%d' % ibin, end = '' if(ibin < g.nbins - 1) else '\n', flush = True)
                    reconparams = np.array([g.MS, g.get_MScenter(), g_traj.nuniquesmp, g_raw.npts, g.nbins, ibin, 0, \
                            g.gplb if(itype == imgtype.GPDYN) else g.dplb])
                    if(usegpu):
                        d_reconparams = cuda.to_device(np.ascontiguousarray(reconparams)) if(usegpu) else reconparams
                        cudarezero[int(g.MS**3 /  threadsperblock) + 1, threadsperblock](self.d_kspace, self.d_kspacenorm)
                        cudarecon[int(len(rawdata) / threadsperblock) + 1, threadsperblock](d_reconparams, d_trajx, d_trajy, \
                                d_trajz, d_binarr, d_rawdata, self.d_kspace, self.d_kspacenorm)
                        cudarenorm[int(g.MS**3 / threadsperblock) + 1, threadsperblock](d_renormparams, self.d_kspace, self.d_kspacenorm)
                        kspace3d = (np.reshape(self.d_kspace.copy_to_host(), (g.MS, g.MS, g.MS), order = 'F'))
                    else:
                        barf
                    thisrspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(kspace3d), axes=(0, 1, 2)))
                    #plt.imshow(np.angle(thisrspace[g.ISLL():g.ISUL(),120,g.ISLL():g.ISUL()]))
                    #plt.imshow(np.angle(self.b[ich,:,50,:]))
                    #plt.show()
                    # phase such that on-resonance components are all real, and scale by relative channel sensitivity
                    thisrspace = thisrspace[g.ISLL():g.ISUL(), g.ISLL():g.ISUL(), g.ISLL():g.ISUL()] * self.b[ich, :, :, :]
                    rspace[ibin, :, :, :] += np.real(thisrspace) if itype == imgtype.GPDYN else thisrspace
                    # SAVE IT FOR NOW
                    print('saving', ibin)
                    np.save('savedbin'+str(ibin), rspace[ibin, :, :, :])
                    # END SAVE IT
            if(itype == imgtype.DPDYN):
                # mask based on gas phase
                for ibin in range(0, g.nbins):
                    noisethresh = np.mean(np.abs(self.getimg(imgtype.GPDYN)[:, 0:5, 0:5, 0:5])) * 5
                    mask = (self.getimg(imgtype.GPDYN)[ibin, :, :, :].copy() > noisethresh).astype('int')
                    #rspace[ibin, :, :, :] *= mask
                    for iph in range(0, 314*2):
                        ph = iph / 100.0
                        thisrspace = rspace[ibin, :, :, :] * np.exp(1j * ph)
                        # split into the two (RBC, TP) frequency components
                        phaseRBC = g_raw.fRBC[0] * g_raw.TEeff * 2 * np.pi
                        phaseTP = g_raw.fTP[0] * g_raw.TEeff * 2 * np.pi
                        # thisrspace = aRBC * exp(1j * phaseRBC) + aTP * exp(1j * phaseTP)
                        # Re(thisrspace) = aRBC * cos(phaseRBC) + aTP * cos(phaseTP)
                        # Im(thisrspace) = aRBC * sin(phaseRBC) + aTP * sin(phaseTP)
                        aTP = (np.real(thisrspace) / np.cos(phaseRBC) - np.imag(thisrspace) / np.sin(phaseRBC)) / \
                                (np.cos(phaseTP) / np.cos(phaseRBC) - np.sin(phaseTP) / np.sin(phaseRBC))
                        aRBC = (np.real(thisrspace) / np.cos(phaseTP) - np.imag(thisrspace) / np.sin(phaseTP)) / \
                                (np.cos(phaseRBC) / np.cos(phaseTP) - np.sin(phaseRBC) / np.sin(phaseTP))
                        sumaTP = np.sum(aTP)
                        sumaRBC = np.sum(aRBC)
                        ratio = np.sum(aRBC*mask) / np.sum(aTP*mask)
                        R = sumaRBC / sumaTP
                        if(iph > 0 and sumaTP > 0.0 and sumaRBC > 0.0 and \
                                ((R - g_raw.RBCTPratio[0]) * (lastR - g_raw.RBCTPratio[0]) < 0.0)):
                            print('SETTING rspace')
                            rspace[ibin, :, :, :] = aRBC + 1j * aTP
                            break
                        lastR = R
                        print(ph, ratio)
            self.setimg(itype, rspace)

    def register(usegpu):
        usegpu = usegpu

