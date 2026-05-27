### cuda recon functions ###

from numba import cuda
from math import exp
from math import cos
from math import sin
from math import fabs
from math import sqrt


usegpu = True
eps = 1.0E-16
bindist0sq = 2
bxsz = 2


def conditional_decorator(func):
    if(usegpu):
        return cuda.jit(func)
    else:
        return func

@conditional_decorator
def bessi0(x):
    ax = fabs(x)
    if ax < 3.75:
        y = x / 3.75
        y *= y
        ans = 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492 + y*(0.2659732 + y*(0.360768e-1 + y*0.45813e-2)))))
    else:
        y = 3.75 / ax
        ans = 0.39894228 + y*(0.1328592e-1 + y*(0.225319e-2 + y*(-0.157565e-2 + y*(0.916281e-2 + \
                y*(-0.2057706e-1 + y*(0.2635537e-1 + y*(-0.1647633e-1 + y*0.392377e-2)))))))
        ans *= (exp(ax) / sqrt(ax))
    return(ans)

#@conditional_decorator 
#def cudarenorm(k, knorm):
#    # renormalizes the MSxMSxMS kr and ki arrays w.r.t. knorm
#    for idx in range(cuda.grid(1) if usegpu else 0, len(k), len(k) if usegpu else 1):
#        k[idx] /= knorm[idx]

@conditional_decorator
def cudarezero(k, knorm):
    # rezeros the MSxMSxMS kspace array
    for idx in range(cuda.grid(1) if usegpu else 0, len(k), len(k) if usegpu else 1):
        k[idx] = 0.0
        knorm[idx] = eps

@conditional_decorator
def cudarenorm(params, k, knorm):
    # idx1 = ix * MS**2 + iy * MS + iz
    # dx = ix - cx, ixp = cx - (ix - cx) = 2 * cx - ix
    # idx2 = (2 * cx - ix) * MS**2 + (2 * cy - iy) * MS + (2 * cz - iz)
    #      = 2 * cx * MX**2 + 2 * cy * MS + 2 * cz - idx1 = 2 * c * (MS**2+MS+1)
    oldway = 1
    cidx = params[0]
    if(oldway):
        for idx in range(cuda.grid(1) if usegpu else 0, len(k), len(k) if usegpu else 1):
            k[idx] /= knorm[idx]
            # TEMPORARY KLUGE
            if(knorm[idx] < 1E-5):
                k[idx] = 0
            # END KLUGE
        return

    for idx in range(cuda.grid(1) if usegpu else 0, cidx + 1, len(k) if usegpu else 1):
        symmidx = 2 * cidx - idx 
        if(idx == cidx or symmidx > len(k)):
            k[idx] /= knorm[idx]
            k[idx] = 0
        else:
            thisk = k[idx]
            symmk = k[symmidx]
            thisk = (thisk + symmk.real - 1j * symmk.imag) / (knorm[idx] + knorm[symmidx]) 
            k[idx] = thisk
            k[symmidx] = thisk.real - 1j * thisk.imag

@conditional_decorator
def cudarecon(params, trajx, trajy, trajz, binarr, raw, k, knorm):
    # idxarr is the arrray of indices to distribute the rawdata into
    # wtarr is the array of weights (same)
    # rawr, rawi are the real, imag parts of one channel's measurements
    # kspace arrays are MS x MS x MS
    # params array is MS, MS center idx, numiquesmp, npts, nbins, ibin, idxoff, smoothing
    MS = params[0]
    MScenter = params[1]
    MS2 = MS**2
    nuniquesmp = params[2]
    npts = params[3]
    nbins = params[4]
    ibin = params[5]
    idxoff = params[6]
    smoothing = params[7]
    binwt = 1.0
    fwt = 1.0
    for idx in range(cuda.grid(1) if usegpu else 0, len(raw), len(raw) if usegpu else 1):
        if(smoothing > eps):
            fwt = exp(-((idx % npts) / smoothing)**2)
        rawval = raw[idx] * fwt
        if(abs(rawval) < eps):
            continue   # skip filtered noise spikes and low SNR points
        kidx = (idx + idxoff) % nuniquesmp
        if(nbins > 1):
            ilvnum = int(idx / npts + 0.5)
            binctr = binarr[ilvnum]
            if(binctr < 0):
                continue
            bindist = min(fabs(ibin - binctr), fabs(ibin + nbins - binctr), \
                    fabs(ibin - nbins - binctr))
            binwt = exp(-bindist * bindist / bindist0sq)
        thiskx = trajx[kidx] # k space matrix point x, y, z
        thisky = trajy[kidx]
        thiskz = trajz[kidx]
        cx = int(thiskx + 0.5) # closest x, y, z indices
        cy = int(thisky + 0.5)
        cz = int(thiskz + 0.5)
        xrng = range(max(0, cx - bxsz), min(MS, cx + bxsz))
        yrng = range(max(0, cy - bxsz), min(MS, cy + bxsz))
        zrng = range(max(0, cz - bxsz), min(MS, cz + bxsz))
        #cdist = (trajx[kidx] - MS/2)**2 + (trajy[kidx]-MS/2)**2+(trajz[kidx]-MS/2)**2
        #beta = 16
        #bessi0beta = bessi0(beta)
        kdist0sq = .2
        for ix in xrng:
            for iy in yrng:
                for iz in zrng:
                    dsq = (thiskx - ix)**2 + (thisky - iy)**2 + (thiskz - iz)**2
                    thiswt = binwt * exp(-dsq / kdist0sq)
                    #thiswt = 0.0
                    #if dsq < bxsz**2:
                    #    thiswt = binwt * bessi0(beta*sqrt(1-dsq / bxsz**2)) / bessi0beta
                    karridx = ix * MS2 + iy * MS + iz
                    cuda.atomic.add(k.real, karridx, thiswt * rawval.real)
                    cuda.atomic.add(k.imag, karridx, thiswt * rawval.imag)
                    cuda.atomic.add(knorm,  karridx, thiswt)
