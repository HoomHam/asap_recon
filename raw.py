import os
from sys import stderr
import numpy as np

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import specfit                      # dissolved-phase spectral fit (vendored from XeCS calspec, 2026-09-24)
from gtypes import gvar, imgtype, graddir, bintype, species


def gaussfit(x, h, w, c):
    return(h * np.exp(-(x - c)**2 / w**2))

def lorfit(x, f0, f1, a0, a1, ph0, ph1, w0, w1):
    res = a0 * np.exp(1j * ph0) / (1 + 1j * (x - f0) / w0) + a1 * np.exp(1j * ph1) / (1 + 1j * (x - f1) / w1) 
    return(np.concatenate((np.real(res), np.imag(res))))

class traj:
    def __init__(self):
        self.all = [[[], [], []], [[], [], []], [[], [], []], [[], [], []]]
        self.nuniquesmp = 0          # number of unique sample points in a trajectory
        self.nsmpperusimg = 0
        self.npts = 0
        self.kgp = []
        self.kdp = []
        self.killpts = 2
        self.BW = 1 / 10E-6 # spiral acquisition bandwidth in Hz
        self.FOV = 350 # FOV in mm
        self.spectBW = 1 / 60E-6 # spectrum acquisition in Hz
        self.sfrq = 17.666 # MHz @ 1.5T
    def gettraj(self, itype, dir): return(self.all[itype.value][dir.value])

    def load_traj_from_npfile(self, gpfilename, dpfilename, nusimg):
        """
        From numpy file, load gas phase and dynamic phase trajectory as 1D array
        Args:
            - gpfilename: filepath to gas phase npy file
            - dpfilename: filepath to dynamic phase npy file
            - nusimg: 
        """
        # load from np files
        if not os.path.exists(gpfilename):
            raise FileNotFoundError(f'Gas phase trajectory file not found: {gpfilename}')
        if not os.path.exists(dpfilename):
            raise FileNotFoundError(f'Dissolved phase trajectory file not found: {dpfilename}')
   
        kgp = np.load(gpfilename)
        print(f'Loaded gas phase trajectory with shape={kgp.shape}', file=stderr)
        kdp = np.load(dpfilename)    
        print(f'Loaded dynamic phase trajectory with shape={kdp.shape}', file=stderr)
        self.load_traj_from_array(kgp, kdp, nusimg)

    def load_traj_from_array(self, gp_array, dp_array, nusimg):
        """
        From 1D np.array of gp and dp, fill in the traj fields with acquired sample points
        Args:
            - gp_array: 1d np.array of gas phase data
            - dp_array: 1d np.array of dissolved in tissue phase trajectory
            - nusimage: number of images to use?
        """
        self.kgp = gp_array * self.FOV # in units of delta-k
        # no dissolved trajectory (gas-only sequence): empty (0, 3) kdp -> rescale_to_MS skips DPDYN
        self.kdp = dp_array * self.FOV if dp_array is not None else np.zeros((0, gp_array.shape[1]))
        # figure out number of points per interleave by looking at the periodicity of the trajectory
        abskgp = np.sum(self.kgp**2, 1)
        absfftkgp = np.abs(np.fft.fft(abskgp))
        for idx in range(1, len(absfftkgp)):
            if(absfftkgp[idx] > absfftkgp[0] / 10):
                break
        self.npts = int(len(abskgp) / idx + 0.1)
        # kill initial points 
        self.nuniquesmp = 0
        for idx in range(int(self.kgp.shape[0] / self.npts + 0.1)):
            if(idx == 0):
                tempkgp = self.kgp[self.killpts:self.npts].copy()
                tempkdp = self.kdp[self.killpts:self.npts].copy()
            else:
                tempkgp = np.concatenate((tempkgp, self.kgp[(idx * self.npts + self.killpts):((idx + 1) * self.npts)].copy()))
                tempkdp = np.concatenate((tempkdp, self.kdp[(idx * self.npts + self.killpts):((idx + 1) * self.npts)].copy()))
            self.nuniquesmp += (self.npts - self.killpts)
        self.nsmpperusimg = int(self.nuniquesmp / nusimg + 0.1)
        self.kgp = tempkgp
        self.kdp = tempkdp
        self.npts -= self.killpts
        self.all = [[[], [], []], [[], [], []], [[], [], []], [[], [], []]]

    def rescale_to_MS(self, MS, IS):
        if(len(self.kgp) > 0):
            for idir in [graddir.X.value, graddir.Y.value, graddir.Z.value]:
                #self.all[imgtype.GPDYN.value][idir] = np.ascontiguousarray(self.kgp[:, int(idir)] * MS / 2 + MS / 2)
                self.all[imgtype.GPDYN.value][idir] = np.ascontiguousarray(self.kgp[:, int(idir)] * MS / IS + MS / 2)
                self.all[imgtype.GPREF.value][idir] = self.all[imgtype.GPDYN.value][idir]
        if(len(self.kdp) > 0):
            for idir in [graddir.X.value, graddir.Y.value, graddir.Z.value]:
                #self.all[imgtype.DPDYN.value][idir] = np.ascontiguousarray(self.kdp[:, int(idir)] * MS / 2 + MS / 2)
                self.all[imgtype.DPDYN.value][idir] = np.ascontiguousarray(self.kdp[:, int(idir)] * MS / IS + MS / 2)
                self.all[imgtype.DPREF.value][idir] = self.all[imgtype.DPDYN.value][idir]
    def nuniqueusimg(self): return(int(self.nuniquesmp / self.nsmpperusimg + 0.1)) 
    
class raw:
    def __init__(self):
        self.all = [[], [], [], []]  # all images in order gpdyn, dpdyn, gpref, dpref (if present)
        self.allstartidx = [0, 0, 0, 0]
        self.nch = 0                 # number of channels
        self.npts = 0                # number of points per interleave
        self.nuniqueilvs = 0         # number of unique interleaves in sampling pattern
        self.ntotalilvs = 0          # total number of interleaves acquired
        self.TR = 0.0                # TR in sec
        self.TE = 0.0                # TE in sec
        self.TEeff = 0.0             # TE effective in sec
        self.dphiRBCTP = 0.0         # RBC-TP phase difference at the image k0 sample (rad), from specfit
        self.specfit = {}            # the specfit.fit_block result (CSV fields) for the output metadata
        self.DPoff = 0.0             # DP offset frequency in ppm
        self.fTP = []
        self.fRBC = []
        self.RBCTPratio = []
        self.ilvperTR = 0            # interleaves per TR
        self.ilvtime = []            # time since trigger for each interleave
        self.ilvvol = [[], [], []]   # NORMALIZED lung volume at time of each interleave
        self.ilvbin = [[], [], []]
        self.bintime = []            # time after EE assigned to each bin
        self.volmeastime = [[], [], []]
        self.volmeasvol = [[], [], []] # UNNORMALIZED
        self.volmeasEEtime = [[], [], []]             # identified time of each EE
        self.excluderanges = []
        self.sspect = []
        self.fitsspect = []
        self.sspectfreq = []
        self.sspectfit = []
        self.RBCphase = []
        self.TPphase = []
        self.TEphase = []
        self.deltaphase = 0.0
    def allimg(self): return(elem for elem in self.all if len(elem) > 0)
    def alldynimg(self): return([elem for elem in [self.all[imgtype.GPDYN.value], self.all[imgtype.DPDYN.value]] if len(elem) > 0])
    def allrefimg(self): return([elem for elem in [self.all[imgtype.GPREF.value], self.all[imgtype.DPREF.value]] if len(elem) > 0])
    def allgpimg(self): return([elem for elem in [self.all[imgtype.GPDYN.value], self.all[imgtype.DPDYN.value]] if len(elem) > 0])
    def alldpimg(self): return([elem for elem in [self.all[imgtype.GPREF.value], self.all[imgtype.DPREF.value]] if len(elem) > 0])
    def hasimg(self, type): return(len(self.all[type.value]) > 0)
    def getimg(self, type): return(self.all[type.value])
    def setimg(self, type, img, startidx): self.all[type.value] = img; self.allstartidx[type.value] = startidx
    def getstartidx(self, type): return(self.allstartidx[type.value])
    def rescale(self, v):
        vgood = [x for x in v if not np.isnan(x)]
        if(len(vgood) < 1):
            return
        mn = np.min(vgood)
        mx = np.max(vgood)
        vgood -= mn
        if(mx > mn):
            vgood /= (mx - mn)
        vidx = 0
        for idx in range(0, len(v)):
            if(not np.isnan(v[idx])):
                v[idx] = vgood[vidx]
                vidx += 1
    def bin(self, val):
        bins = val.copy()
        v = np.array([x for x in val if not np.isnan(x)])
        if(len(v) > 2):
            breathdir = np.zeros(len(v))
            if(v[1] < v[0]): 
                breathdir[0] = 1
            for iilv in range(1, len(v)):
                if(v[iilv] < v[iilv - 1]):
                    breathdir[iilv] = 1
            iv = np.sort(v[breathdir == 1])
            ev = np.sort(v[breathdir == 0])
            maxiv = 0.0 if len(iv) == 0 else max(iv) * 1.00001
            maxev = 0.0 if len(ev) == 0 else max(ev)
            v[breathdir == 1] = maxiv + maxev - v[breathdir == 1]
            vsort = np.sort(v)
            vidx = 0
            for iilv in range(0, len(bins)):
                if(np.isnan(bins[iilv])):
                    continue
                bins[iilv] = np.searchsorted(vsort, v[vidx])
                vidx += 1
            bins[np.isnan(bins)] = -1
            bins /= len(v)
        return(bins)
    
    
    def load_pneumotach_from_arrays(self, t, P):
        """
        Pneumotach NdArray parsed by the converter: t = time in seconds
        (zeroed to first packet), P = pressure. Smooths, crops to the
        acquisition window and integrates pressure into (unnormalized) volume.
        Call after self.ilvtime is set.
        """
        bt = bintype.PNEUMOTACH
        t = np.asarray(t, dtype = 'float64').copy()
        P = np.asarray(P, dtype = 'float64').copy()
        if(len(P) > 51):
            P = savgol_filter(P, 51, 2)
        keep = (t > self.ilvtime[0]) & (t < self.ilvtime[len(self.ilvtime) - 1])
        P = P[keep]
        t = t[keep]
        self.volmeastime[bt] = t
        self.volmeasvol[bt] = np.zeros(len(P))
        for j in range(0, len(t) - 1):
            self.volmeasvol[bt][j + 1] = self.volmeasvol[bt][j] + P[j]

    def load_from_arr(self, traj, ref_acq_arr, dyn_acq_arr, pneumo_arr, fileformat, meta=None):
        self.__init__()
        noisespikethresh = 10
        # load raw datafiles, identify gas/dissolved excitation pattern, split into
        # gas and dissolved
        if fileformat == 'mrd_siemens' or fileformat == 'siemens':
            # acquisition arrays arrive channels-first (nch, npts, nilv); transpose to
            # the internal (npts, nch, nilv) ordering used throughout the recon
            if ref_acq_arr is not None:
                ref_acq_arr = np.ascontiguousarray(ref_acq_arr.transpose(1, 0, 2))
                self.npts = ref_acq_arr.shape[0]
                self.nch = ref_acq_arr.shape[1]
            if dyn_acq_arr is not None:
                dyn_acq_arr = np.ascontiguousarray(dyn_acq_arr.transpose(1, 0, 2))
                self.npts = dyn_acq_arr.shape[0]
                self.nch = dyn_acq_arr.shape[1]
                # parse acquisition metadata up front (the spectrum fit below uses self.TE)
                numspec_raw = 0
                dtdyn = 0.0
                dtspec = 0.0
                if fileformat == 'siemens':
                    self.TR = meta[('alTR', '0')] * 1.0E-6 # s
                    self.TE = meta[('alTE', '0')] * 1.0E-6 # s
                    self.DPoff = meta[('sWipMemBlock', 'adFree', '2')]
                    dtdyn = meta[('sRXSPEC', 'alDwellTime', '0')] * 1.0E-9 # s
                    try:
                        dtdyn = meta[('sRXSPEC', 'alDwellTime', '1')] * 1.0E-9
                    except:
                        pass
                    try:
                        dtspec = meta[('sWipMemBlock', 'alFree', '12')] * 1.0E-6 # s
                    except:
                        pass
                    try:
                        numspec_raw = int(meta[('sWipMemBlock', 'alFree', '10')] * \
                                meta[('sWipMemBlock', 'alFree', '11')] + 0.1)
                    except:
                        print(f'sWipMemBlock not found, setting numspec = 0', file=stderr)
                elif fileformat == 'mrd_siemens':
                    # plain dict built from the MRD header user parameters, all in SI units
                    self.TR = meta.get('TR', 0.0)
                    self.TE = meta.get('TE', 0.0)
                    self.DPoff = meta.get('DPoff', 0.0)
                    dtdyn = meta.get('dtdyn', 0.0)
                    dtspec = meta.get('dtspec', 0.0)
                    numspec_raw = int(meta.get('numspec', 0))
                if(np.fabs(self.TR - 0.0223) < 1E-6):
                    self.TR = 22.26E-3 ### KLUGE FOR NOW, IT ROUNDS IN THE FILE!
                for ich in range(self.nch):
                    std = np.std(np.real(dyn_acq_arr[:, ich, :]))
                    y, x = np.histogram(np.real(dyn_acq_arr), range=(-3*std, 3*std), bins=int(np.prod(dyn_acq_arr[:, ich, :].shape) / 1000))
                    popt, pcov = curve_fit(gaussfit, x[:-1], y, p0=(np.max(y), std, 0.0))
                    dyn_acq_arr[:, ich, :] /= popt[1]
                # separate out initial spectrum acquisition
                # TEMP KLUGE BECAUSE OF 20 MAGIC COOKIE!!!!!!!!!!!!!
                numspec = int(numspec_raw / 20 * traj.nsmpperusimg / self.npts + .1) if numspec_raw else 0
                # END TEMP KLUGE
                rawspec = dyn_acq_arr[..., :numspec]
                dyn_acq_arr = dyn_acq_arr[..., numspec:]

                # separate out gas and dissolved acquisitions based on pattern of signal intensities.
                # can be just gas, gas/dissolved or gas/dissolved/dissolved
                pat = np.abs(np.fft.fft(dyn_acq_arr[0, 0, :])) # FFT tells you acq ordering
                if(max(pat[(int(len(pat) / 3) - 5):(int(len(pat) / 3) + 5)]) > pat[0] / 5):
                    print(f'identified sampling pattern gas-dissolved-dissolved', file=stderr)
                    self.setimg(imgtype.GPDYN, dyn_acq_arr[..., 0:(3 * int(dyn_acq_arr.shape[2] / 3)):3], 0)
                    self.setimg(imgtype.DPDYN, dyn_acq_arr[..., 1:(3 * int(dyn_acq_arr.shape[2] / 3)):3] + \
                            dyn_acq_arr[..., 2:(3 * int(dyn_acq_arr.shape[2] / 3)):3], 0)
                    if ref_acq_arr is not None:
                        self.setimg(imgtype.GPREF, ref_acq_arr[..., 0:(3 * int(ref_acq_arr.shape[2] / 3)):3], 0)
                        self.setimg(imgtype.DPREF, ref_acq_arr[..., 1:(3 * int(ref_acq_arr.shape[2] / 3)):3] + \
                                ref_acq_arr[..., 2:(3 * int(ref_acq_arr.shape[2] / 3)):3], 0)
                    self.ilvperTR = 3
                elif(max(pat[(int(len(pat) / 2) - 5):(int(len(pat) / 2) + 5)]) > pat[0] / 3):
                    print(f'identified sampling pattern gas-dissolved', file=stderr)
                    self.setimg(imgtype.GPDYN, dyn_acq_arr[..., 0:(2 * int(dyn_acq_arr.shape[2] / 2)):2], 0)
                    self.setimg(imgtype.DPDYN, dyn_acq_arr[..., 1:(2 * int(dyn_acq_arr.shape[2] / 2)):2], 0)
                    if ref_acq_arr is not None:
                        self.setimg(imgtype.GPREF, ref_acq_arr[..., 0:(2 * int(ref_acq_arr.shape[2] / 2)):2], 0)
                        self.setimg(imgtype.DPREF, ref_acq_arr[..., 1:(2 * int(ref_acq_arr.shape[2] / 2)):2], 0)
                    self.ilvperTR = 2
                else:
                    print(f'identified sampling pattern gas-only', file=stderr)
                    self.setimg(imgtype.GPDYN, dyn_acq_arr, 0)
                    if ref_acq_arr is not None:
                        self.setimg(imgtype.GPREF, ref_acq_arr, 0)
                    self.ilvperTR = 1
                # --- TR (gas-interleave time base) ---
                # self.TR is the spacing between consecutive GAS interleaves for
                # the dynamic time axis (results.dyn_*_recon: t = iilv * TR). It
                # is taken straight from metadata (alTR / MRD 'TR') and used
                # as-is. A gas-only dataset legitimately carries either the
                # minimum TR (half the dual-phase TR) or a non-minimal ~17.2 ms;
                # both are real acquisition timings, so no correction is applied.
                # Logged so the TR actually in force is visible per run.
                print(f'TR (from metadata) = {self.TR * 1e3:.3f} ms, '
                      f'ilvperTR={self.ilvperTR} '
                      f'({"gas-only" if self.ilvperTR == 1 else "dual-phase"})',
                      file=stderr)
                # rephase everything so that the beginning of the fids has zero phase
                for ich in range(self.nch):
                    gasphase = np.mean(self.getimg(imgtype.GPDYN)[0:4, ich, :])
                    gasrephase = np.conj(gasphase) / np.abs(gasphase)
                    self.getimg(imgtype.GPDYN)[:, ich, :] *= gasrephase
                    if(rawspec.size > 0):
                        rawspec[:, ich, :] *= gasrephase
                    if(self.hasimg(imgtype.DPDYN)):
                        self.getimg(imgtype.DPDYN)[:, ich, :] *= gasrephase
                    # force spectra and regular dissolved phase imaging interleaves to have the same phase
                    if(rawspec.size > 0 and self.hasimg(imgtype.DPDYN)):
                        specphase = np.mean(rawspec[0:4, ich, :])
                        dissphase = np.mean(self.getimg(imgtype.DPDYN)[0:4, ich, :])
                        rawspec[:, ich, :] *= np.conj(specphase) / np.abs(specphase) * dissphase / np.abs(dissphase)
                if(numspec > 0):
                    # Dissolved-phase spectral fit = specfit.py (vendored verbatim from the XeCS calspec work,
                    # 2026-09-24): time-domain complex fit of the pooled cal-block FID with a chemical-shift
                    # prior (gas L / RBC L / membrane Voigt [M2] or L+L [M3]), replacing the frequency-domain
                    # two-Lorentzian fit + 15-pseudo-TE polyfit. Why: (1) the old 4-peak picker locked onto
                    # wrong peaks on 7/89 sessions and the unbounded LM fit returned a negative ratio on one
                    # (2steve/05 C); (2) RBC is a shoulder on the membrane line at RBC/mem 0.1-0.3, so an
                    # unconstrained two-line fit is multimodal -- the prior is not optional; (3) the split needs
                    # the RBC - membrane phase difference AT THE IMAGE k0 SAMPLE, which the old code did not give
                    # (it used the t = 0 intercept, F59). specfit returns exactly that (dphiRBCTP), the ratio
                    # a_RBC/|mem| that a two-component split reproduces by construction (RBCTPratio; the
                    # literature's scalar ratio = RBCTPratio / F_lump), and the line frequencies. Details:
                    # workspace/notes/fork_patch_2026-09-24.md.
                    dt_img = dtdyn if dtdyn > 0.0 else 1 / traj.BW
                    dwell = dtspec if dtspec > 0.0 else 1 / traj.spectBW
                    hz_per_ppm = 17.61           # Larmor 17.612 MHz from the headers (traj.sfrq = 17.666 is nominal)
                    for ich in range(self.nch):
                        fids = rawspec[:, ich, :]    # (samples, spectral lines); killpts already trimmed, no apodisation
                        # keep fids whose first 100 points carry >= 2x the signal of the last 100 (xenon present)
                        idx = [ifid for ifid in range(numspec)
                               if np.sum(np.abs(fids[:100, ifid])) > np.sum(np.abs(fids[(self.npts - 100):, ifid])) * 2]
                        # if there are less than 5 spectra, forget it; multi-coil spectra are not fitted (as before)
                        if(len(idx) > 5 and self.nch == 1):
                            kept = fids[:, idx]
                            # per-rep means for the within-block phase SD (one rep = one navigator block of lines)
                            nlin = int(traj.nsmpperusimg / self.npts + 0.1) or 1
                            reps = [kept[:, i:i + nlin] for i in range(0, kept.shape[1] - nlin + 1, nlin)]
                            try:
                                sf = specfit.fit_block(kept, dwell, self.TE, first_sample=traj.killpts,
                                        t_k0_offset_s=traj.killpts * dt_img, hz_per_ppm=hz_per_ppm,
                                        reps=reps if len(reps) >= 3 else None)
                            except Exception as err:
                                print(f'spectral fit failed ({err}) -- no RBC/TP params, dissolved image will be '
                                      f'kept unsplit', file=stderr)
                                continue
                            # stability: refit with one more leading sample dropped (gas line fixed); the M3
                            # membrane decomposition can hinge on a single early sample on RBC-weak blocks
                            # (042DR: 30 deg swing) -- reject when the k0 angle moves by more than 10 deg
                            try:
                                sf2 = specfit.fit_block(kept, dwell, self.TE, first_sample=traj.killpts,
                                        t_k0_offset_s=traj.killpts * dt_img, hz_per_ppm=hz_per_ppm,
                                        gas_hz=sf['gas_hz_from_carrier'], drop_before=traj.killpts + 1)
                                stab = abs(float(np.degrees(np.angle(np.exp(1j * np.radians(
                                        sf['dphi_k0_deg'] - sf2['dphi_k0_deg']))))))
                            except Exception:
                                stab = float('nan')
                            sf['stab_dphi_deg'] = round(stab, 1)
                            gain = 1 / max(abs(np.sin(np.radians(sf['dphi_k0_deg']))), 1e-9)
                            why = [] if sf['valid'] == 'yes' else [sf['reason']]
                            if stab > 10.0:
                                why.append(f'RBC-TP phase unstable ({stab:.1f} deg change when one more leading sample is dropped)')
                            if gain > 1 / 0.3:
                                why.append(f'RBC-TP phase separation {sf["dphi_k0_deg"]:.1f} deg too small for a '
                                           f'one-point split (noise gain {gain:.1f})')
                            print(f'spectral fit ({sf["model_used"]}): gas {sf["gas_ppm_from_carrier"]:.1f} ppm from carrier, '
                                  f'RBC {sf["rbc_ppm"]:.1f} / mem {sf["mem1_ppm"]:.1f}+{sf["mem2_ppm"]:.1f} ppm from gas, '
                                  f'df {sf["df_hz"]:.0f} Hz, RBC/TP lumped {sf["ratio_lumped"]:.3f} (scalar '
                                  f'{sf["ratio_scalar"]:.3f}, F_lump {sf["F_lump"]:.2f}), RBC-TP phase at k0 '
                                  f'{sf["dphi_k0_deg"]:.1f} deg (per-rep SD {sf["dphi_rep_sd_deg"]:.1f}, drop-one-more '
                                  f'change {stab:.1f}, noise gain {gain:.2f}), SNR {sf["snr_diss"]:.0f}', file=stderr)
                            if why:
                                print('spectral fit rejected: ' + '; '.join(why) + ' -- discarding RBC/TP params, '
                                      'dissolved image will be kept unsplit', file=stderr)
                                continue
                            fRBC = sf['gas_hz_from_carrier'] + sf['rbc_ppm'] * hz_per_ppm
                            self.fRBC.append(fRBC)
                            self.fTP.append(fRBC - sf['df_hz'])          # lumped-membrane centroid
                            self.RBCTPratio.append(sf['ratio_lumped'])
                            self.dphiRBCTP = np.radians(sf['dphi_k0_deg'])
                            self.specfit = {k: v for k, v in sf.items() if k != 'fits'}
                            # spectrum + fitted model on the fitted samples, for the GUI plots (main.py)
                            model = sf['model_used']
                            y, t = specfit._prep(kept, dwell, traj.killpts)
                            fitted = specfit.model_fid(sf['fits'][model]['fit']['theta'], t, specfit.MODELS[model])
                            self.sspect.append(np.fft.fftshift(np.fft.fft(y)))
                            self.sspectfit.append(np.fft.fftshift(np.fft.fft(fitted)))
                            self.sspectfreq.append(np.fft.fftshift(np.fft.fftfreq(y.size, dwell)))
                            # legacy fields (nothing downstream needs them any more; kept so old callers do not break)
                            self.RBCphase.append(np.radians(sf['fits'][model]['tb']['rbc']['phi_deg']))
                            self.TPphase.append(np.nan)                  # the membrane phase is the lumped phasor's
                            self.TEphase.append(self.TE)
                            self.deltaphase = self.dphiRBCTP
                            self.TEeff = self.dphiRBCTP / (2 * np.pi * sf['df_hz'])   # so that 2*pi*df*TEeff == dphiRBCTP
                            self.TEeff2 = self.TEeff
        if(fileformat == 'bruker'):
            BHlength = 1    # temporary init value
            self.nch = 1
            self.TR = 10E-3
            self.npts = 800
            self.nuniqueilvs = 640
            npts = int(BHlength / self.TR)
            # look for breath hold
            lns = np.log(np.sum(np.abs(dyn_acq_arr[:6, 0, :]), 0))
            x = np.array(range(0, npts))
            a = np.zeros(len(lns) - npts)
            for j in range(0, len(lns) - npts):
                y = lns[j:(j+npts)]
                p = np.polyfit(x, y, 1)
                diff = np.polyval(p, x) - y
                a[j] = np.sum(diff**2)
            bhstartidx = np.argmin(a)
            bhendidx = bhstartidx + npts
            for j in range(bhendidx, bhstartidx, -1):
                if(lns[j] < lns[bhendidx]):
                    bhendidx = j
            for j in range(bhstartidx, bhendidx):
                if(lns[j] > lns[bhstartidx]):
                    bhstartidx = j
            self.setimg(imgtype.GPREF, dyn_acq_arr[..., bhstartidx:bhendidx].copy(), bhstartidx)
            dyn_acq_arr[..., bhstartidx:bhendidx] = 0.0
            self.setimg(imgtype.GPDYN, dyn_acq_arr, 0)
            self.ntotalilvs = dyn_acq_arr.shape[2]

        # remove fully sampled images with low SNR
        self.nuniqueilvs = int(traj.nuniquesmp / self.npts)
        self.ntotalilvs = self.getimg(imgtype.GPDYN).shape[2]
        avgsig = np.zeros(int(self.ntotalilvs / self.nuniqueilvs) + 1)
        for ifs in range(0, int(self.ntotalilvs / self.nuniqueilvs) + 1):
            avgsig[ifs] = np.sum(np.abs(self.getimg(imgtype.GPDYN)[0:8, :, (ifs * self.nuniqueilvs):((ifs + 1) * self.nuniqueilvs)]))
        lowsig = list((avgsig < np.mean(avgsig) / 2))
        self.excluderanges = []
        rangeend = -1
        while(any(lowsig[rangeend:])):
            rangestart = lowsig.index(True, rangeend + 1)
            if all(lowsig[(rangestart + 1):]):
                rangeend = len(avgsig)
            else:
                rangeend = lowsig.index(False, rangestart + 1)
            self.excluderanges.append(range(int(rangestart * self.nuniqueilvs * self.TR), int(rangeend * self.nuniqueilvs * self.TR) + 1))
        self.ilvtime = np.array(range(0, self.ntotalilvs)) * self.TR
        # remove all sample points > noisespikethresh x the mean for that point
        # across all the interleaves. Intended to filter out the noise spikes
        print(f'filtering noise spikes...', file=stderr)
        for acq in  self.alldynimg():
            nzero = np.zeros(self.npts)
            for ipt in range(0, self.npts):
                for ich in range(0, self.nch):
                    temp = np.abs(acq[ipt, ich, :])
                    nzero[ipt] += np.count_nonzero(acq[ipt, ich, :])
                    temp[temp > np.mean(temp[temp > 0]) * noisespikethresh] = 0
                    acq[ipt, ich, temp == 0] = 0.0
                    nzero[ipt] -= np.count_nonzero(acq[ipt, ich, :])
            print(f'removed {np.sum(nzero)} points (noise spikes) of {np.prod(acq.shape)}', file=stderr)
        if(self.hasimg(imgtype.GPDYN)):
            self.volmeasvol[bintype.SIGNAL] = np.sum(np.abs(self.getimg(imgtype.GPDYN)[:8, :, :]), axis = (0, 1))
            self.volmeasvol[bintype.SIGNAL] -= np.min(self.volmeasvol[bintype.SIGNAL])
            self.volmeasvol[bintype.SIGNAL] /= (np.max(self.volmeasvol[bintype.SIGNAL]) - np.min(self.volmeasvol[bintype.SIGNAL]))
            self.volmeastime[bintype.SIGNAL] = self.ilvtime

        # pneumotach time/pressure NdArray from the converter, if present
        if pneumo_arr is not None:
            self.load_pneumotach_from_arrays(pneumo_arr[0], pneumo_arr[1])
        for bt in [bintype.SIGNAL, bintype.PNEUMOTACH] if len(self.volmeasvol[bintype.PNEUMOTACH]) else [bintype.SIGNAL]:
                # identify minima
                mincnt = 0
                N = 50
                for iter in range(0, 2):
                    for j in range(N, len(self.volmeastime[bt]) - N - 1):
                        for k in range(j - N, j + N + 1):
                            if((not j == k) and self.volmeasvol[bt][k] <= self.volmeasvol[bt][j]):
                                break
                        if(k == j + N):
                            if(iter == 1):
                                self.volmeasEEtime[bt][mincnt] = self.volmeastime[bt][j]
                                minv[mincnt] = self.volmeasvol[bt][j]
                            mincnt += 1
                    if(iter == 0):
                        self.volmeasEEtime[bt] = np.zeros(mincnt)
                        minv = np.zeros(mincnt)
                        mincnt = 0
                if(bt == bintype.PNEUMOTACH):
                    p = np.polyfit(self.volmeasEEtime[bt] - np.mean(self.volmeasEEtime[bt]), minv, 8)
                    self.volmeasvol[bt]-= np.polyval(p, self.volmeastime[bt] - np.mean(self.volmeasEEtime[bt]))
                self.volmeasvol[bt] -= np.min(self.volmeasvol[bt])
                self.volmeasvol[bt] /= np.max(self.volmeasvol[bt])
                self.ilvvol[bt] = np.interp(self.ilvtime, self.volmeastime[bt], self.volmeasvol[bt])
                self.rescale(self.ilvvol[bt])
                if(bt == bintype.PNEUMOTACH):
                    self.ilvbin[bt] = self.bin(self.ilvvol[bt])
                if(bt == bintype.SIGNAL):
                    self.ilvbin[bt] = np.zeros((len(self.ilvtime)))
                    for j in range(0, len(self.ilvtime)):
                        if(len(self.volmeasEEtime[bt]) == 0 or self.ilvtime[j] < self.volmeasEEtime[bt][0] or \
                                self.ilvtime[j] > self.volmeasEEtime[bt][-1]):
                            self.ilvbin[bt][j] = -1
                        else:
                            for k in range(0, len(self.volmeasEEtime[bt]) - 1):
                                if(self.ilvtime[j] >= self.volmeasEEtime[bt][k] and self.ilvtime[j] <= self.volmeasEEtime[bt][k + 1]):
                                    self.ilvbin[bt][j] = (self.ilvtime[j] - self.volmeasEEtime[bt][k]) / \
                                            (self.volmeasEEtime[bt][k + 1] - self.volmeasEEtime[bt][k])

