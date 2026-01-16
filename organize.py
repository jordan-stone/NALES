import os
import glob
import fnmatch
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.dates import MinuteLocator, DateFormatter
from scipy.stats import norm, sigmaclip
import pickle
from scipy.signal import savgol_filter
import gc
from NALES import jFits
from NALES.fix_pix import correct_with_precomputed_neighbors, find_neighbors, get_bpm

def get_flags(directory='.'):
    '''Read the fits files in a directory and return
    lists of the filenames, timestamps, flags, and parallactic angles.
    The flags are encoded according to:
    {'PRI':0,'DRK':1,'SEC':2,'SKY':3}
    INPUT:
    a directory as a string, defaults to the working 
    directory
    RETURNS:
    fns, times, flags, paras'''
    fns0 = sorted(fnmatch.filter(os.listdir(directory), '*.fits'))
    if len(fns0) == 0:
        fns0 = sorted(fnmatch.filter(os.listdir(directory),'*.fits.gz'))
    fns = [directory+'/'+f for f in fns0]
    times = []
    flags = []
    paras = []
    flag_nums = {'PRI':0, 
                 'NOD_A':0,
                 'DRK':1, 
                 '':1,
                 'SEC':2, 
                 'SKY':3, 
                 'NOD_B':3,
                 'SCI':4}
    for f in fns:
        print(f)
        h, d = jFits.get_fits_array(f)
        try:
            times.append(datetime.strptime(h['DATE-OBS']+'T'+
                                           h['TIME-OBS'],
                                           '%Y-%m-%dT%H:%M:%S.%f'))
        except ValueError:
            try:
                times.append(datetime.strptime(h['DATE-OBS']+'T'+
                                               h['TIME-OBS'],
                                               '%Y-%m-%dT%H:%M:%S'))
            except ValueError:
                times.append(datetime.strptime(h['DATE-OBS']+'T'+
                                               h['TIME-OBS'],
                                               ' %Y-%m-%dT %H:%M:%S:%f'))
        flags.append(flag_nums[h['FLAG'].strip()])
        paras.append(h['LBT_PARA'])
    return fns, times, flags, paras

def get_tdiffs(times):
    tdiffs = []
    for ii in range(1, len(times)):
        tdiffs.append((times[ii] - times[ii-1]).total_seconds())
    return np.array(tdiffs)

def organize_blocks(fns, times, flags, paras, plot=True, breakdiff=20):
    '''separate files into their respective cycles'''
    time_diffs = get_tdiffs(times)
    break_times = np.where(time_diffs > (breakdiff))[0]#20 secs by eye, could automate...
    break_times += 1
    break_times = np.r_[0, break_times, len(fns)]
    blockedfns = []
    blockedts = []
    blockedflags = []
    blockedparas = []
    for ii in range(1, len(break_times)):
        blockedfns.append(fns[break_times[ii-1] : break_times[ii]])
        blockedts.append(times[break_times[ii-1] : break_times[ii]])
        blockedflags.append(flags[break_times[ii-1] : break_times[ii]])
        blockedparas.append(paras[break_times[ii-1] : break_times[ii]])
    if plot:
        fig = mpl.figure()
        ax = fig.add_subplot(111)
        for t, f in zip(blockedfns, blockedflags):
            fns = [int(name.split('_')[2].split('.')[0]) for name in t]
            ax.plot(fns, f, marker='.')
        fig.savefig('blocksvfnumber.png')
        fig = mpl.figure()
        ax = fig.add_subplot(111)
        for t, f in zip(blockedts, blockedflags):
            ax.plot(t, f, marker='.')
        ax = mpl.gca()
        ax.xaxis.set_major_locator(MinuteLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=80)
        fig.savefig('blocksvstime.png')
        mpl.show()
    return list(zip(blockedfns, blockedts, blockedflags, blockedparas))

def get_cycles(block,SCI=False):
    flags = np.array(block[2])
    pri_inds = np.where(flags==0)[0]
    if SCI:
        pri_inds = np.where(flags==4)[0]
    pri_inds_diff = np.diff(pri_inds)
    cycle_boundaries = [pri_inds[ii+1] for ii in np.where(pri_inds_diff > 1)[0]]
    cycle_breaks = np.r_[0,cycle_boundaries,len(block[2])].astype(int)
    cyclefns = []
    cyclets = []
    cycleflags = []
    cycleparas = []
    for ii in range(1, len(cycle_breaks)):
        cyclefns.append(block[0][cycle_breaks[ii-1] : cycle_breaks[ii]])
        cyclets.append(block[1][cycle_breaks[ii-1] : cycle_breaks[ii]])
        cycleflags.append(block[2][cycle_breaks[ii-1] : cycle_breaks[ii]])
        cycleparas.append(block[3][cycle_breaks[ii-1] : cycle_breaks[ii]])
    return list(zip(cyclefns, cyclets, cycleflags, cycleparas))

#def make_cycle_medians(cycle, cds=True):
#    flag_names = {0:'PRI', 1:'DRK', 2:'SEC', 3:'SKY', 4:'SCI'}
#    fns = np.array(cycle[0])
#    cyclets = np.array(cycle[1])
#    cycleflags = np.array(cycle[2])
#    cycleparas = np.array(cycle[3])
#    medians={}
#    for flag in np.unique(cycleflags):
#        fns_flag = fns[cycleflags==flag]
#        arrs = []
#        for f in fns_flag:
#            h, d = jFits.get_fits_array(f)
#            if cds:
#                arrs.append(d[1].astype(float)-d[0].astype(float))#cds
#            else:
#                arrs.append(d[1].astype(float))
#        medians[flag_names[flag]] = (np.median(arrs, axis=0), 
#                                     np.std(arrs, axis=0))
#    return medians

def make_cycle_medians(cycle, noLightUpTo=30, bad_and_neighbors=None, cds=False):
    flag_names = {0:'PRI', 1:'DRK', 2:'SEC', 3:'SKY', 4:'SCI'}
    fns = np.array(cycle[0])
    cyclets = np.array(cycle[1])
    cycleflags = np.array(cycle[2])
    cycleparas = np.array(cycle[3])
    medians={}
    for flag in (0, 1, 2, 3, 4):
        fns_flag = fns[cycleflags==flag]
        if len(fns_flag) < 1:
            medians[flag_names[flag]] = []
        else:
            arrs = []
            arrs_none = []
            jbs = []
            jbs2 = []
            jbs3 = []
            for f in fns_flag:
                h, d0 = jFits.get_fits_array(f)
                d0 = d0.astype(float)
                if cds:
                    d = d0[1]-d0[0]
                else:
                    d = d0[1]
                if bad_and_neighbors is None:
                    ovscn = get_overscan2(d)
                    d-=ovscn[None,:]
                    jail_bars = get_jailbars(d,noLightUpTo=noLightUpTo)
                    arrs.append(d-jail_bars[None,:])
                else:
                    ovscn = get_overscan2(d)
                    d-=ovscn[None,:]
                    correct_with_precomputed_neighbors(d, bad_and_neighbors)
                    jail_bars = get_jailbars(d,noLightUpTo=noLightUpTo)
                    out = d-jail_bars[None,:]

                    correct_with_precomputed_neighbors(out, bad_and_neighbors)
                    jail_bars2 = get_jailbars(out,noLightUpTo=noLightUpTo)
                    out2 = out - jail_bars2[None,:]
                    jail_bars3 = get_jailbars(out2,noLightUpTo=noLightUpTo)
                    jbs.append(jail_bars)
                    jbs2.append(jail_bars2)
                    jbs3.append(jail_bars3)
                    arrs.append(out)
            medians[flag_names[flag]] = (np.median(arrs, axis=0), 
                                         np.std(arrs, axis=0))
    return medians

def process_no_nod_frames(fns_list, noLightUpTo=100, band_and_neighbors=None):
    arrs = []
    for f in fns_list:
        h, d0 = jFits.get_fits_array(f)
        d = d0.astype(float)
        if bad_and_neighbors is None:
            jail_bars = get_jailbars(d,noLightUpTo=noLightUpTo)
            arrs.append(d-jail_bars[None,:])
        else:
            correct_with_precomputed_neighbors(d, bad_and_neighbors)
            jail_bars = get_jailbars(d,noLightUpTo=noLightUpTo)
            out = d-jail_bars[None,:]
            correct_with_precomputed_neighbors(out, bad_and_neighbors)
            arrs.append(out)
    return arrs

def cycle_mean_paras(cycle):
    flag_names = {0:'PRI', 1:'DRK', 2:'SEC', 3:'SKY', 4:'SCI'}
    fns = np.array(cycle[0])
    cyclets = np.array(cycle[1])
    cycleflags = np.array(cycle[2])
    cycleparas = np.array(cycle[3])
    means={}
    for flag in np.unique(cycleflags):
        fns_flag = fns[cycleflags==flag]
        paras = []
        for f in fns_flag:
            h, d = jFits.get_fits_array(f)
            paras.append(h['LBT_PARA'])
        means[flag_names[flag]] = (np.mean(paras))
    return means

def cycle_mean_kw(cycle, kw='LBT_ALT'):
    print('cycle_mean_kw', kw)
    flag_names = {0:'PRI', 1:'DRK', 2:'SEC', 3:'SKY', 4:'SCI'}
    fns = np.array(cycle[0])
    cyclets = np.array(cycle[1])
    cycleflags = np.array(cycle[2])
    cyclekw = np.array(cycle[3])
    means={}
    for flag in (0, 1, 2, 3, 4):
        print(flag)
        fns_flag = fns[cycleflags==flag]
        print(fns_flag)
        kws = []
        for f in fns_flag:
            h, d = jFits.get_fits_array(f)
            kws.append(h[kw])
        try:
            means[flag_names[flag]] = (np.mean(kws))
        except TypeError:
            try:
                kws1 = [datetime.strptime(kw,'%H:%M:%S.%f') for kw in kws]
                kws2 = np.mean([(kw-kws1[0]).total_seconds() for kw in kws1])
                out = kws1[0]+timedelta(seconds=kws2)
                means[flag_names[flag]] = out.strftime('%H:%M:%S.%f')
            except ValueError:
                kws1 = [datetime.strptime(kw,'%H:%M:%S') for kw in kws]
                kws2 = np.mean([(kw-kws1[0]).total_seconds() for kw in kws1])
                out = kws1[0]+timedelta(seconds=kws2)
                means[flag_names[flag]] = out.strftime('%H:%M:%S')
    return means
    
def get_jailbars(im,noLightUpTo=30):
    n_amps = im.shape[1]//64
    snip = im[:noLightUpTo,:]
    meds = np.zeros(im.shape[1])
    for amp in range(n_amps):
        meds[64*amp:64*(amp+1)] = np.median(snip[:,64*amp:64*(amp+1)])
    return meds

def get_overscan2(im):
    n_amps = 2048//64
    #top 4 overscan behave odd
    #snip = np.r_[im[:4,:],im[-4:,:]]
    snip = im[:4,:]
    snip = snip.astype(float)
    snip0 = snip.copy()

    xs = np.arange(2048)
    #columns 127+128*i and 128+128*i behave odd, handle separately
    edges = xs[::128]
    edges2 = xs[127::128]

    snip[:,edges]=np.nan
    snip[:,edges2]=np.nan

    meds = np.zeros(2048)
    for amp in range(n_amps):
        meds[64*amp:64*(amp+1)] = np.nanmedian(snip[:,64*amp:64*(amp+1)])

    for edg in edges:
        meds[edg] = np.median(snip0[:,edg])
    for edg in edges2:
        meds[edg] = np.median(snip0[:,edg])

    meds_out = np.rint(meds).astype(im.dtype)
    return meds_out

def get_vert_overscan(im, savgol_window=31):
    snip = np.c_[im[:,:4],im[:,-4:]]
    snip = snip.astype(float)
    med_snip = np.nanmedian(snip,axis=1)
    out = savgol_filter(med_snip,savgol_window, 3)
    return out

def make_cds_plus(f, outname=None):
    hdu = jFits.pyfits.open(f)
    cds = hdu[0].data[1]-hdu[0].data[0]
    ovscn2 = get_overscan2(cds)
    cds_ovscn = cds - ovscn2[None,:]
    vovscn = get_vert_overscan(cds_ovscn)
    out = cds_ovscn - vovscn[:,None]
    hdu_out = jFits.pyfits.PrimaryHDU(out,header=hdu[0].header)
    if outname is None:
        hdu_out.writeto('cdsplus/cds_'+os.path.basename(f))
    else:
        hdu_out.writeto(outname)
    hdu.close()
    gc.collect()

def do_cds_plus(arr, bad_and_neighbors=None, vertical_overscan=True, jailbars_instead=False):
    arr0 = arr.copy().astype(float)
    cds = arr0[1]-arr0[0]
    if jailbars_instead:
        ovscn2 = get_jailbars(cds)
    else:
        ovscn2 = get_overscan2(cds)
    out = cds - ovscn2[None,:]
    if vertical_overscan:
        vovscn = get_vert_overscan(out)
        out = out - vovscn[:,None]
    if (bad_and_neighbors is not None):
        correct_with_precomputed_neighbors(out, bad_and_neighbors)
    return out


def organize_wavecal_frames(directory='.', maxNdarks=100):
    fns=sorted( glob.glob(directory+'/*.fits') )
    if len(fns) == 0:
        fns = sorted( glob.glob(directory+'/*.fits.gz') )

    os.mkdir('nb29')
    nb29s = {}
    os.mkdir('nb33')
    nb33s = {}
    os.mkdir('nb35')
    nb35s = {}
    os.mkdir('nb39')
    nb39s = {}
    os.mkdir('darks')
    darks = {}

    for f in fns:
        print(f)
        hdu = jFits.pyfits.open(f)
        h = hdu[0].header
        d = hdu[0].data
        hdu.close()
        del hdu
        gc.collect()
        #h, d = jFits.get_fits_array(f)
        texp = float(h['EXPTIME'])
        if h['lmir_FW4'].startswith('Blank'):    
            if texp in darks.keys():
                darks[texp].append(d)
                os.symlink('../../'+f,'darks/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
            else:
                darks[texp]=[d]
                os.mkdir('darks/'+'{:3.2f}'.format(texp))
                os.symlink('../../'+f,'darks/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
        else:
            if h['lmir_FW2'].startswith('NB29'):
                if texp in nb29s.keys():
                    nb29s[texp].append(d)
                    os.symlink('../../'+f,'nb29/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
                else:
                    nb29s[texp] = [d]
                    os.mkdir('nb29/'+'{:3.2f}/'.format(texp))
                    os.symlink('../../'+f,'nb29/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
            elif h['lmir_FW2'].startswith('NB33'):
                if texp in nb33s.keys():
                    nb33s[texp].append(d)
                    os.symlink('../../'+f,'nb33/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
                else:
                    nb33s[texp] = [d]
                    os.mkdir('nb33/'+'{:3.2f}/'.format(texp))
                    os.symlink('../../'+f,'nb33/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
            elif h['lmir_FW2'].startswith('NB35'):
                if texp in nb35s.keys():
                    nb35s[texp].append(d)
                    os.symlink('../../'+f,'nb35/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
                else:
                    nb35s[texp] = [d]
                    os.mkdir('nb35/'+'{:3.2f}/'.format(texp))
                    os.symlink('../../'+f,'nb35/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
            elif h['lmir_FW2'].startswith('NB39'):
                if texp in nb39s.keys():
                    nb39s[texp].append(d)
                    os.symlink('../../'+f,'nb39/'+'{:3.2f}/'.format(texp)+os.path.basename(f))
                else:
                    nb39s[texp] = [d]
                    os.mkdir('nb39/'+'{:3.2f}/'.format(texp))
                    os.symlink('../../'+f,'nb39/'+'{:3.2f}/'.format(texp)+os.path.basename(f))

    med_dark = {}
    for txp in darks.keys():
        med_dark[txp] = np.median(darks[txp][:maxNdarks], axis=0)
        jFits.pyfits.writeto('darks/median_{:3.2f}.fits'.format(txp), med_dark[txp])

    #make bad and neighbors tuned with this night's darks.
    max_txp = np.max(tuple(darks.keys()))
    mx=np.max(darks[max_txp][-1] - darks[max_txp][0],axis=0)#cds
    mxm = np.zeros(mx.shape)
    mxc = mx - np.median(mx, axis=0)[None, :]#subtract column average
    mxcr = mxc - np.median(mxc, axis=1)[:, None]#subtract row average
    percent_chance = norm.ppf(1-0.01*(1./2048**2))
    ca, l, u = sigmaclip(mxcr, low=percent_chance, high=percent_chance)
    mxm[mxcr > u] = 1
    bpm = get_bpm()
    bpm_and_these_hots = np.logical_or(mxm, bpm).astype(int)
    jFits.pyfits.writeto('bpm_and_these_hots.fits', bpm_and_these_hots)
    bad_and_neighbors = find_neighbors(bpm_and_these_hots)
    fo = open('bad_and_neighbors_bpm_and_these_darks.pkl', 'wb')
    pickle.dump(bad_and_neighbors, fo)
    fo.close()

    for txp in nb29s.keys():
        try:
            ds29txp = np.median(nb29s[txp], axis=0) - med_dark[txp]
            cdsp_ds29txp = do_cds_plus(ds29txp, bad_and_neighbors=bad_and_neighbors)
            jFits.pyfits.writeto('nb29/{:3.2f}/nb29_median_ds.fits'.format(txp), ds29txp)
            jFits.pyfits.writeto('nb29/{:3.2f}/cdsp_nb29_median_ds.fits'.format(txp), cdsp_ds29txp)
        except KeyError:
            'print no dark with t_exp={}'.format(txp)

    for txp in nb33s.keys():
        try:
            ds33txp = np.median(nb33s[txp], axis=0) - med_dark[txp]
            cdsp_ds33txp = do_cds_plus(ds33txp, bad_and_neighbors=bad_and_neighbors)
            jFits.pyfits.writeto('nb33/{:3.2f}/nb33_median_ds.fits'.format(txp), ds33txp)
            jFits.pyfits.writeto('nb33/{:3.2f}/cdsp_nb33_median_ds.fits'.format(txp), cdsp_ds33txp)
        except KeyError:
            'print no dark with t_exp={}'.format(txp)

    for txp in nb35s.keys():
        try:
            ds35txp = np.median(nb35s[txp], axis=0) - med_dark[txp]
            cdsp_ds35txp = do_cds_plus(ds35txp, bad_and_neighbors=bad_and_neighbors)
            jFits.pyfits.writeto('nb35/{:3.2f}/nb35_median_ds.fits'.format(txp), ds35txp)
            jFits.pyfits.writeto('nb35/{:3.2f}/cdsp_nb35_median_ds.fits'.format(txp), cdsp_ds35txp)
        except KeyError:
            'print no dark with t_exp={}'.format(txp)

    for txp in nb39s.keys():
        try:
            ds39txp = np.median(nb39s[txp], axis=0) - med_dark[txp]
            cdsp_ds39txp = do_cds_plus(ds39txp, bad_and_neighbors=bad_and_neighbors)
            jFits.pyfits.writeto('nb39/{:3.2f}/nb39_median_ds.fits'.format(txp), ds39txp)
            jFits.pyfits.writeto('nb39/{:3.2f}/cdsp_nb39_median_ds.fits'.format(txp), cdsp_ds39txp)
        except KeyError:
            'print no dark with t_exp={}'.format(txp)
