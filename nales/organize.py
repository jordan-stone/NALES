"""
Data organization and preprocessing utilities for ALES observations.

This module provides functions for organizing raw ALES data files,
processing calibration frames, and preparing data for cube extraction.
"""

import os
import glob
import gc
import fnmatch
import pickle
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MinuteLocator, DateFormatter
from scipy.stats import norm, sigmaclip
from scipy.signal import savgol_filter
from astropy.io import fits

from nales.utils.bad_pixels import (
    correct_with_precomputed_neighbors,
    find_neighbors,
    get_bpm,
)


def get_flags(directory='.'):
    """
    Read FITS files and extract metadata for data organization.
    
    Parses FITS headers to extract timestamps, observation flags,
    and parallactic angles for all files in a directory.
    
    Parameters
    ----------
    directory : str, optional
        Path to directory containing FITS files. Default: '.'
        
    Returns
    -------
    fns : list of str
        Full paths to FITS files.
    times : list of datetime
        Observation timestamps.
    flags : list of int
        Observation type flags:
        - 0: Primary (PRI) or NOD_A
        - 1: Dark (DRK) or blank
        - 2: Secondary (SEC)
        - 3: Sky (SKY) or NOD_B
        - 4: Science (SCI)
    paras : list of float
        Parallactic angles in degrees.
    """
    fns0 = sorted(fnmatch.filter(os.listdir(directory), '*.fits'))
    if len(fns0) == 0:
        fns0 = sorted(fnmatch.filter(os.listdir(directory), '*.fits.gz'))
        
    fns = [os.path.join(directory, f) for f in fns0]
    times = []
    flags = []
    paras = []
    
    flag_nums = {
        'PRI': 0,
        'NOD_A': 0,
        'DRK': 1,
        '': 1,
        'SEC': 2,
        'SKY': 3,
        'NOD_B': 3,
        'SCI': 4
    }
    
    for f in fns:
        print(f"Reading: {f}")
        hdu = fits.open(f)
        h = hdu[0].header
        hdu.close()
        
        # Parse timestamp (try different formats)
        date_obs = h['DATE-OBS']
        time_obs = h['TIME-OBS']
        
        for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', ' %Y-%m-%dT %H:%M:%S:%f']:
            try:
                times.append(datetime.strptime(f'{date_obs}T{time_obs}', fmt))
                break
            except ValueError:
                continue
                
        flags.append(flag_nums[h['FLAG'].strip()])
        paras.append(h['LBT_PARA'])
        
    return fns, times, flags, paras


def get_tdiffs(times):
    """
    Calculate time differences between consecutive observations.
    
    Parameters
    ----------
    times : list of datetime
        Observation timestamps.
        
    Returns
    -------
    tdiffs : numpy.ndarray
        Time differences in seconds.
    """
    tdiffs = []
    for ii in range(1, len(times)):
        tdiffs.append((times[ii] - times[ii - 1]).total_seconds())
    return np.array(tdiffs)


def organize_blocks(fns, times, flags, paras, plot=True, breakdiff=20):
    """
    Separate observations into blocks based on timing gaps.
    
    Groups consecutive observations into blocks, splitting when
    the time gap exceeds a threshold.
    
    Parameters
    ----------
    fns : list of str
        File paths.
    times : list of datetime
        Observation timestamps.
    flags : list of int
        Observation type flags.
    paras : list of float
        Parallactic angles.
    plot : bool, optional
        If True, generate diagnostic plots. Default: True
    breakdiff : float, optional
        Time gap threshold in seconds. Default: 20
        
    Returns
    -------
    blocks : list of tuple
        Each block is (filenames, times, flags, parallactic_angles).
    """
    time_diffs = get_tdiffs(times)
    break_times = np.where(time_diffs > breakdiff)[0]
    break_times += 1
    break_times = np.r_[0, break_times, len(fns)]
    
    blockedfns = []
    blockedts = []
    blockedflags = []
    blockedparas = []
    
    for ii in range(1, len(break_times)):
        blockedfns.append(fns[break_times[ii - 1]:break_times[ii]])
        blockedts.append(times[break_times[ii - 1]:break_times[ii]])
        blockedflags.append(flags[break_times[ii - 1]:break_times[ii]])
        blockedparas.append(paras[break_times[ii - 1]:break_times[ii]])
        
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for t, f in zip(blockedfns, blockedflags):
            file_nums = [int(name.split('_')[2].split('.')[0]) for name in t]
            ax.plot(file_nums, f, marker='.')
        fig.savefig('blocksvfnumber.png')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for t, f in zip(blockedts, blockedflags):
            ax.plot(t, f, marker='.')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MinuteLocator(interval=3))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=80)
        fig.savefig('blocksvstime.png')
        plt.show()
        
    return list(zip(blockedfns, blockedts, blockedflags, blockedparas))


def get_cycles(block, SCI=False):
    """
    Extract individual nod cycles from an observation block.
    
    Parameters
    ----------
    block : tuple
        Output from organize_blocks: (fns, times, flags, paras).
    SCI : bool, optional
        If True, use SCI flag (4) instead of PRI (0). Default: False
        
    Returns
    -------
    cycles : list of tuple
        Each cycle is (filenames, times, flags, parallactic_angles).
    """
    flags = np.array(block[2])
    
    if SCI:
        pri_inds = np.where(flags == 4)[0]
    else:
        pri_inds = np.where(flags == 0)[0]
        
    pri_inds_diff = np.diff(pri_inds)
    cycle_boundaries = [pri_inds[ii + 1] for ii in np.where(pri_inds_diff > 1)[0]]
    cycle_breaks = np.r_[0, cycle_boundaries, len(block[2])].astype(int)
    
    cyclefns = []
    cyclets = []
    cycleflags = []
    cycleparas = []
    
    for ii in range(1, len(cycle_breaks)):
        cyclefns.append(block[0][cycle_breaks[ii - 1]:cycle_breaks[ii]])
        cyclets.append(block[1][cycle_breaks[ii - 1]:cycle_breaks[ii]])
        cycleflags.append(block[2][cycle_breaks[ii - 1]:cycle_breaks[ii]])
        cycleparas.append(block[3][cycle_breaks[ii - 1]:cycle_breaks[ii]])
        
    return list(zip(cyclefns, cyclets, cycleflags, cycleparas))


def make_cycle_medians(cycle, noLightUpTo=30, bad_and_neighbors=None, cds=False):
    """
    Create median-combined images for each observation type in a cycle.
    
    Parameters
    ----------
    cycle : tuple
        Output from get_cycles: (fns, times, flags, paras).
    noLightUpTo : int, optional
        Number of rows without illumination (for jail bar correction). Default: 30
    bad_and_neighbors : list, optional
        Pre-computed bad pixel neighbor info from find_neighbors.
    cds : bool, optional
        If True, apply correlated double sampling. Default: False
        
    Returns
    -------
    medians : dict
        Dictionary with keys 'PRI', 'DRK', 'SEC', 'SKY', 'SCI'.
        Each value is (median_image, std_image) or empty list if no data.
    """
    flag_names = {0: 'PRI', 1: 'DRK', 2: 'SEC', 3: 'SKY', 4: 'SCI'}
    fns = np.array(cycle[0])
    cycleflags = np.array(cycle[2])
    
    medians = {}
    
    for flag in (0, 1, 2, 3, 4):
        fns_flag = fns[cycleflags == flag]
        
        if len(fns_flag) < 1:
            medians[flag_names[flag]] = []
        else:
            arrs = []
            
            for f in fns_flag:
                hdu = fits.open(f)
                d0 = hdu[0].data.astype(float)
                hdu.close()
                
                if cds:
                    d = d0[1] - d0[0]
                else:
                    d = d0[1]
                    
                if bad_and_neighbors is None:
                    ovscn = get_overscan2(d)
                    d -= ovscn[None, :]
                    jail_bars = get_jailbars(d, noLightUpTo=noLightUpTo)
                    arrs.append(d - jail_bars[None, :])
                else:
                    ovscn = get_overscan2(d)
                    d -= ovscn[None, :]
                    correct_with_precomputed_neighbors(d, bad_and_neighbors)
                    jail_bars = get_jailbars(d, noLightUpTo=noLightUpTo)
                    out = d - jail_bars[None, :]
                    correct_with_precomputed_neighbors(out, bad_and_neighbors)
                    jail_bars2 = get_jailbars(out, noLightUpTo=noLightUpTo)
                    out2 = out - jail_bars2[None, :]
                    arrs.append(out)
                    
            medians[flag_names[flag]] = (
                np.median(arrs, axis=0),
                np.std(arrs, axis=0)
            )
            
    return medians


def cycle_mean_paras(cycle):
    """
    Calculate mean parallactic angle for each observation type in a cycle.
    
    Parameters
    ----------
    cycle : tuple
        Output from get_cycles.
        
    Returns
    -------
    means : dict
        Mean parallactic angle for each observation type.
    """
    flag_names = {0: 'PRI', 1: 'DRK', 2: 'SEC', 3: 'SKY', 4: 'SCI'}
    fns = np.array(cycle[0])
    cycleflags = np.array(cycle[2])
    
    means = {}
    for flag in np.unique(cycleflags):
        fns_flag = fns[cycleflags == flag]
        paras = []
        for f in fns_flag:
            hdu = fits.open(f)
            h = hdu[0].header
            hdu.close()
            paras.append(h['LBT_PARA'])
        means[flag_names[flag]] = np.mean(paras)
        
    return means


def cycle_mean_kw(cycle, kw='LBT_ALT'):
    """
    Calculate mean value of a header keyword for each observation type.
    
    Parameters
    ----------
    cycle : tuple
        Output from get_cycles.
    kw : str, optional
        FITS header keyword to average. Default: 'LBT_ALT'
        
    Returns
    -------
    means : dict
        Mean keyword value for each observation type.
    """
    print(f'cycle_mean_kw: {kw}')
    flag_names = {0: 'PRI', 1: 'DRK', 2: 'SEC', 3: 'SKY', 4: 'SCI'}
    fns = np.array(cycle[0])
    cycleflags = np.array(cycle[2])
    
    means = {}
    for flag in (0, 1, 2, 3, 4):
        print(flag)
        fns_flag = fns[cycleflags == flag]
        print(fns_flag)
        kws = []
        for f in fns_flag:
            hdu = fits.open(f)
            h = hdu[0].header
            hdu.close()
            kws.append(h[kw])
            
        try:
            means[flag_names[flag]] = np.mean(kws)
        except TypeError:
            # Handle time-format keywords
            try:
                kws1 = [datetime.strptime(k, '%H:%M:%S.%f') for k in kws]
                kws2 = np.mean([(k - kws1[0]).total_seconds() for k in kws1])
                out = kws1[0] + timedelta(seconds=kws2)
                means[flag_names[flag]] = out.strftime('%H:%M:%S.%f')
            except ValueError:
                kws1 = [datetime.strptime(k, '%H:%M:%S') for k in kws]
                kws2 = np.mean([(k - kws1[0]).total_seconds() for k in kws1])
                out = kws1[0] + timedelta(seconds=kws2)
                means[flag_names[flag]] = out.strftime('%H:%M:%S')
                
    return means


def get_jailbars(im, noLightUpTo=30):
    """
    Calculate column-wise bias pattern ("jail bars").
    
    The ALES detector exhibits a column-dependent bias pattern
    that varies by amplifier. This function measures and returns
    this pattern using rows that receive no illumination.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D detector image.
    noLightUpTo : int, optional
        Number of rows at the bottom without illumination. Default: 30
        
    Returns
    -------
    jailbars : numpy.ndarray
        1D array of column bias values.
    """
    n_amps = im.shape[1] // 64
    snip = im[:noLightUpTo, :]
    meds = np.zeros(im.shape[1])
    
    for amp in range(n_amps):
        meds[64 * amp:64 * (amp + 1)] = np.median(snip[:, 64 * amp:64 * (amp + 1)])
        
    return meds


def get_overscan2(im):
    """
    Extract horizontal overscan bias level.
    
    Uses the first few rows (overscan region) to measure
    the per-column bias level, handling edge pixels specially.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D detector image.
        
    Returns
    -------
    overscan : numpy.ndarray
        1D array of overscan values for each column.
    """
    n_amps = 2048 // 64
    snip = im[:4, :].astype(float).copy()
    snip0 = snip.copy()

    xs = np.arange(2048)
    # Columns at amplifier boundaries behave oddly
    edges = xs[::128]
    edges2 = xs[127::128]

    snip[:, edges] = np.nan
    snip[:, edges2] = np.nan

    meds = np.zeros(2048)
    for amp in range(n_amps):
        meds[64 * amp:64 * (amp + 1)] = np.nanmedian(snip[:, 64 * amp:64 * (amp + 1)])

    # Handle edge columns separately
    for edg in edges:
        meds[edg] = np.median(snip0[:, edg])
    for edg in edges2:
        meds[edg] = np.median(snip0[:, edg])

    return np.rint(meds).astype(im.dtype)


def get_vert_overscan(im, savgol_window=31):
    """
    Extract vertical overscan pattern.
    
    Uses columns at the image edges to measure row-dependent
    bias pattern.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D detector image.
    savgol_window : int, optional
        Savitzky-Golay smoothing window. Default: 31
        
    Returns
    -------
    vert_overscan : numpy.ndarray
        1D array of vertical overscan values.
    """
    snip = np.c_[im[:, :4], im[:, -4:]].astype(float)
    med_snip = np.nanmedian(snip, axis=1)
    return savgol_filter(med_snip, savgol_window, 3)


def make_cds_plus(f, outname=None):
    """
    Apply CDS+ processing and write to file.
    
    Parameters
    ----------
    f : str
        Input FITS filename.
    outname : str, optional
        Output filename. If None, writes to 'cdsplus/' directory.
    """
    hdu = fits.open(f)
    cds = hdu[0].data[1] - hdu[0].data[0]
    ovscn2 = get_overscan2(cds)
    cds_ovscn = cds - ovscn2[None, :]
    vovscn = get_vert_overscan(cds_ovscn)
    out = cds_ovscn - vovscn[:, None]
    
    hdu_out = fits.PrimaryHDU(out, header=hdu[0].header)
    if outname is None:
        os.makedirs('cdsplus', exist_ok=True)
        hdu_out.writeto(f'cdsplus/cds_{os.path.basename(f)}')
    else:
        hdu_out.writeto(outname)
        
    hdu.close()
    gc.collect()


def do_cds_plus(arr, bad_and_neighbors=None, vertical_overscan=True, jailbars_instead=False):
    """
    Apply correlated double sampling with additional corrections.
    
    This is the standard preprocessing for ALES raw frames:
    1. Correlated double sampling (subtract reset from signal)
    2. Overscan subtraction
    3. Optional vertical overscan
    4. Optional bad pixel correction
    
    Parameters
    ----------
    arr : numpy.ndarray
        Raw 3D array (2 reads, ny, nx).
    bad_and_neighbors : list, optional
        Pre-computed bad pixel corrections.
    vertical_overscan : bool, optional
        If True, subtract vertical overscan pattern. Default: True
    jailbars_instead : bool, optional
        If True, use jail bar pattern instead of overscan. Default: False
        
    Returns
    -------
    processed : numpy.ndarray
        2D processed image.
    """
    arr0 = arr.copy().astype(float)
    cds = arr0[1] - arr0[0]
    
    if jailbars_instead:
        ovscn2 = get_jailbars(cds)
    else:
        ovscn2 = get_overscan2(cds)
        
    out = cds - ovscn2[None, :]
    
    if vertical_overscan:
        vovscn = get_vert_overscan(out)
        out = out - vovscn[:, None]
        
    if bad_and_neighbors is not None:
        correct_with_precomputed_neighbors(out, bad_and_neighbors)
        
    return out


def process_no_nod_frames(fns_list, noLightUpTo=100, bad_and_neighbors=None):
    """
    Process frames without nodding (single position observations).
    
    Parameters
    ----------
    fns_list : list of str
        List of FITS filenames.
    noLightUpTo : int, optional
        Rows without illumination. Default: 100
    bad_and_neighbors : list, optional
        Pre-computed bad pixel corrections.
        
    Returns
    -------
    arrs : list of numpy.ndarray
        Processed images.
    """
    arrs = []
    for f in fns_list:
        hdu = fits.open(f)
        d = hdu[0].data.astype(float)
        hdu.close()
        
        if bad_and_neighbors is None:
            jail_bars = get_jailbars(d, noLightUpTo=noLightUpTo)
            arrs.append(d - jail_bars[None, :])
        else:
            correct_with_precomputed_neighbors(d, bad_and_neighbors)
            jail_bars = get_jailbars(d, noLightUpTo=noLightUpTo)
            out = d - jail_bars[None, :]
            correct_with_precomputed_neighbors(out, bad_and_neighbors)
            arrs.append(out)
            
    return arrs


def compute_light_leak(wave_cal_ims, output_file=None):
    """
    Compute the light leak model from narrow-band calibration images.
    
    The light leak is estimated as the median of all four narrow-band
    images. This captures the common background structure that appears
    in all filters due to light leaking around the narrow-band filters.
    
    Parameters
    ----------
    wave_cal_ims : dict
        Dictionary with keys 'nb29', 'nb33', 'nb35', 'nb39' containing
        preprocessed narrow-band images.
    output_file : str, optional
        If provided, save the light leak model to this file.
        
    Returns
    -------
    light_leak : numpy.ndarray
        2D light leak model.
        
    Examples
    --------
    >>> wave_cal_ims = organize_wavecal_frames('/data/wavecal/')
    >>> light_leak = compute_light_leak(wave_cal_ims, 'light_leak.fits')
    >>> 
    >>> # Subtract from wavecal images before building Cubifier
    >>> for nb in wave_cal_ims:
    ...     wave_cal_ims[nb] = wave_cal_ims[nb] - light_leak
    """
    # Stack all narrow-band images
    all_nbs = [
        wave_cal_ims['nb29'],
        wave_cal_ims['nb33'],
        wave_cal_ims['nb35'],
        wave_cal_ims['nb39'],
    ]
    
    # Median gives the common light leak component
    light_leak = np.median(all_nbs, axis=0)
    
    if output_file is not None:
        fits.writeto(output_file, light_leak, overwrite=True)
        print(f"Saved light leak model to {output_file}")
        
    return light_leak


def organize_wavecal_frames(directory='.', maxNdarks=100, subtract_light_leak=True,
                            light_leak_output='light_leak_median.fits'):
    """
    Organize and process wavelength calibration data.
    
    This is a high-level convenience function that:
    1. Sorts files by filter and exposure time
    2. Creates median-combined darks
    3. Creates bad pixel mask from hot pixels in darks
    4. Produces dark-subtracted, preprocessed narrow-band images
    5. Optionally computes and subtracts light leak
    
    The output is ready to be passed directly to the Cubifier.
    
    Parameters
    ----------
    directory : str, optional
        Directory containing raw calibration FITS files. Default: '.'
    maxNdarks : int, optional
        Maximum number of darks to combine. Default: 100
    subtract_light_leak : bool, optional
        If True, compute the light leak model and subtract it from
        all narrow-band images. Default: True
    light_leak_output : str, optional
        Filename for saving the light leak model. Only used if
        subtract_light_leak is True. Default: 'light_leak_median.fits'
        
    Returns
    -------
    wave_cal_ims : dict
        Dictionary with keys 'nb29', 'nb33', 'nb35', 'nb39' containing
        preprocessed narrow-band images ready for Cubifier.
        If subtract_light_leak is True, light leak has been subtracted.
    light_leak : numpy.ndarray or None
        The light leak model if subtract_light_leak is True, else None.
        This can be subtracted from the median sky as well.
        
    Notes
    -----
    This function creates several output directories and files:
    - nb29/, nb33/, nb35/, nb39/: organized raw files with symlinks
    - darks/: dark frames organized by exposure time
    - bpm_and_these_hots.fits: combined bad pixel mask
    - bad_and_neighbors_bpm_and_these_darks.pkl: pre-computed corrections
    - light_leak_median.fits: light leak model (if subtract_light_leak=True)
    
    The light leak model should also be subtracted from the median sky
    image before passing to the Cubifier, if the sky was taken at the
    same exposure time as the wavecal frames.
    
    Examples
    --------
    >>> wave_cal_ims, light_leak = organize_wavecal_frames('/data/wavecal/')
    >>> 
    >>> # Build median sky (light leak already subtracted in SkyBuilder)
    >>> builder = SkyBuilder('/data/science/')
    >>> builder.identify_frames()
    >>> median_sky = builder.build_median_sky(
    ...     dark_file='darks/median_6392.88.fits',
    ...     bad_pixel_file='bad_and_neighbors_bpm_and_these_darks.pkl',
    ...     light_leak_file='light_leak_median.fits',
    ...     output='median_sky.fits'
    ... )
    >>> 
    >>> # Build Cubifier
    >>> cubifier = Cubifier(wave_cal_ims, median_sky, np.ones_like(median_sky),
    ...                     start_offsets=(77, 193))
    """
    fns = sorted(glob.glob(os.path.join(directory, '*.fits')))
    if len(fns) == 0:
        fns = sorted(glob.glob(os.path.join(directory, '*.fits.gz')))

    # Create output directories
    for dirname in ['nb29', 'nb33', 'nb35', 'nb39', 'darks']:
        os.makedirs(dirname, exist_ok=True)

    nb29s = {}
    nb33s = {}
    nb35s = {}
    nb39s = {}
    darks = {}

    # Sort files by filter and exposure time
    for f in fns:
        print(f"Processing: {f}")
        hdu = fits.open(f)
        h = hdu[0].header
        d = hdu[0].data
        hdu.close()
        del hdu
        gc.collect()

        texp = float(h['EXPTIME'])
        
        if h['lmir_FW4'].startswith('Blank'):
            # Dark frame
            if texp in darks:
                darks[texp].append(d)
                os.symlink(
                    os.path.join('..', '..', f),
                    os.path.join('darks', f'{texp:3.2f}', os.path.basename(f))
                )
            else:
                darks[texp] = [d]
                os.makedirs(os.path.join('darks', f'{texp:3.2f}'), exist_ok=True)
                os.symlink(
                    os.path.join('..', '..', f),
                    os.path.join('darks', f'{texp:3.2f}', os.path.basename(f))
                )
        else:
            # Narrow-band filter frames
            filter_name = h['lmir_FW2']
            
            if filter_name.startswith('NB29'):
                target_dict = nb29s
                target_dir = 'nb29'
            elif filter_name.startswith('NB33'):
                target_dict = nb33s
                target_dir = 'nb33'
            elif filter_name.startswith('NB35'):
                target_dict = nb35s
                target_dir = 'nb35'
            elif filter_name.startswith('NB39'):
                target_dict = nb39s
                target_dir = 'nb39'
            else:
                continue
                
            if texp in target_dict:
                target_dict[texp].append(d)
                os.symlink(
                    os.path.join('..', '..', f),
                    os.path.join(target_dir, f'{texp:3.2f}', os.path.basename(f))
                )
            else:
                target_dict[texp] = [d]
                os.makedirs(os.path.join(target_dir, f'{texp:3.2f}'), exist_ok=True)
                os.symlink(
                    os.path.join('..', '..', f),
                    os.path.join(target_dir, f'{texp:3.2f}', os.path.basename(f))
                )

    # Create median darks
    med_dark = {}
    for txp in darks:
        med_dark[txp] = np.median(darks[txp][:maxNdarks], axis=0)
        fits.writeto(
            os.path.join('darks', f'median_{txp:3.2f}.fits'),
            med_dark[txp],
            overwrite=True
        )

    # Make bad pixel mask from hot pixels in darks
    max_txp = max(darks.keys())
    mx = np.max(darks[max_txp][-1] - darks[max_txp][0], axis=0)  # CDS
    mxm = np.zeros(mx.shape)
    mxc = mx - np.median(mx, axis=0)[None, :]  # subtract column average
    mxcr = mxc - np.median(mxc, axis=1)[:, None]  # subtract row average
    
    # Statistical threshold for hot pixels
    percent_chance = norm.ppf(1 - 0.01 * (1.0 / 2048**2))
    ca, l, u = sigmaclip(mxcr, low=percent_chance, high=percent_chance)
    mxm[mxcr > u] = 1
    
    # Combine with static bad pixel mask
    bpm = get_bpm()
    bpm_and_these_hots = np.logical_or(mxm, bpm).astype(int)
    fits.writeto('bpm_and_these_hots.fits', bpm_and_these_hots, overwrite=True)
    
    # Pre-compute bad pixel corrections
    bad_and_neighbors = find_neighbors(bpm_and_these_hots)
    with open('bad_and_neighbors_bpm_and_these_darks.pkl', 'wb') as fo:
        pickle.dump(bad_and_neighbors, fo)

    # Process narrow-band frames
    wave_cal_ims = {}
    
    for nb_name, nb_dict, nb_dir in [
        ('nb29', nb29s, 'nb29'),
        ('nb33', nb33s, 'nb33'),
        ('nb35', nb35s, 'nb35'),
        ('nb39', nb39s, 'nb39'),
    ]:
        for txp in nb_dict:
            try:
                ds_txp = np.median(nb_dict[txp], axis=0) - med_dark[txp]
                cdsp_ds_txp = do_cds_plus(ds_txp, bad_and_neighbors=bad_and_neighbors)
                
                fits.writeto(
                    os.path.join(nb_dir, f'{txp:3.2f}', f'{nb_name}_median_ds.fits'),
                    ds_txp,
                    overwrite=True
                )
                fits.writeto(
                    os.path.join(nb_dir, f'{txp:3.2f}', f'cdsp_{nb_name}_median_ds.fits'),
                    cdsp_ds_txp,
                    overwrite=True
                )
                
                # Use the processed frame for the return dictionary
                # (take the first/only exposure time available)
                if nb_name not in wave_cal_ims:
                    wave_cal_ims[nb_name] = cdsp_ds_txp
                    
            except KeyError:
                print(f'No dark with t_exp={txp} for {nb_name}')

    # Compute and subtract light leak if requested
    light_leak = None
    if subtract_light_leak:
        # Check we have all four filters
        if len(wave_cal_ims) == 4:
            print("Computing light leak model...")
            light_leak = compute_light_leak(wave_cal_ims, output_file=light_leak_output)
            
            # Subtract from all narrow-band images
            print("Subtracting light leak from narrow-band images...")
            for nb_name in wave_cal_ims:
                wave_cal_ims[nb_name] = wave_cal_ims[nb_name] - light_leak
                fits.writeto(
                    os.path.join(nb_name, f'{txp:3.2f}', f'cdsp_{nb_name}_median_ds_ll.fits'),
                    wave_cal_ims[nb_name], overwrite=True)
        else:
            print(f"Warning: Only {len(wave_cal_ims)} filters found, need 4 for light leak.")
            print("Skipping light leak subtraction.")
        # After subtracting light leak, save the corrected versions
    return wave_cal_ims, light_leak
