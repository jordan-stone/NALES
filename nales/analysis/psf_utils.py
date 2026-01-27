"""
PSF generation utilities for ALES postprocessing.

This module provides functions for extracting instrumental PSFs from either
the central star in science frames or from auxiliary calibrator observations,
with proper handling of exposure time differences.

The PSFs generated here are used for:
1. KLIP-FM forward modeling (pyklip.fmlib)
2. Flux calibration via dn_per_contrast
3. Fake planet injection for throughput calibration

Key Features
------------
- Exposure time scaling between science and calibrator
- Multiple centroiding methods (Gaussian fit, center-of-mass, peak)
- Sub-pixel PSF centering via interpolation
- Background subtraction from annulus
- Three normalization modes with clear documentation on implications

Normalization Modes
-------------------
'none' (RECOMMENDED for flux calibration)
    Keep PSFs in raw DN units (scaled by exposure time if using aux calibrator).
    Preserves chromatic throughput and stellar spectrum shape.
    Required for proper dn_per_contrast calculation.

'per_channel'
    Normalize each wavelength channel independently to peak=1.
    DESTROYS chromatic information - only use for relative morphology.
    NOT suitable for flux calibration.

'cube'
    Normalize entire cube to max=1 (single scale factor).
    Preserves relative chromatic information.
    Can be used for flux calibration if you track the scale factor.

Example
-------
>>> from nales.analysis import ALESData
>>> 
>>> dataset = ALESData('cubes/cube_*.fits')
>>> 
>>> # Using auxiliary calibrator (exposure times read from headers automatically)
>>> dataset.generate_psfs(
...     aux_psf_files='calibrator/cal_cube.fits',
...     normalize='none'      # Keep DN for flux calibration
... )
>>>
>>> # Using central star (same exposure time, no scaling needed)
>>> dataset.generate_psfs(boxrad=12, normalize='none')
"""

import warnings
import numpy as np
from scipy.ndimage import shift, gaussian_filter
from scipy.optimize import curve_fit
from astropy.io import fits


def generate_psfs(dataset, boxrad=10, time_collapse='median',
                  aux_psf_files=None, aux_exptime=None, sci_exptime=None,
                  exptime_keyword='EXPTIME', normalize='none',
                  centroid_method='gaussian', background_subtract=True):
    """
    Generate instrumental PSF cube for KLIP-FM forward modeling.
    
    Extracts PSFs from either the central star in science frames or from
    auxiliary calibrator observations. Handles exposure time differences
    between science and calibrator data.
    
    Parameters
    ----------
    dataset : ALESData
        The dataset object to attach PSFs to.
    
    boxrad : int, optional
        Half-width of the PSF stamp in pixels. Output PSF will be 
        (2*boxrad+1) x (2*boxrad+1). Default is 10.
    
    time_collapse : str, optional
        Method to combine PSFs across time/frames:
        - 'median': Median combine (robust to outliers, default)
        - 'mean': Mean combine
        - 'weighted': Inverse-variance weighted mean
        - None: Keep all frames separate (returns 4D array)
    
    aux_psf_files : str or list of str, optional
        Path(s) to auxiliary PSF calibrator FITS files. If None, uses
        the central star from science frames. The auxiliary calibrator
        is typically an unsaturated observation of a nearby star.
    
    aux_exptime : float, optional
        Exposure time of auxiliary calibrator in seconds. If None (default),
        reads automatically from FITS header using exptime_keyword.
        Only specify to override the header value.
    
    sci_exptime : float, optional
        Exposure time of science frames in seconds. If None (default),
        reads automatically from FITS header using exptime_keyword.
        Only specify to override the header value.
    
    exptime_keyword : str, optional
        FITS header keyword for exposure time. Default is 'EXPTIME'.
    
    normalize : str, optional
        PSF normalization method:
        - 'none': Keep raw DN (RECOMMENDED for flux calibration)
        - 'per_channel': Each wavelength peak = 1 (loses chromatic info!)
        - 'cube': Entire cube max = 1 (preserves relative chromatic info)
        Default is 'none'.
    
    centroid_method : str, optional
        Method for finding PSF center:
        - 'gaussian': Fit 2D Gaussian (default, most robust for point sources)
        - 'com': Center of mass
        - 'peak': Location of peak pixel
        - 'radon': Radon transform via pyklip (best for diffraction spikes)
    
    background_subtract : bool, optional
        If True (default), subtract local background annulus from PSF.
    
    Returns
    -------
    psfs : ndarray
        PSF cube with shape (n_wavelengths, 2*boxrad+1, 2*boxrad+1).
        Units depend on normalize parameter.
    
    Notes
    -----
    **Exposure Time Scaling**
    
    When using an auxiliary calibrator with different exposure time:
    
        PSF_scaled = PSF_calibrator × (sci_exptime / aux_exptime)
    
    This ensures the PSF represents what the calibrator star would look
    like if observed with the science exposure time, which is required
    for correct flux calibration.
    
    **Normalization and Flux Calibration**
    
    For proper flux calibration to physical units, use normalize='none'.
    This preserves:
    - Chromatic instrumental throughput
    - Chromatic atmospheric transmission  
    - Stellar spectrum shape
    
    These effects divide out when computing contrast (planet/star), leaving
    only the intrinsic planet spectrum. If you normalize per-channel, you
    lose this information and cannot recover physical fluxes.
    
    **Attributes Set on Dataset**
    
    - dataset.psfs : The generated PSF cube
    - dataset.psf_flux : Total flux in each wavelength's PSF
    - dataset.psf_exptime_scale : Exposure time scaling factor applied
    """
    
    # Validate normalize parameter
    if normalize not in ['none', 'per_channel', 'cube']:
        raise ValueError(
            f"normalize must be 'none', 'per_channel', or 'cube', got '{normalize}'"
        )
    
    if normalize == 'per_channel':
        warnings.warn(
            "normalize='per_channel' destroys chromatic throughput information. "
            "Flux calibration to physical units will NOT be possible. "
            "Use normalize='none' for flux calibration workflows."
        )
    
    # Determine source of PSF (auxiliary calibrator or central star)
    if aux_psf_files is not None:
        psfs, exptime_scale = _generate_psfs_from_calibrator(
            dataset, aux_psf_files, boxrad, time_collapse,
            aux_exptime, sci_exptime, exptime_keyword,
            centroid_method, background_subtract
        )
    else:
        psfs, exptime_scale = _generate_psfs_from_central_star(
            dataset, boxrad, time_collapse,
            centroid_method, background_subtract
        )
    
    # Apply normalization
    if normalize == 'per_channel':
        for i in range(psfs.shape[0]):
            peak = np.nanmax(psfs[i])
            if peak > 0:
                psfs[i] /= peak
    elif normalize == 'cube':
        cube_max = np.nanmax(psfs)
        if cube_max > 0:
            psfs /= cube_max
    # normalize == 'none': do nothing
    
    # Store results on dataset
    dataset.psfs = psfs
    dataset.psf_flux = np.array([np.nansum(psfs[i]) for i in range(psfs.shape[0])])
    dataset.psf_exptime_scale = exptime_scale
    
    # Print summary
    n_wv = psfs.shape[0]
    stamp_size = psfs.shape[1]
    print(f"Generated PSF cube: {n_wv} wavelengths × {stamp_size}×{stamp_size} pixels")
    print(f"  Exposure time scale factor: {exptime_scale:.3f}")
    print(f"  Normalization: {normalize}")
    print(f"  Total flux range: {dataset.psf_flux.min():.3e} - {dataset.psf_flux.max():.3e}")
    
    return psfs


def _generate_psfs_from_calibrator(dataset, aux_psf_files, boxrad, time_collapse,
                                    aux_exptime, sci_exptime, exptime_keyword,
                                    centroid_method, background_subtract):
    """
    Extract PSFs from auxiliary calibrator observations.
    
    Handles exposure time normalization between calibrator and science.
    """
    import glob as glob_module
    
    # Ensure aux_psf_files is a list
    if isinstance(aux_psf_files, str):
        # Could be glob pattern
        expanded = glob_module.glob(aux_psf_files)
        if len(expanded) == 0:
            # Literal path
            aux_psf_files = [aux_psf_files]
        else:
            aux_psf_files = sorted(expanded)
    else:
        aux_psf_files = list(aux_psf_files)
    
    # Read calibrator data
    cal_cubes = []
    cal_wavelengths = []
    cal_exptimes = []
    
    for filepath in aux_psf_files:
        with fits.open(filepath) as hdulist:
            cube = hdulist[0].data.astype(np.float64)
            header = hdulist[0].header
            
            # Get wavelengths from binary table extension
            if len(hdulist) > 1:
                try:
                    wvs = hdulist[1].data['WAVELENGTH'].astype(np.float64)
                except (KeyError, TypeError):
                    try:
                        wvs = hdulist[1].data['WAVE'].astype(np.float64)
                    except (KeyError, TypeError):
                        wvs = None
            else:
                wvs = None
            
            # Fallback: use dataset wavelengths
            if wvs is None:
                if hasattr(dataset, '_unique_wvs'):
                    wvs = dataset._unique_wvs[:cube.shape[0]]
                else:
                    wvs = np.arange(cube.shape[0], dtype=np.float64)
            
            # Get exposure time
            if aux_exptime is None:
                try:
                    exptime = float(header[exptime_keyword])
                except KeyError:
                    warnings.warn(
                        f"Could not find {exptime_keyword} in calibrator header. "
                        f"No exposure time scaling will be applied."
                    )
                    exptime = None
            else:
                exptime = aux_exptime
            
            cal_cubes.append(cube)
            cal_wavelengths.append(wvs)
            cal_exptimes.append(exptime)
    
    # Get science exposure time
    if sci_exptime is None:
        if hasattr(dataset, '_exptime') and dataset._exptime is not None:
            sci_exptime = dataset._exptime
        elif hasattr(dataset, 'prihdrs') and len(dataset.prihdrs) > 0:
            try:
                sci_exptime = float(dataset.prihdrs[0][exptime_keyword])
            except KeyError:
                warnings.warn(
                    f"Could not find {exptime_keyword} in science header. "
                    f"No exposure time scaling will be applied."
                )
                sci_exptime = None
    
    # Calculate exposure time scale factor
    if sci_exptime is not None and cal_exptimes[0] is not None:
        # Average calibrator exposure time if multiple files
        valid_exptimes = [e for e in cal_exptimes if e is not None]
        if valid_exptimes:
            avg_cal_exptime = np.mean(valid_exptimes)
            exptime_scale = sci_exptime / avg_cal_exptime
            print(f"Exposure time scaling: {sci_exptime:.2f}s / {avg_cal_exptime:.2f}s = {exptime_scale:.3f}")
        else:
            exptime_scale = 1.0
    else:
        exptime_scale = 1.0
        if aux_exptime is not None or sci_exptime is not None:
            warnings.warn("Incomplete exposure time info. No scaling applied.")
    
    # Determine number of wavelengths
    n_wv = cal_cubes[0].shape[0]
    
    # Find center of PSF (use wavelength-collapsed image for robustness)
    collapsed = np.nanmedian(np.concatenate([c for c in cal_cubes], axis=0), axis=0)
    center = _find_psf_center(collapsed, method=centroid_method)
    
    print(f"Calibrator PSF center: ({center[0]:.2f}, {center[1]:.2f})")
    
    # Extract PSF stamps at each wavelength
    psf_stamps = []
    
    for wv_idx in range(n_wv):
        # Collect stamps from all frames at this wavelength
        stamps_this_wv = []
        
        for cube in cal_cubes:
            if wv_idx >= cube.shape[0]:
                continue
                
            frame = cube[wv_idx]
            
            # Refine center for this wavelength if needed
            wv_center = _find_psf_center(frame, method=centroid_method, 
                                         initial_guess=center)
            
            # Extract stamp
            stamp = _extract_stamp(frame, wv_center, boxrad, 
                                   background_subtract=background_subtract)
            stamps_this_wv.append(stamp)
        
        if len(stamps_this_wv) == 0:
            # No valid stamps, fill with zeros
            stamps_this_wv = [np.zeros((2*boxrad+1, 2*boxrad+1))]
        
        # Combine stamps across frames
        stamps_array = np.array(stamps_this_wv)
        
        if time_collapse == 'median':
            combined = np.nanmedian(stamps_array, axis=0)
        elif time_collapse == 'mean':
            combined = np.nanmean(stamps_array, axis=0)
        elif time_collapse == 'weighted':
            # Inverse variance weighting
            variances = np.nanvar(stamps_array, axis=0)
            variances = np.where(variances > 0, variances, np.inf)
            weights = 1.0 / variances
            combined = np.nansum(stamps_array * weights, axis=0) / np.nansum(weights, axis=0)
        elif time_collapse is None:
            combined = stamps_array  # Keep all frames
        else:
            raise ValueError(f"Unknown time_collapse method: {time_collapse}")
        
        psf_stamps.append(combined)
    
    psfs = np.array(psf_stamps)
    
    # Apply exposure time scaling
    psfs *= exptime_scale
    
    return psfs, exptime_scale


def _generate_psfs_from_central_star(dataset, boxrad, time_collapse,
                                      centroid_method, background_subtract):
    """
    Extract PSFs from the central star in science frames.
    
    Only appropriate when the central star is unsaturated.
    """
    
    # Get unique wavelengths
    unique_wvs = dataset._unique_wvs
    n_wv = len(unique_wvs)
    
    # Get centers
    centers = dataset.centers
    
    # Extract PSF stamps
    psf_stamps = []
    
    for wv_idx, wv in enumerate(unique_wvs):
        # Find all frames at this wavelength
        wv_mask = np.isclose(dataset.wvs, wv, rtol=1e-4)
        frame_indices = np.where(wv_mask)[0]
        
        stamps_this_wv = []
        
        for frame_idx in frame_indices:
            frame = dataset.input[frame_idx]
            center = centers[frame_idx]
            
            # Refine center if requested
            if centroid_method != 'header':
                refined_center = _find_psf_center(frame, method=centroid_method,
                                                   initial_guess=center)
            else:
                refined_center = center
            
            stamp = _extract_stamp(frame, refined_center, boxrad,
                                   background_subtract=background_subtract)
            stamps_this_wv.append(stamp)
        
        if len(stamps_this_wv) == 0:
            stamps_this_wv = [np.zeros((2*boxrad+1, 2*boxrad+1))]
        
        # Combine stamps
        stamps_array = np.array(stamps_this_wv)
        
        if time_collapse == 'median':
            combined = np.nanmedian(stamps_array, axis=0)
        elif time_collapse == 'mean':
            combined = np.nanmean(stamps_array, axis=0)
        elif time_collapse == 'weighted':
            variances = np.nanvar(stamps_array, axis=0)
            variances = np.where(variances > 0, variances, np.inf)
            weights = 1.0 / variances
            combined = np.nansum(stamps_array * weights, axis=0) / np.nansum(weights, axis=0)
        elif time_collapse is None:
            combined = stamps_array
        else:
            raise ValueError(f"Unknown time_collapse method: {time_collapse}")
        
        psf_stamps.append(combined)
    
    psfs = np.array(psf_stamps)
    
    # No exposure time scaling needed (same as science)
    exptime_scale = 1.0
    
    print("Using central star from science frames for PSF")
    
    return psfs, exptime_scale


def _find_psf_center(image, method='gaussian', initial_guess=None):
    """
    Find the center of a PSF in an image.
    
    Parameters
    ----------
    image : ndarray
        2D image containing the PSF.
    method : str
        Centroiding method:
        - 'gaussian': Fit 2D Gaussian (default, most robust for point sources)
        - 'com': Center of mass
        - 'peak': Location of peak pixel
        - 'radon': Radon transform (requires pyklip, best for diffraction spikes)
    initial_guess : tuple, optional
        Initial (x, y) guess for center.
    
    Returns
    -------
    center : ndarray
        [x, y] coordinates of PSF center.
    
    Notes
    -----
    The 'radon' method uses pyklip's radonCenter module, which implements the
    algorithm from Pueyo et al. (2015). It works by finding the center that
    maximizes the Radon transform cost function along diffraction spike angles.
    This is particularly useful for coronagraphic data with prominent spikes.
    
    See: pyklip.instruments.utils.radonCenter.searchCenter
    """
    
    # Handle NaN values
    image = np.nan_to_num(image, nan=0.0)
    
    ny, nx = image.shape
    
    if initial_guess is None:
        initial_guess = (nx / 2, ny / 2)
    
    if method == 'peak':
        # Simple peak finding
        y_peak, x_peak = np.unravel_index(np.argmax(image), image.shape)
        return np.array([float(x_peak), float(y_peak)])
    
    elif method == 'com':
        # Center of mass
        # Use region around initial guess
        search_rad = min(nx, ny) // 4
        x0, y0 = int(initial_guess[0]), int(initial_guess[1])
        
        y_lo = max(0, y0 - search_rad)
        y_hi = min(ny, y0 + search_rad)
        x_lo = max(0, x0 - search_rad)
        x_hi = min(nx, x0 + search_rad)
        
        subimage = image[y_lo:y_hi, x_lo:x_hi].copy()
        subimage -= np.percentile(subimage, 10)  # Background subtract
        subimage = np.maximum(subimage, 0)
        
        y_grid, x_grid = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        total = np.sum(subimage)
        
        if total > 0:
            x_com = np.sum(x_grid * subimage) / total
            y_com = np.sum(y_grid * subimage) / total
            return np.array([x_com, y_com])
        else:
            return np.array(initial_guess)
    
    elif method == 'radon':
        # Radon transform centroiding using pyklip
        # Best for images with diffraction spikes
        try:
            from pyklip.instruments.utils.radonCenter import searchCenter
            
            x0, y0 = initial_guess
            size_window = min(nx, ny) // 2
            
            # searchCenter returns (x, y) center
            x_ctr, y_ctr = searchCenter(
                image, 
                x_ctr_assign=x0, 
                y_ctr_assign=y0,
                size_window=size_window,
                m=0.2,           # Inner fraction to exclude (saturated core)
                M=0.8,           # Outer fraction to include
                size_cost=10,    # Search within +/- this many pixels
                theta=[45, 135], # Diffraction spike angles (diagonal)
                smooth=2,
                decimals=2
            )
            
            return np.array([x_ctr, y_ctr])
            
        except ImportError:
            warnings.warn(
                "Radon transform centroiding requires pyklip. "
                "Install with: pip install pyklip. "
                "Falling back to Gaussian fit."
            )
            method = 'gaussian'
        except Exception as e:
            warnings.warn(
                f"Radon transform centroiding failed: {e}. "
                f"Falling back to Gaussian fit."
            )
            method = 'gaussian'
    
    if method == 'gaussian':
        # Fit 2D Gaussian
        # Start from peak
        y_peak, x_peak = np.unravel_index(np.argmax(image), image.shape)
        
        # Extract subregion
        fit_rad = min(15, min(nx, ny) // 4)
        y_lo = max(0, y_peak - fit_rad)
        y_hi = min(ny, y_peak + fit_rad)
        x_lo = max(0, x_peak - fit_rad)
        x_hi = min(nx, x_peak + fit_rad)
        
        subimage = image[y_lo:y_hi, x_lo:x_hi].copy()
        
        y_sub, x_sub = np.mgrid[0:subimage.shape[0], 0:subimage.shape[1]]
        
        def gaussian_2d(coords, amp, x0, y0, sigma, offset):
            x, y = coords
            return (amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset).ravel()
        
        try:
            # Initial guesses
            amp0 = np.max(subimage) - np.median(subimage)
            x0_local = x_peak - x_lo
            y0_local = y_peak - y_lo
            sigma0 = 3.0
            offset0 = np.median(subimage)
            
            popt, _ = curve_fit(
                gaussian_2d, 
                (x_sub, y_sub), 
                subimage.ravel(),
                p0=[amp0, x0_local, y0_local, sigma0, offset0],
                bounds=([0, 0, 0, 0.5, -np.inf], 
                        [np.inf, subimage.shape[1], subimage.shape[0], 20, np.inf]),
                maxfev=1000
            )
            
            x_fit = popt[1] + x_lo
            y_fit = popt[2] + y_lo
            
            return np.array([x_fit, y_fit])
            
        except Exception:
            # Fall back to peak
            return np.array([float(x_peak), float(y_peak)])
    
    else:
        raise ValueError(f"Unknown centroid method: {method}")


def _extract_stamp(image, center, boxrad, background_subtract=True):
    """
    Extract a PSF stamp centered on the given position.
    
    Uses sub-pixel shifting to center the PSF exactly.
    
    Parameters
    ----------
    image : ndarray
        2D image to extract from.
    center : array-like
        [x, y] center position.
    boxrad : int
        Half-width of stamp.
    background_subtract : bool
        If True, subtract background estimated from outer annulus.
    
    Returns
    -------
    stamp : ndarray
        Extracted stamp of shape (2*boxrad+1, 2*boxrad+1).
    """
    
    image = np.nan_to_num(image, nan=0.0)
    ny, nx = image.shape
    cx, cy = float(center[0]), float(center[1])
    
    # Integer center
    icx, icy = int(np.round(cx)), int(np.round(cy))
    
    # Sub-pixel shift needed
    dx = icx - cx
    dy = icy - cy
    
    # Extract larger region for shifting
    pad = 5
    y_lo = max(0, icy - boxrad - pad)
    y_hi = min(ny, icy + boxrad + pad + 1)
    x_lo = max(0, icx - boxrad - pad)
    x_hi = min(nx, icx + boxrad + pad + 1)
    
    subimage = image[y_lo:y_hi, x_lo:x_hi].copy()
    
    # Apply sub-pixel shift
    if abs(dx) > 0.01 or abs(dy) > 0.01:
        subimage = shift(subimage, (dy, dx), order=3, mode='constant', cval=0)
    
    # Extract final stamp
    sub_cy = icy - y_lo
    sub_cx = icx - x_lo
    
    stamp_y_lo = sub_cy - boxrad
    stamp_y_hi = sub_cy + boxrad + 1
    stamp_x_lo = sub_cx - boxrad
    stamp_x_hi = sub_cx + boxrad + 1
    
    # Handle edge cases
    stamp_size = 2 * boxrad + 1
    if (stamp_y_lo < 0 or stamp_y_hi > subimage.shape[0] or
        stamp_x_lo < 0 or stamp_x_hi > subimage.shape[1]):
        # Pad with zeros if necessary
        stamp = np.zeros((stamp_size, stamp_size))
        
        # Calculate valid regions
        src_y_lo = max(0, stamp_y_lo)
        src_y_hi = min(subimage.shape[0], stamp_y_hi)
        src_x_lo = max(0, stamp_x_lo)
        src_x_hi = min(subimage.shape[1], stamp_x_hi)
        
        dst_y_lo = src_y_lo - stamp_y_lo
        dst_y_hi = dst_y_lo + (src_y_hi - src_y_lo)
        dst_x_lo = src_x_lo - stamp_x_lo
        dst_x_hi = dst_x_lo + (src_x_hi - src_x_lo)
        
        stamp[dst_y_lo:dst_y_hi, dst_x_lo:dst_x_hi] = \
            subimage[src_y_lo:src_y_hi, src_x_lo:src_x_hi]
    else:
        stamp = subimage[stamp_y_lo:stamp_y_hi, stamp_x_lo:stamp_x_hi].copy()
    
    # Background subtraction using outer annulus
    if background_subtract and stamp.size > 0:
        y, x = np.ogrid[:stamp.shape[0], :stamp.shape[1]]
        r = np.sqrt((x - boxrad)**2 + (y - boxrad)**2)
        
        # Annulus from 0.7*boxrad to boxrad
        annulus_mask = (r > 0.7 * boxrad) & (r <= boxrad)
        
        if np.sum(annulus_mask) > 10:
            background = np.nanmedian(stamp[annulus_mask])
            if np.isfinite(background):
                stamp -= background
    
    return stamp
