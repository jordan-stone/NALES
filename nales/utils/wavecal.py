"""
Wavelength calibration utilities for ALES IFS data.

This module provides functions for locating narrow-band calibration spots
and fitting wavelength solutions across the ALES detector.

Heritage
--------
The spot-spoofing and grid alignment algorithms were developed collaboratively
between the nales and MEAD pipelines. Original nales code was extended by
Zackery Briesemeister for the MEAD pipeline (Briesemeister et al. 2018,
SPIE 10702), and those improvements were subsequently incorporated back
into nales.

References
----------
Briesemeister, Z., Skemer, A. J., Stone, J. M., et al. 2018, Proc. SPIE,
    10702, 107022Q. "MEAD: data reduction pipeline for ALES integral field
    spectrograph and LBTI thermal infrared calibration unit"
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift

from nales.utils.climb import climb
from nales.utils.registration import SubPixelRegister


def get_spots_grid(
    f1,
    start_offsets,
    smoothed=False,
    plot=True,
    stride=30,
    dims=(63, 67),
    verbose=False
):
    """
    Automatically locate all narrow-band calibration spots on the detector.
    
    Starting from an initial position near the lower-left spot, this function
    uses a peak-climbing algorithm to find each spot in a regular grid pattern.
    
    Parameters
    ----------
    f1 : numpy.ndarray
        2D narrow-band calibration image (typically NB39 for brightest spots).
    start_offsets : tuple
        Approximate position (y, x) of the lower-left spot. Required.
    smoothed : bool, optional
        If True, input is already smoothed. Default: False
    plot : bool, optional
        If True, display results interactively. Default: True
    stride : int, optional
        Approximate pixel spacing between spots. Default: 30
    dims : tuple, optional
        Expected grid dimensions (n_rows, n_cols). Default: (63, 67)
    verbose : bool, optional
        If True, print progress messages. Default: False
        
    Returns
    -------
    spots : tuple of numpy.ndarray
        Tuple of (y_positions, x_positions), each with shape `dims`.
        
    Raises
    ------
    ValueError
        If start_offsets is None.
    """
    if start_offsets is None:
        raise ValueError(
            "start_offsets is required. Inspect your NB39 calibration image "
            "to find the approximate (y, x) pixel position of the lower-left spot."
        )
    
    import matplotlib.pyplot as plt
    import jfits
    
    if verbose:
        print(f"Starting spot detection from offset {start_offsets}")
    gf1 = gaussian_filter(f1, 2)
        
    out_to_in_0 = np.empty(dims)
    out_to_in_1 = np.empty(dims)
    
    start = [start_offsets[1] - stride, start_offsets[0] - stride]
    
    for ii in range(dims[1]):
        start = (start[0], start[1] + int(stride))  # increment column
        
        for ll in range(dims[0]):
            start = (int(stride) + start[0], start[1])  # increment row
            if verbose:
                print(f"  Searching near {start}")
            
            # Climb to peak on smoothed image, then refine on original
            gpeak_pix = climb(gf1, start)
            peak_pix = climb(f1, gpeak_pix)
            peak_pix = gpeak_pix  # Use smoothed result (more stable)
            
            out_to_in_0[ll, ii] = peak_pix[0]
            out_to_in_1[ll, ii] = peak_pix[1]
                
            start = (peak_pix[0], peak_pix[1])
            
        # Return to first row for next column
        start = (out_to_in_0[0, ii] - int(stride), out_to_in_1[0, ii])
    
    # Show final plot with all detected spots
    if plot:
        disp = jfits.InteractiveDisplay(f1, vmin=-1, vmax=600)
        disp.ax.plot(out_to_in_1.flatten(), out_to_in_0.flatten(), 'ro', markersize=2)
        plt.show()
        
    return (out_to_in_0, out_to_in_1)


def spoof_spots(
    target,
    to_shift,
    direction='-+',
    best_spots_slice=(slice(920, 1020), slice(900, 1000))
):
    """
    Improve weak narrow-band spot positions using a brighter reference.
    
    When narrow-band spots are weak (especially NB29), their positions can
    be improved by measuring the shift relative to a brighter filter (NB39)
    and applying that shift.
    
    Parameters
    ----------
    target : numpy.ndarray
        Weak narrow-band image to improve.
    to_shift : numpy.ndarray
        Bright reference image (typically NB39).
    direction : str, optional
        Shift direction encoding for scipy.ndimage.shift. Default: '-+'
    best_spots_slice : tuple of slice, optional
        Region with best signal for alignment. 
        Default: (slice(920, 1020), slice(900, 1000))
        
    Returns
    -------
    shifted : numpy.ndarray
        Target image shifted to align with reference.
        
    Notes
    -----
    This function was adapted from Zack Briesemeister's implementation.
    """
    sl = best_spots_slice

    # Use unsharp masking to enhance spot edges
    unsharp = target[sl] - gaussian_filter(target[sl], 0.5)
    subreg = SubPixelRegister(unsharp)
    sh = subreg(to_shift[sl], 1000, direction=direction)
    
    return shift(to_shift, sh)


def shift_spots_to_anchor(
    target,
    wavecal_dict,
    target_filter='nb39',
    direction='-+',
    best_spots_slice=(slice(920, 1020), slice(900, 1000))
):
    """
    Shift all wavelength calibration images to align with a target.
    
    Parameters
    ----------
    target : numpy.ndarray
        Reference image to align to.
    wavecal_dict : dict
        Dictionary of narrow-band images.
    target_filter : str, optional
        Which filter in wavecal_dict to use as reference. Default: 'nb39'
    direction : str, optional
        Shift direction encoding. Default: '-+'
    best_spots_slice : tuple of slice, optional
        Region for alignment measurement.
        
    Returns
    -------
    sh_dict : dict
        Dictionary of shifted images.
        
    Notes
    -----
    This function was adapted from Zack Briesemeister's implementation.
    """
    sl = best_spots_slice

    unsharp = target[sl] - gaussian_filter(target[sl], 0.5)
    subreg = SubPixelRegister(unsharp)
    sh = subreg(wavecal_dict[target_filter][sl], 1000, direction=direction)
    
    sh_dict = {}
    for nbfilt in wavecal_dict.keys():
        sh_dict[nbfilt] = shift(wavecal_dict[nbfilt], sh)
        
    return sh_dict


def w_func(px, A, B, C):
    """
    Wavelength solution functional form.
    
    Maps pixel position to wavelength using a square-root relation,
    which approximates the prism dispersion curve.
    
    Parameters
    ----------
    px : array_like
        Pixel positions.
    A, B, C : float
        Fit coefficients.
        
    Returns
    -------
    wavelength : array_like
        Wavelengths in microns.
        
    Notes
    -----
    The functional form wavelength = A * sqrt(px - B) + C was chosen empirically
    to match ALES's prism dispersion characteristics.
    """
    return A * (px - B)**0.5 + C


def onedmoments(profile):
    """
    Calculate the first and second moments of a 1D intensity profile.
    
    Used to find the centroid of narrow-band spots in extracted spectra.
    
    Parameters
    ----------
    profile : numpy.ndarray
        1D intensity profile.
        
    Returns
    -------
    std : float
        Standard deviation (width) of the profile.
    mu : float
        Centroid position.
    A : float
        Amplitude (max - min).
    H : float
        Height (minimum value / background).
    """
    data = np.copy(profile)
    xs = np.arange(len(data))
    
    H = data.min()
    A = data.max() - data.min()
    
    # Subtract background and normalize
    data -= H
    data /= data.sum()
    
    # First moment (centroid)
    mu = xs.dot(data)
    
    # Second moment -> variance -> std
    mom2 = np.power(xs, 2).dot(data)
    var = mom2 - mu**2
    
    return np.sqrt(var), mu, A, H
