"""
Bad pixel detection and correction for infrared detector data.

This module provides functions for identifying bad pixels and replacing
them with values interpolated from good neighbors.
"""

import numpy as np
import os


def get_bpm():
    """
    Load the default bad pixel mask for ALES.
    
    Returns
    -------
    bpm : numpy.ndarray
        2D bad pixel mask where 1 = bad pixel, 0 = good pixel.
    """
    from astropy.io import fits
    
    bpm_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bpm.fits')
    
    if os.path.exists(bpm_path):
        return fits.getdata(bpm_path)
    else:
        # Fall back to package data location
        import importlib.resources
        try:
            # Python 3.9+
            with importlib.resources.files('nales.data').joinpath('bpm.fits') as p:
                return fits.getdata(str(p))
        except AttributeError:
            # Older Python
            import pkg_resources
            bpm_path = pkg_resources.resource_filename('nales', 'data/bpm.fits')
            return fits.getdata(bpm_path)


def bfixpix(data, badmask, n=4, retdat=False):
    """
    Replace bad pixels with the average of nearby good pixels.
    
    For each bad pixel, finds the n nearest good pixels and replaces
    the bad value with their mean.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image data array.
    badmask : numpy.ndarray
        Boolean or integer mask with same shape as data.
        Non-zero values indicate bad pixels.
    n : int, optional
        Number of nearby good pixels to average. Default: 4
    retdat : bool, optional
        If True, return a copy of the corrected array instead of
        modifying in place. Default: False
        
    Returns
    -------
    corrected : numpy.ndarray or None
        If retdat is True, returns the corrected array.
        Otherwise returns None (data is modified in place).
        
    Notes
    -----
    This algorithm was originally from Ian Crossfield's LPL code,
    with bug fixes applied.
    
    For processing many images with the same bad pixel mask, consider
    using `find_neighbors` followed by `correct_with_precomputed_neighbors`
    for much better performance.
    
    See Also
    --------
    find_neighbors : Pre-compute neighbor information for repeated use.
    correct_with_precomputed_neighbors : Apply pre-computed corrections.
    """
    nx, ny = data.shape
    badx, bady = np.nonzero(badmask)
    nbad = len(badx)
    
    print(f'bfixpix: {nbad} bad pixels')

    if retdat:
        data = np.array(data, copy=True)

    print(f'Looping over {nbad} pixels...')
    
    for ii in range(nbad):
        thisloc = badx[ii], bady[ii]
        rad = 0
        numNearbyGoodPixels = 0

        # Expand search radius until we find enough good neighbors
        while numNearbyGoodPixels < n:
            rad += 1
            xmin = max(0, badx[ii] - rad)
            xmax = min(nx, badx[ii] + rad)
            ymin = max(0, bady[ii] - rad)
            ymax = min(ny, bady[ii] + rad)
            
            x = np.arange(nx)[xmin:xmax + 1]
            y = np.arange(ny)[ymin:ymax + 1]
            yy, xx = np.meshgrid(y, x)

            # Calculate distances, masking out bad pixels (distance = 0)
            rr = (
                abs((xx - badx[ii]) + 1j * (yy - bady[ii])) * 
                (1.0 - badmask[xmin:xmax + 1, ymin:ymax + 1])
            )
            numNearbyGoodPixels = (rr > 0).sum()

        # Find the n closest good pixels
        closestDistances = np.unique(np.sort(rr[rr > 0])[0:n])
        numDistances = len(closestDistances)
        
        localSum = 0.0
        localDenominator = 0.0
        
        for jj in range(numDistances):
            localSum += data[xmin:xmax + 1, ymin:ymax + 1][rr == closestDistances[jj]].sum()
            localDenominator += (rr == closestDistances[jj]).sum()

        data[badx[ii], bady[ii]] = localSum / localDenominator

    if retdat:
        return data
    else:
        return None


def find_neighbors(badmask, n=4):
    """
    Pre-compute neighbor information for bad pixel correction.
    
    The most time-consuming part of bad pixel correction is finding
    the good neighbors for each bad pixel. When the same mask applies
    to many images (e.g., all frames from a night), we can compute
    the neighbors once and reuse them.
    
    Parameters
    ----------
    badmask : numpy.ndarray
        Boolean or integer mask. Non-zero values indicate bad pixels.
    n : int, optional
        Number of neighbors to find for each bad pixel. Default: 4
        
    Returns
    -------
    bad_and_neighbors : list
        List of tuples, one per bad pixel:
        (bad_x, bad_y, xmin, xmax, ymin, ymax, neighbor_mask)
        
    See Also
    --------
    correct_with_precomputed_neighbors : Apply the corrections.
    
    Examples
    --------
    >>> bpm = get_bpm()
    >>> neighbors = find_neighbors(bpm)
    >>> 
    >>> # Now process many images quickly
    >>> for filename in science_files:
    ...     data = fits.getdata(filename)
    ...     correct_with_precomputed_neighbors(data, neighbors)
    """
    nx, ny = badmask.shape
    badx, bady = np.nonzero(badmask)
    nbad = len(badx)
    
    print(f'find_neighbors: {nbad} bad pixels')
    print(f'Looping over {nbad} pixels...')
    
    bad_and_neighbors = []
    
    for ii in range(nbad):
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels < n:
            rad += 1
            xmin = max(0, badx[ii] - rad)
            xmax = min(nx, badx[ii] + rad)
            ymin = max(0, bady[ii] - rad)
            ymax = min(ny, bady[ii] + rad)
            
            x = np.arange(nx)[xmin:xmax + 1]
            y = np.arange(ny)[ymin:ymax + 1]
            yy, xx = np.meshgrid(y, x)

            rr = (
                abs((xx - badx[ii]) + 1j * (yy - bady[ii])) * 
                (1.0 - badmask[xmin:xmax + 1, ymin:ymax + 1])
            )
            numNearbyGoodPixels = (rr > 0).sum()

        closestDistances = np.unique(np.sort(rr[rr > 0])[0:n])
        numDistances = len(closestDistances)
        
        pixToTake = []
        for jj in range(numDistances):
            pixToTake.append(rr == closestDistances[jj])

        bad_and_neighbors.append((
            badx[ii], 
            bady[ii], 
            xmin, 
            xmax, 
            ymin, 
            ymax, 
            np.any(pixToTake, axis=0)
        ))
        
    return bad_and_neighbors


def correct_with_precomputed_neighbors(data, bad_and_neighbors):
    """
    Apply bad pixel correction using pre-computed neighbor information.
    
    This is much faster than `bfixpix` when processing multiple images
    with the same bad pixel mask.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image to correct. Modified in place.
    bad_and_neighbors : list
        Output from `find_neighbors`.
        
    Examples
    --------
    >>> # Pre-compute once
    >>> neighbors = find_neighbors(badmask)
    >>> 
    >>> # Apply to many images
    >>> for data in images:
    ...     correct_with_precomputed_neighbors(data, neighbors)
    """
    for badx, bady, xmin, xmax, ymin, ymax, take in bad_and_neighbors:
        mean_good = data[xmin:xmax + 1, ymin:ymax + 1][take].sum() / take.sum()
        data[badx, bady] = mean_good
