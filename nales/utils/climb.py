"""
Peak climbing algorithms for spot detection.

This module provides hill-climbing algorithms used to locate
the positions of narrow-band calibration spots on the detector.
"""

import numpy as np


def climb(arr, start, verbose=False):
    """
    2D hill-climbing to find local maximum.
    
    Starting from an initial position, iteratively moves to the
    brightest neighboring pixel until a local maximum is reached.
    
    Parameters
    ----------
    arr : numpy.ndarray
        2D image array.
    start : tuple or list
        Starting (y, x) position.
    verbose : bool, optional
        If True, print progress. Default: False
        
    Returns
    -------
    peak : tuple
        (y, x) position of the local maximum.
        
    Notes
    -----
    Uses a 5x5 neighborhood for finding the next step, which helps
    avoid getting stuck on noise features.
    """
    start = [int(np.rint(s)) for s in start]
    
    if verbose:
        print('Start climb ***************************************')
        print(start)
        
    this_val = arr[start[0], start[1]]
    
    # Check 5x5 neighborhood (excluding center)
    neighbor_vals = np.r_[
        arr[start[0] - 1, start[1] - 2:start[1] + 3],
        arr[start[0] + 1, start[1] - 2:start[1] + 3],
        arr[start[0], start[1] - 2:start[1]],
        arr[start[0], start[1] + 1:start[1] + 3],
        arr[start[0] - 2, start[1] - 2:start[1] + 3],
        arr[start[0] + 2, start[1] - 2:start[1] + 3]
    ]
    
    while np.any(neighbor_vals > this_val):
        stamp = arr[start[0] - 2:start[0] + 3, start[1] - 2:start[1] + 3]
        maxp = np.where((stamp - this_val) == np.max(stamp - this_val))
        new0 = start[0] + (maxp[0][0] - 2)
        new1 = start[1] + (maxp[1][0] - 2)
        start = (new0, new1)
        
        if verbose:
            print(start)

        this_val = arr[start[0], start[1]]
        neighbor_vals = np.r_[
            arr[start[0] - 1, start[1] - 2:start[1] + 3],
            arr[start[0] + 1, start[1] - 2:start[1] + 3],
            arr[start[0], start[1] - 2:start[1]],
            arr[start[0], start[1] + 1:start[1] + 3],
            arr[start[0] - 2, start[1] - 2:start[1] + 3],
            arr[start[0] + 2, start[1] - 2:start[1] + 3]
        ]
        
    if verbose:
        print('End climb ***************************************')
        
    return start


def climb1d(y, av=None):
    """
    1D hill-climbing along a single column.
    
    Finds the local maximum in a 1D array starting from a given position.
    
    Parameters
    ----------
    y : float
        Initial y coordinate.
    av : numpy.ndarray
        1D array (image column) to search.
        
    Returns
    -------
    y_apex : int
        Y position of the local maximum.
        
    Notes
    -----
    If the climb wanders off the array, returns the starting position.
    """
    y0 = int(y)
    y = int(y)
    
    try:
        # Find direction of steepest ascent in 3-pixel neighborhood
        up = np.where(av[y - 1:y + 2] == np.max(av[y - 1:y + 2]))[0][0]
        
        # Keep climbing while not at peak and within bounds
        while up != 1 and y > 2 and y < len(av) - 2:
            y += (up - 1)
            up = np.where(av[y - 1:y + 2] == np.max(av[y - 1:y + 2]))[0][0]
            
        return y
        
    except (ValueError, IndexError):
        print('Climb wandered off... returning input')
        return y0
