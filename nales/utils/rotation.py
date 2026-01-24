"""
Image rotation and trace angle measurement for ALES data.

This module provides functions for rotating images and measuring
the angle of microspectral traces on the detector.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from nales.utils.climb import climb1d


def find_rotation_angle(im, ref_pos=None, plot=False, slope=-3, leftof=20, rightof=20, debug=False):
    """
    Measure the rotation angle of a microspectral trace.
    
    Traces the position of the trace ridge across columns using a
    peak-climbing algorithm, then fits a line to find the angle.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D image stamp containing a single trace (masked sky image).
    ref_pos : tuple, optional
        Starting (x, y) position near the trace center.
    plot : bool, optional
        If True, display diagnostic plot. Default: False
    slope : float, optional
        Expected slope for guiding the trace following. Default: -3
    leftof : int, optional
        Number of columns to trace leftward. Default: 20
    rightof : int, optional  
        Number of columns to trace rightward. Default: 20
    debug : bool, optional
        If True, pause for debugging. Default: False
        
    Returns
    -------
    angle : float
        Rotation angle in degrees (relative to horizontal).
    """
    import matplotlib.pyplot as plt
    import jfits
    
    x1 = []
    y1 = []
    
    # Start at reference position
    x1.append(ref_pos[0])
    y1.append(climb1d(ref_pos[1], im[:, x1[-1]]))
    
    # Trace forward (rightward) from reference
    for xx in range(ref_pos[0] + 1, ref_pos[0] + rightof):
        x1.append(xx)
        y1.append(climb1d(y1[-1] + slope, im[:, xx]))

    x2 = []
    y2 = []
    x2.append(ref_pos[0])
    y2.append(y1[0])
    
    # Trace backward (leftward) from reference
    for xx in range(ref_pos[0] - leftof, ref_pos[0])[::-1]:
        x2.append(xx)
        y2.append(climb1d(y2[-1] - slope, im[:, xx]))

    # Combine traces
    xs = x2[::-1] + x1
    ys = y2[::-1] + y1

    # Fit line and convert to angle
    pcs = np.polyfit(xs, ys, 1)
    angle = 90 - (180 / np.pi) * np.arctan(-1 * pcs[0])
    
    if plot or debug:
        disp = jfits.InteractiveDisplay(im)
        disp.ax.plot(xs, ys, 'ro')
        disp.ax.plot(xs, np.polyval(pcs, xs), 'w-')
        if debug:
            plt.show()
            raise SystemExit("Debug stop")

    return angle


def rot(im, angle, axis, order=3, pivot=False, out_size=None):
    """
    Rotate an image clockwise about a specified axis point.
    
    Uses scipy's map_coordinates for sub-pixel interpolation during
    the rotation.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D image to rotate.
    angle : float
        Rotation angle in degrees (clockwise positive).
    axis : tuple
        (x, y) coordinates of the rotation axis.
    order : int, optional
        Interpolation order (0-5). Default: 3 (cubic)
    pivot : bool, optional
        If True, pivot about the axis point. If False, center the
        axis in the output image. Default: False
    out_size : tuple, optional
        Output image dimensions (ny, nx). If None, uses 2x max dimension.
        
    Returns
    -------
    rotated : numpy.ndarray
        Rotated image. Pixels outside the original image are set to NaN.
    """
    # Convert to radians
    angle_rad = angle * np.pi / 180.0
    
    # Set up output coordinate grid
    if out_size is None:
        y, x = np.indices((2 * max(im.shape), 2 * max(im.shape)))
        y -= int(max(im.shape) / 2)
        x -= int(max(im.shape) / 2)
    else:
        y, x = np.indices(out_size)
        y -= int((out_size[0] - im.shape[0]) / 2)
        x -= int((out_size[1] - im.shape[1]) / 2)
    
    # Calculate axis offset when pivoting from bottom left corner
    theta_axis = np.arctan2(axis[1], axis[0])
    r_axis = np.abs(axis[0] + 1j * axis[1])
    yoffset = r_axis * np.sin(theta_axis) - r_axis * np.sin(theta_axis - angle_rad)
    xoffset = r_axis * np.cos(theta_axis) - r_axis * np.cos(theta_axis - angle_rad)

    # Center the axis if not pivoting
    ycenter_offset = (1 - int(pivot)) * ((im.shape[0] / 2.0) - axis[1])
    xcenter_offset = (1 - int(pivot)) * ((im.shape[1] / 2.0) - axis[0])
    yoffset += ycenter_offset
    xoffset += xcenter_offset

    # Rotation matrix elements
    ct = np.cos(angle_rad)
    st = np.sin(angle_rad)

    # Apply inverse rotation to find source coordinates
    new_x = ct * (x - xoffset) - st * (y - yoffset)
    new_y = st * (x - xoffset) + ct * (y - yoffset)
    
    return map_coordinates(im, [new_y, new_x], order=order, cval=np.nan)


def rotate(im, angle, order=3):
    """
    Simple rotation about image center.
    
    Convenience wrapper for rot() with default centering.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D image to rotate.
    angle : float
        Rotation angle in degrees.
    order : int, optional
        Interpolation order. Default: 3
        
    Returns
    -------
    rotated : numpy.ndarray
        Rotated image.
    """
    center = (im.shape[1] / 2.0, im.shape[0] / 2.0)
    return rot(im, angle, center, order=order)


def rot_and_interp(im, angle, axis, ys, order=3):
    """
    Experimental: Rotate and interpolate at specific y positions.
    
    Parameters
    ----------
    im : numpy.ndarray
        2D image.
    angle : float
        Rotation angle in degrees.
    axis : tuple
        Rotation axis point.
    ys : numpy.ndarray
        Y positions to sample.
    order : int, optional
        Interpolation order.
        
    Returns
    -------
    result : numpy.ndarray
        Interpolated values.
    """
    angle_rad = angle * np.pi / 180.0
    
    y = np.empty((len(ys), 20)) + ys[:, None]
    x = np.empty((len(ys), 20)) + np.arange(20)[None, :]
    x -= 10
    
    theta_axis = np.arctan2(axis[1], axis[0])
    r_axis = np.abs(axis[0] + 1j * axis[1])
    yoffset = r_axis * np.sin(theta_axis) - r_axis * np.sin(theta_axis - angle_rad)
    xoffset = r_axis * np.cos(theta_axis) - r_axis * np.cos(theta_axis - angle_rad)

    ct = np.cos(angle_rad)
    st = np.sin(angle_rad)

    new_x = ct * (x - xoffset) - st * (y - yoffset)
    new_y = st * (x - xoffset) + ct * (y - yoffset)
    
    return map_coordinates(im, [new_y, new_x], order=order, cval=np.nan)
