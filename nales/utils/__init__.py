"""
Utility functions for ALES data processing.

This subpackage provides lower-level algorithmic functions used by
the main nales classes. Most users won't need to import from here
directly - the commonly-used functions are re-exported at the
top level (e.g., `nales.bfixpix`, `nales.get_bpm`).

Modules
-------
bad_pixels
    Bad pixel detection and correction.
wavecal
    Wavelength calibration utilities (spot grids, fitting).
rotation
    Image rotation and trace angle measurement.
registration
    Sub-pixel image registration.
climb
    Peak climbing algorithm for spot detection.
"""

from nales.utils.bad_pixels import (
    bfixpix,
    find_neighbors,
    correct_with_precomputed_neighbors,
    get_bpm,
)

from nales.utils.wavecal import (
    get_spots_grid,
    spoof_spots,
    shift_spots_to_anchor,
    w_func,
    onedmoments,
)

from nales.utils.rotation import (
    find_rotation_angle,
    rot,
    rotate,
)

from nales.utils.registration import SubPixelRegister

from nales.utils.climb import climb, climb1d

__all__ = [
    # Bad pixels
    "bfixpix",
    "find_neighbors",
    "correct_with_precomputed_neighbors",
    "get_bpm",
    # Wavelength
    "get_spots_grid",
    "spoof_spots",
    "shift_spots_to_anchor",
    "w_func",
    "onedmoments",
    # Rotation
    "find_rotation_angle",
    "rot",
    "rotate",
    # Registration
    "SubPixelRegister",
    # Peak climbing
    "climb",
    "climb1d",
]
