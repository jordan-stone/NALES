"""
nales: LBTI/ALES Integral Field Spectrograph Data Reduction Pipeline
=====================================================================

nales transforms raw ALES detector images into calibrated 3D spectral cubes.

Main Classes
------------
Cubifier
    The main class for building and applying cube extraction calibrations.
SkyBuilder
    Helper class for building median sky frames from nod sequences.
CubeExtractor
    Pipeline class for batch cube extraction with sky subtraction.

Main Functions
--------------
organize_wavecal_frames
    Organize and preprocess wavelength calibration data (with light leak).
compute_light_leak
    Compute light leak model from narrow-band images.

Example
-------
>>> import nales
>>> import numpy as np
>>>
>>> # Step 1: Process wavelength calibration (light leak auto-subtracted)
>>> wave_cal_ims, light_leak = nales.organize_wavecal_frames('/path/to/wavecal/')
>>>
>>> # Step 2: Identify frames and build median sky
>>> builder = nales.SkyBuilder('/path/to/science_data/')
>>> builder.identify_frames()
>>> builder.save('target_frames.pkl')
>>> median_sky = builder.build_median_sky(
...     dark_file='darks/median_6392.88.fits',
...     bad_pixel_file='bad_and_neighbors_bpm_and_these_darks.pkl',
...     light_leak_file='light_leak_median.fits',  # subtract from sky too
...     output='median_sky.fits'
... )
>>>
>>> # Step 3: Build cubifier (start_offsets from inspecting NB39 image)
>>> cubifier = nales.Cubifier(
...     wave_cal_ims, median_sky, sky_var=np.ones_like(median_sky),
...     start_offsets=(77, 193)
... )
>>>
>>> # Step 4: Extract cubes with CubeExtractor
>>> extractor = nales.CubeExtractor(
...     cubifier=cubifier,
...     sky_builder=builder,
...     bad_pixel_file='bad_and_neighbors_bpm_and_these_darks.pkl'
... )
>>> extractor.run(output_dir='cubes/', n_sky_frames=50)
"""

__version__ = "1.0.0"
__author__ = "Jordan Stone"
# NOTE: Update email before release
__email__ = "your.email@example.edu"

# Main classes
from nales.cubifier import Cubifier
from nales.sky_builder import SkyBuilder
from nales.cube_extractor import CubeExtractor

# High-level workflow functions
from nales.organize import (
    organize_wavecal_frames,
    compute_light_leak,
    get_flags,
    organize_blocks,
    get_cycles,
    make_cycle_medians,
    do_cds_plus,
)

# Utility functions (commonly used)
from nales.utils.bad_pixels import (
    bfixpix,
    find_neighbors,
    correct_with_precomputed_neighbors,
    get_bpm,
)

# Analysis tools
from nales.analysis.pca import PCA, parallel_annular_PCA

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Main classes
    "Cubifier",
    "SkyBuilder",
    "CubeExtractor",
    # Workflow functions
    "organize_wavecal_frames",
    "compute_light_leak",
    "get_flags",
    "organize_blocks", 
    "get_cycles",
    "make_cycle_medians",
    "do_cds_plus",
    # Utilities
    "bfixpix",
    "find_neighbors",
    "correct_with_precomputed_neighbors",
    "get_bpm",
    # Analysis
    "PCA",
    "parallel_annular_PCA",
]
