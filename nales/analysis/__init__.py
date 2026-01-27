"""
nales.analysis: High-Contrast Imaging Analysis for ALES
========================================================

This module provides tools for high-contrast imaging analysis of ALES
datacubes, including integration with pyKLIP for PSF subtraction and
spectral extraction.

Requirements
------------
The pyKLIP integration requires pyklip to be installed:

    pip install pyklip

Or install nales with pyklip support:

    pip install -e ".[pyklip]"

Main Classes
------------
ALESData
    pyKLIP-compatible dataset class for ALES IFS datacubes.
    Inherits from pyklip.instruments.Instrument.Data.

Main Functions
--------------
generate_psfs
    Generate instrumental PSF cube from central star or auxiliary calibrator.
compute_dn_per_contrast
    Compute flux conversion factors using stellar models and W1 photometry.
contrast_to_flux
    Convert extracted contrast spectrum to physical flux units.

Example
-------
>>> from nales.analysis import ALESData
>>> import pyklip.parallelized as parallelized
>>> import glob
>>>
>>> # Load NALES-reduced datacubes
>>> dataset = ALESData(glob.glob('cubes/cube_*.fits'), highpass=True)
>>>
>>> # Generate PSFs (exposure times read from headers automatically)
>>> dataset.generate_psfs(
...     aux_psf_files='calibrator/cal_cube.fits',
...     normalize='none'
... )
>>>
>>> # Set up flux calibration
>>> dataset.compute_dn_per_contrast(W1_mag=5.23, spectral_type='G2V')
>>>
>>> # Run KLIP
>>> parallelized.klip_dataset(
...     dataset,
...     outputdir='klip_output/',
...     fileprefix='target',
...     annuli=9,
...     subsections=4,
...     movement=1,
...     numbasis=[1, 5, 10, 20, 50],
...     mode='ADI+SDI'
... )

Notes
-----
FITS Structure Expected (NALES output format):
    - HDU 0: Primary header + 3D datacube (n_wavelengths, y, x)
    - HDU 1: Binary table with 'WAVELENGTH' column (microns)

Required FITS Keywords:
    - EXPTIME: Exposure time in seconds
    - LBT_PARA: Parallactic angle in degrees

See Also
--------
pyklip : https://pyklip.readthedocs.io/
"""

__all__ = [
    'ALESData',
    'generate_psfs',
    'compute_dn_per_contrast',
    'contrast_to_flux',
    'is_pyklip_available',
]

# Attempt to import pyklip-dependent modules
# If pyklip is not installed, provide helpful error messages

_PYKLIP_INSTALL_MSG = (
    "pyklip is required for nales.analysis PSF subtraction features.\n"
    "Install with: pip install pyklip\n"
    "Or install nales with pyklip support: pip install -e \".[pyklip]\""
)

try:
    # Test that pyklip is available
    import pyklip.instruments.Instrument
    _PYKLIP_AVAILABLE = True
    
except ImportError as _import_error:
    _PYKLIP_AVAILABLE = False
    _pyklip_import_error = _import_error


def _check_pyklip():
    """Raise ImportError with helpful message if pyklip is not available."""
    if not _PYKLIP_AVAILABLE:
        raise ImportError(_PYKLIP_INSTALL_MSG) from _pyklip_import_error


# Define lazy-loading classes and functions that check for pyklip on use
class ALESData:
    """
    pyKLIP-compatible dataset class for ALES IFS datacubes.
    
    This is a lazy-loading wrapper. The actual implementation is in
    nales.analysis.pyklip_interface.ALESData.
    
    See the module docstring or ALESData.__init__ for full documentation.
    """
    
    _real_class = None
    
    def __new__(cls, *args, **kwargs):
        _check_pyklip()
        
        # Import the real class on first use
        if cls._real_class is None:
            from nales.analysis.pyklip_interface import ALESData as _RealALESData
            cls._real_class = _RealALESData
        
        # Create and return an instance of the real class
        return cls._real_class(*args, **kwargs)
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support for type hints like ALESData[...]"""
        _check_pyklip()
        if cls._real_class is None:
            from nales.analysis.pyklip_interface import ALESData as _RealALESData
            cls._real_class = _RealALESData
        return cls._real_class


def generate_psfs(*args, **kwargs):
    """
    Generate instrumental PSF cube from central star or auxiliary calibrator.
    
    This is a convenience function. For full documentation, see
    ALESData.generate_psfs() method.
    
    Note: This function requires an ALESData instance as the first argument.
    It's typically easier to call dataset.generate_psfs(...) directly.
    """
    _check_pyklip()
    from nales.analysis.psf_utils import generate_psfs as _generate_psfs
    return _generate_psfs(*args, **kwargs)


def compute_dn_per_contrast(*args, **kwargs):
    """
    Compute flux conversion factors using stellar models and W1 photometry.
    
    This is a convenience function. For full documentation, see
    ALESData.compute_dn_per_contrast() method.
    
    Note: This function requires an ALESData instance as the first argument.
    It's typically easier to call dataset.compute_dn_per_contrast(...) directly.
    """
    _check_pyklip()
    from nales.analysis.flux_calibration import compute_dn_per_contrast as _compute
    return _compute(*args, **kwargs)


def contrast_to_flux(contrast_spectrum, stellar_model_flux):
    """
    Convert contrast spectrum to physical flux.
    
    This is a convenience function for the final step after KLIP-FM
    extraction. Does not require pyklip.
    
    Parameters
    ----------
    contrast_spectrum : ndarray
        Extracted planet/star contrast ratio at each wavelength.
    stellar_model_flux : ndarray
        Stellar model flux from dataset.stellar_model_flux.
    
    Returns
    -------
    planet_flux : ndarray
        Planet flux in same units as stellar_model_flux.
    
    Example
    -------
    >>> from nales.analysis import contrast_to_flux
    >>> planet_flux = contrast_to_flux(contrast_spectrum, dataset.stellar_model_flux)
    """
    from nales.analysis.flux_calibration import contrast_to_flux as _ctf
    return _ctf(contrast_spectrum, stellar_model_flux)


def is_pyklip_available():
    """
    Check if pyklip is available for import.
    
    Returns
    -------
    bool
        True if pyklip is installed and can be imported, False otherwise.
    
    Example
    -------
    >>> from nales.analysis import is_pyklip_available
    >>> if is_pyklip_available():
    ...     from nales.analysis import ALESData
    ... else:
    ...     print("pyklip not installed, skipping KLIP analysis")
    """
    return _PYKLIP_AVAILABLE
