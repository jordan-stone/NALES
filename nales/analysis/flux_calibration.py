"""
Flux calibration utilities for ALES postprocessing.

This module provides functions for computing dn_per_contrast conversion
factors that enable calibrating KLIP-FM extracted contrast spectra to
physical flux units (e.g., Jy, W/m²/µm).

The Workflow
------------
1. Generate PSFs in DN units (normalize='none')
2. Get stellar model spectrum for PSF reference star
3. Scale stellar model to match WISE W1 photometry
4. Compute dn_per_contrast = PSF_flux[λ] / stellar_model_flux[λ]

After KLIP-FM extraction:
    planet_flux[λ] = contrast_spectrum[λ] × stellar_model_flux[λ]

Key Equations
-------------
The PSF in DN represents:
    PSF_DN[λ] = stellar_flux[λ] × throughput[λ] × exptime × gain × ...

The stellar model (scaled to W1) gives:
    stellar_model[λ] in physical units (Jy or W/m²/µm)

So dn_per_contrast relates DN to physical units:
    dn_per_contrast[λ] = PSF_DN[λ] / stellar_model[λ]
    
And for a planet at contrast C:
    planet_DN[λ] = C[λ] × PSF_DN[λ]
    planet_flux[λ] = C[λ] × stellar_model[λ]

WISE W1 Band
------------
- Central wavelength: 3.35 µm
- Zero point: 309.54 Jy (Vega system)
- W1_flux = 309.54 × 10^(-W1_mag/2.5) Jy

Example
-------
>>> from nales.analysis import ALESData
>>> 
>>> dataset = ALESData('cubes/cube_*.fits')
>>> dataset.generate_psfs(normalize='none')
>>> dataset.compute_dn_per_contrast(W1_mag=5.23, spectral_type='G2V')
>>>
>>> # After KLIP-FM extraction:
>>> # planet_flux = contrast_spectrum * dataset.stellar_model_flux
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


# WISE W1 band parameters (Jarrett et al. 2011)
W1_CENTRAL_WAVELENGTH = 3.35  # microns
W1_ZERO_POINT_JY = 309.54     # Jy (Vega system)
W1_BANDWIDTH = 0.66           # microns (effective width)


def compute_dn_per_contrast(dataset, W1_mag, spectral_type=None,
                            Teff=None, logg=4.5, metallicity=0.0,
                            output_units='Jy', ales_resolution=35):
    """
    Compute flux conversion factors using stellar models scaled to W1.
    
    This function:
    1. Retrieves a stellar model spectrum (Pickles library or blackbody)
    2. Scales it to match the observed WISE W1 magnitude
    3. Resamples to ALES wavelength grid
    4. Computes dn_per_contrast = PSF_flux / stellar_model_flux
    
    Parameters
    ----------
    dataset : ALESData
        Dataset with PSFs already generated (dataset.psfs must exist).
        PSFs should be in DN units (normalize='none').
    
    W1_mag : float
        WISE W1 magnitude of the PSF reference star (Vega system).
        Look up on VizieR, SIMBAD, or WISE catalog.
    
    spectral_type : str, optional
        Spectral type of the star (e.g., 'G2V', 'K5III', 'A0V').
        Used to select Pickles library template.
        If None, uses Teff to select closest match or blackbody.
    
    Teff : float, optional
        Effective temperature in Kelvin. Used if spectral_type is None.
        If both are None, defaults to 5800 K (solar-type).
    
    logg : float, optional
        Surface gravity log(g). Default is 4.5 (dwarf).
        Only used for model selection, not critical for L-band.
    
    metallicity : float, optional
        Metallicity [Fe/H]. Default is 0.0 (solar).
        Only used for model selection, not critical for L-band.
    
    output_units : str, optional
        Units for stellar model flux: 'Jy' or 'W/m2/um'.
        Default is 'Jy'.
    
    ales_resolution : float, optional
        Spectral resolution (R = λ/Δλ) of ALES. Default is 35.
        Used to smooth stellar model to match ALES resolution.
    
    Returns
    -------
    dn_per_contrast : ndarray
        Conversion factors for each wavelength channel.
        Shape: (n_wavelengths,)
    
    Raises
    ------
    ValueError
        If PSFs have not been generated yet.
    
    Notes
    -----
    The stellar model is retrieved from the Pickles library if available
    via pyklip.spectra_management. If not available, a blackbody
    approximation is used.
    
    **Attributes Set on Dataset**
    
    - dataset.dn_per_contrast : The conversion factors
    - dataset.stellar_model_flux : Stellar model in output_units
    - dataset.stellar_model_wvs : Wavelengths for stellar model
    - dataset.W1_mag : Input W1 magnitude
    - dataset.W1_flux_jy : W1 flux in Jy
    
    **Using the Results**
    
    After KLIP-FM extractSpec returns contrast_spectrum:
    
        planet_flux = contrast_spectrum * dataset.stellar_model_flux
    
    The units of planet_flux will match output_units.
    
    See Also
    --------
    generate_psfs : Must be called first with normalize='none'
    """
    
    # Validate that PSFs exist
    if dataset.psfs is None:
        raise ValueError(
            "PSFs not generated yet. Call dataset.generate_psfs() first "
            "with normalize='none' for proper flux calibration."
        )
    
    if dataset.psf_flux is None or len(dataset.psf_flux) == 0:
        raise ValueError("PSF flux not computed. Regenerate PSFs.")
    
    # Get wavelength grid
    wavelengths = dataset._unique_wvs  # microns
    n_wv = len(wavelengths)
    
    # Compute W1 flux in Jy
    W1_flux_jy = W1_ZERO_POINT_JY * 10**(-W1_mag / 2.5)
    print(f"W1 magnitude: {W1_mag:.3f}")
    print(f"W1 flux: {W1_flux_jy:.4f} Jy")
    
    # Get stellar model spectrum
    stellar_wvs, stellar_flux = _get_stellar_spectrum(
        spectral_type=spectral_type,
        Teff=Teff,
        logg=logg,
        metallicity=metallicity
    )
    
    # Scale stellar model to W1 flux
    stellar_flux_scaled = _scale_to_W1(
        stellar_wvs, stellar_flux, W1_flux_jy
    )
    
    # Convert units if needed
    if output_units == 'W/m2/um':
        # Convert from Jy to W/m²/µm
        # 1 Jy = 1e-26 W/m²/Hz
        # F_λ = F_ν × c / λ²
        c = 2.998e14  # µm/s
        stellar_flux_scaled = stellar_flux_scaled * 1e-26 * c / stellar_wvs**2
    elif output_units != 'Jy':
        warnings.warn(f"Unknown output_units '{output_units}', using Jy")
        output_units = 'Jy'
    
    # Smooth to ALES resolution
    stellar_flux_smoothed = _smooth_to_resolution(
        stellar_wvs, stellar_flux_scaled, ales_resolution
    )
    
    # Resample to ALES wavelength grid
    interp_func = interp1d(
        stellar_wvs, stellar_flux_smoothed,
        kind='linear', bounds_error=False, fill_value='extrapolate'
    )
    stellar_model_flux = interp_func(wavelengths)
    
    # Ensure positive values
    stellar_model_flux = np.maximum(stellar_model_flux, 1e-30)
    
    # Compute dn_per_contrast
    # dn_per_contrast[λ] = PSF_flux[λ] / stellar_model_flux[λ]
    dn_per_contrast = dataset.psf_flux / stellar_model_flux
    
    # Store results on dataset
    dataset.dn_per_contrast = dn_per_contrast
    dataset.stellar_model_flux = stellar_model_flux
    dataset.stellar_model_wvs = wavelengths.copy()
    dataset.W1_mag = W1_mag
    dataset.W1_flux_jy = W1_flux_jy
    
    # Print summary
    print(f"Stellar model: {spectral_type or f'Teff={Teff}K'}")
    print(f"Wavelength range: {wavelengths.min():.3f} - {wavelengths.max():.3f} µm")
    print(f"Output units: {output_units}")
    print(f"dn_per_contrast range: {dn_per_contrast.min():.3e} - {dn_per_contrast.max():.3e}")
    
    return dn_per_contrast


def _get_stellar_spectrum(spectral_type=None, Teff=None, logg=4.5, metallicity=0.0):
    """
    Retrieve a stellar spectrum from the Pickles library or generate blackbody.
    
    Parameters
    ----------
    spectral_type : str, optional
        Spectral type (e.g., 'G2V', 'K5III').
    Teff : float, optional
        Effective temperature in Kelvin.
    logg : float
        Surface gravity.
    metallicity : float
        Metallicity [Fe/H].
    
    Returns
    -------
    wavelengths : ndarray
        Wavelengths in microns.
    flux : ndarray
        Flux in relative units (will be scaled to W1).
    """
    
    # Try to use pyklip's spectra_management
    try:
        from pyklip.spectra_management import get_pickles_spectrum
        
        if spectral_type is not None:
            # Clean up spectral type format
            spt = spectral_type.upper().replace(' ', '')
            
            try:
                wavelengths, flux = get_pickles_spectrum(spt)
                print(f"Using Pickles spectrum for {spt}")
                return wavelengths, flux
            except Exception as e:
                warnings.warn(
                    f"Could not get Pickles spectrum for {spectral_type}: {e}. "
                    f"Falling back to blackbody."
                )
        
    except ImportError:
        if spectral_type is not None:
            warnings.warn(
                "pyklip.spectra_management not available. "
                "Using blackbody approximation."
            )
    
    # Fall back to blackbody
    if Teff is None:
        if spectral_type is not None:
            Teff = _spectral_type_to_teff(spectral_type)
        else:
            Teff = 5800  # Solar default
    
    wavelengths, flux = _generate_blackbody(Teff)
    print(f"Using blackbody with Teff = {Teff} K")
    
    return wavelengths, flux


def _spectral_type_to_teff(spectral_type):
    """
    Convert spectral type to approximate effective temperature.
    
    Parameters
    ----------
    spectral_type : str
        Spectral type (e.g., 'G2V', 'K5III', 'A0V').
    
    Returns
    -------
    Teff : float
        Effective temperature in Kelvin.
    """
    
    # Approximate Teff values for main sequence stars
    teff_table = {
        'O': 35000, 'B': 20000, 'A': 9000, 'F': 7000,
        'G': 5500, 'K': 4500, 'M': 3500
    }
    
    # Subtype adjustments (per subtype step)
    subtype_step = {
        'O': -2000, 'B': -1500, 'A': -400, 'F': -200,
        'G': -100, 'K': -150, 'M': -150
    }
    
    spt = spectral_type.upper().replace(' ', '')
    
    if len(spt) == 0:
        return 5800
    
    spectral_class = spt[0]
    
    if spectral_class not in teff_table:
        return 5800
    
    Teff = teff_table[spectral_class]
    
    # Try to get subtype
    try:
        # Extract numeric subtype
        subtype_str = ''
        for c in spt[1:]:
            if c.isdigit() or c == '.':
                subtype_str += c
            else:
                break
        
        if subtype_str:
            subtype = float(subtype_str)
            Teff += subtype_step[spectral_class] * subtype
    except (ValueError, IndexError):
        pass
    
    return max(2500, min(50000, Teff))


def _generate_blackbody(Teff, wv_min=0.3, wv_max=30.0, n_points=10000):
    """
    Generate a Planck blackbody spectrum.
    
    Parameters
    ----------
    Teff : float
        Effective temperature in Kelvin.
    wv_min, wv_max : float
        Wavelength range in microns.
    n_points : int
        Number of wavelength points.
    
    Returns
    -------
    wavelengths : ndarray
        Wavelengths in microns.
    flux : ndarray
        Flux in arbitrary units (B_λ).
    """
    
    # Physical constants
    h = 6.626e-34   # Planck constant (J·s)
    c = 2.998e8     # Speed of light (m/s)
    k = 1.381e-23   # Boltzmann constant (J/K)
    
    # Wavelength grid in meters
    wavelengths_um = np.linspace(wv_min, wv_max, n_points)
    wavelengths_m = wavelengths_um * 1e-6
    
    # Planck function B_λ
    # B_λ = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
    with np.errstate(over='ignore', divide='ignore'):
        x = h * c / (wavelengths_m * k * Teff)
        # Avoid overflow for large x
        x = np.minimum(x, 700)
        B_lambda = (2 * h * c**2 / wavelengths_m**5) / (np.exp(x) - 1)
    
    # Handle any infinities
    B_lambda = np.nan_to_num(B_lambda, nan=0.0, posinf=0.0, neginf=0.0)
    
    return wavelengths_um, B_lambda


def _scale_to_W1(wavelengths, flux, W1_flux_jy):
    """
    Scale stellar spectrum to match WISE W1 flux.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelengths in microns.
    flux : ndarray
        Flux in arbitrary units.
    W1_flux_jy : float
        Target W1 flux in Jy.
    
    Returns
    -------
    flux_scaled : ndarray
        Flux scaled to Jy, matching W1 at 3.35 µm.
    """
    
    # Define W1 bandpass (approximate as top-hat)
    w1_lo = W1_CENTRAL_WAVELENGTH - W1_BANDWIDTH / 2
    w1_hi = W1_CENTRAL_WAVELENGTH + W1_BANDWIDTH / 2
    
    # Find wavelengths in W1 band
    in_band = (wavelengths >= w1_lo) & (wavelengths <= w1_hi)
    
    if np.sum(in_band) < 2:
        # If model doesn't cover W1, extrapolate from nearest point
        idx = np.argmin(np.abs(wavelengths - W1_CENTRAL_WAVELENGTH))
        model_W1_flux = flux[idx]
    else:
        # Compute mean flux in W1 band
        model_W1_flux = np.trapz(flux[in_band], wavelengths[in_band]) / W1_BANDWIDTH
    
    if model_W1_flux <= 0:
        warnings.warn("Model flux at W1 is zero or negative. Using unity scaling.")
        return flux
    
    # Scale factor to match W1
    scale = W1_flux_jy / model_W1_flux
    
    return flux * scale


def _smooth_to_resolution(wavelengths, flux, R):
    """
    Smooth spectrum to match instrumental resolution.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelengths in microns.
    flux : ndarray
        Flux values.
    R : float
        Spectral resolution (λ/Δλ).
    
    Returns
    -------
    flux_smoothed : ndarray
        Smoothed flux.
    """
    
    if R is None or R <= 0:
        return flux
    
    # At resolution R, FWHM = λ/R
    # For Gaussian, σ = FWHM / 2.355
    # In wavelength space, σ varies with λ
    
    # Use average wavelength for approximate constant kernel
    avg_wv = np.mean(wavelengths)
    delta_wv = avg_wv / R / 2.355
    
    # Convert to pixels
    wv_step = np.median(np.diff(wavelengths))
    sigma_pixels = delta_wv / wv_step
    
    if sigma_pixels < 0.5:
        return flux  # Already lower resolution
    
    flux_smoothed = gaussian_filter1d(flux, sigma=sigma_pixels, mode='nearest')
    
    return flux_smoothed


def contrast_to_flux(contrast_spectrum, stellar_model_flux):
    """
    Convert contrast spectrum to physical flux.
    
    This is a convenience function for the final step after KLIP-FM
    extraction.
    
    Parameters
    ----------
    contrast_spectrum : ndarray
        Extracted planet/star contrast ratio at each wavelength.
        This is the output of pyklip.fmlib.extractSpec.
    stellar_model_flux : ndarray
        Stellar model flux from dataset.stellar_model_flux.
        Units are whatever was specified in compute_dn_per_contrast.
    
    Returns
    -------
    planet_flux : ndarray
        Planet flux in same units as stellar_model_flux.
    
    Example
    -------
    >>> # After KLIP-FM extraction
    >>> from nales.analysis.flux_calibration import contrast_to_flux
    >>> 
    >>> planet_flux = contrast_to_flux(
    ...     contrast_spectrum, 
    ...     dataset.stellar_model_flux
    ... )
    >>> # planet_flux is now in Jy (or W/m²/µm)
    """
    return contrast_spectrum * stellar_model_flux


def mag_to_flux_jy(magnitude, zero_point_jy=W1_ZERO_POINT_JY):
    """
    Convert Vega magnitude to flux in Jy.
    
    Parameters
    ----------
    magnitude : float
        Vega magnitude.
    zero_point_jy : float
        Zero point flux in Jy. Default is W1 zero point (309.54 Jy).
    
    Returns
    -------
    flux_jy : float
        Flux in Jy.
    """
    return zero_point_jy * 10**(-magnitude / 2.5)


def flux_jy_to_mag(flux_jy, zero_point_jy=W1_ZERO_POINT_JY):
    """
    Convert flux in Jy to Vega magnitude.
    
    Parameters
    ----------
    flux_jy : float
        Flux in Jy.
    zero_point_jy : float
        Zero point flux in Jy. Default is W1 zero point (309.54 Jy).
    
    Returns
    -------
    magnitude : float
        Vega magnitude.
    """
    return -2.5 * np.log10(flux_jy / zero_point_jy)
