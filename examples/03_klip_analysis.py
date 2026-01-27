#!/usr/bin/env python
"""
NALES Analysis Example: KLIP PSF Subtraction and Flux Calibration
==================================================================

This example demonstrates the complete workflow for high-contrast imaging
analysis of ALES datacubes using pyklip:

1. Load NALES-reduced datacubes into ALESData format
2. Generate instrumental PSFs (exposure times read from headers)
3. Set up flux calibration using stellar models
4. Run KLIP PSF subtraction
5. Extract companion spectra with forward modeling
6. Calibrate to physical flux units

Requirements
------------
- nales (with pyklip optional dependency)
- pyklip >= 2.9

Install with:
    pip install -e ".[pyklip]"

Example Data Structure
----------------------
science_data/
    cube_lm_001.fits
    cube_lm_002.fits
    ...
calibrator/
    cal_cube.fits

Author: NALES Pipeline
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Paths (modify for your data)
SCIENCE_DIR = 'science_data/'
CALIBRATOR_FILE = 'calibrator/cal_cube.fits'  # or None to use central star
OUTPUT_DIR = 'klip_output/'

# Target/calibrator info
TARGET_NAME = 'HD_12345'
W1_MAG = 5.23              # WISE W1 magnitude of PSF star
SPECTRAL_TYPE = 'G2V'      # Spectral type of PSF star

# Exposure times - usually read automatically from FITS headers
# Only set these to override header values if needed
SCI_EXPTIME = None         # None = read from header (recommended)
CAL_EXPTIME = None         # None = read from header (recommended)

# KLIP parameters
ANNULI = 9                 # Number of annuli
SUBSECTIONS = 4            # Azimuthal subsections per annulus
MOVEMENT = 1.0             # Minimum movement in pixels
KL_MODES = [1, 5, 10, 20, 50]  # KL modes to try
KLIP_MODE = 'ADI+SDI'      # 'ADI', 'SDI', or 'ADI+SDI'

# Planet search location (if known)
PLANET_SEP = 15.0          # Separation in pixels
PLANET_PA = 45.0           # Position angle in degrees

# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """Run the complete postprocessing workflow."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading NALES datacubes")
    print("=" * 60)
    
    from nales.analysis import ALESData
    
    # Find science files
    science_files = sorted(glob.glob(os.path.join(SCIENCE_DIR, 'cube_*.fits')))
    
    if len(science_files) == 0:
        print(f"No science files found in {SCIENCE_DIR}")
        print("Creating demonstration with mock data...")
        dataset = create_mock_dataset()
    else:
        # Load dataset with high-pass filtering to remove smooth background
        dataset = ALESData(
            science_files,
            highpass=True,      # Apply high-pass filter
            IWA=3,              # Inner working angle (pixels)
            OWA=None            # Outer working angle (auto from image size)
        )
    
    print(dataset)
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Generate PSFs
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Generating instrumental PSFs")
    print("=" * 60)
    
    if CALIBRATOR_FILE and os.path.exists(CALIBRATOR_FILE):
        # Use auxiliary calibrator (exposure times read from headers automatically)
        print(f"Using auxiliary calibrator: {CALIBRATOR_FILE}")
        dataset.generate_psfs(
            aux_psf_files=CALIBRATOR_FILE,
            aux_exptime=CAL_EXPTIME,   # None = read from header
            sci_exptime=SCI_EXPTIME,   # None = read from header
            normalize='none',          # IMPORTANT: Keep DN for flux calibration
            centroid_method='gaussian',
            boxrad=12
        )
    else:
        # Use central star from science frames
        print("Using central star from science frames")
        dataset.generate_psfs(
            normalize='none',
            centroid_method='gaussian',
            boxrad=12
        )
    
    # Plot PSF at central wavelength
    plot_psf(dataset, OUTPUT_DIR)
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Set Up Flux Calibration
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Setting up flux calibration")
    print("=" * 60)
    
    dataset.compute_dn_per_contrast(
        W1_mag=W1_MAG,
        spectral_type=SPECTRAL_TYPE,
        output_units='Jy'
    )
    
    # Plot stellar model and dn_per_contrast
    plot_flux_calibration(dataset, OUTPUT_DIR)
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Run KLIP
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Running KLIP PSF subtraction")
    print("=" * 60)
    
    try:
        import pyklip.parallelized as parallelized
        
        parallelized.klip_dataset(
            dataset,
            outputdir=OUTPUT_DIR,
            fileprefix=TARGET_NAME,
            annuli=ANNULI,
            subsections=SUBSECTIONS,
            movement=MOVEMENT,
            numbasis=KL_MODES,
            mode=KLIP_MODE,
            verbose=True
        )
        
        print(f"KLIP output saved to {OUTPUT_DIR}")
        
    except ImportError:
        print("pyklip not available - skipping KLIP step")
        print("Install with: pip install pyklip")
        return dataset
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Extract Companion Spectrum (Forward Modeling)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 5: Extracting companion spectrum")
    print("=" * 60)
    
    try:
        from pyklip import fmlib
        
        # This is a simplified example - real usage requires setting up
        # the forward model properly based on your specific science case
        
        print(f"Planet location: sep={PLANET_SEP:.1f} px, PA={PLANET_PA:.1f} deg")
        print("Forward modeling extraction would go here...")
        print("See pyklip.fmlib documentation for details.")
        
        # Example of what the extraction might look like:
        # contrast_spectrum = fmlib.extractSpec(...)
        
    except ImportError:
        print("pyklip.fmlib not available")
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Calibrate to Physical Flux
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 6: Flux calibration example")
    print("=" * 60)
    
    # Create mock contrast spectrum for demonstration
    n_wv = len(dataset.unique_wvs)
    mock_contrast = 1e-4 * np.ones(n_wv)  # 10^-4 contrast
    
    # Add some spectral features (mock planetary atmosphere)
    wvs = dataset.unique_wvs
    # Methane absorption at 3.3 µm
    ch4_depth = 0.3 * np.exp(-((wvs - 3.3) / 0.1)**2)
    mock_contrast *= (1 - ch4_depth)
    
    # Convert to physical flux
    planet_flux = dataset.calibrate_contrast_spectrum(mock_contrast)
    
    print(f"Mock contrast: {mock_contrast.mean():.2e}")
    print(f"Planet flux: {planet_flux.mean():.4f} Jy (mean)")
    
    # Plot result
    plot_extracted_spectrum(dataset, mock_contrast, planet_flux, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("Workflow complete!")
    print("=" * 60)
    
    return dataset


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_psf(dataset, output_dir):
    """Plot the generated PSF."""
    
    if dataset.psfs is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    n_wv = dataset.psfs.shape[0]
    indices = [0, n_wv // 2, n_wv - 1]
    
    for ax, idx in zip(axes, indices):
        wv = dataset.unique_wvs[idx]
        psf = dataset.psfs[idx]
        
        im = ax.imshow(psf, origin='lower', cmap='magma')
        ax.set_title(f'λ = {wv:.3f} µm')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='DN')
    
    plt.suptitle('Instrumental PSFs', fontsize=14)
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, 'psf_cube.png')
    plt.savefig(outpath, dpi=150)
    print(f"Saved PSF plot: {outpath}")
    plt.close()


def plot_flux_calibration(dataset, output_dir):
    """Plot flux calibration quantities."""
    
    if dataset.stellar_model_flux is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    wvs = dataset.unique_wvs
    
    # PSF flux
    ax = axes[0, 0]
    ax.semilogy(wvs, dataset.psf_flux, 'b-', lw=2)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('PSF Flux (DN)')
    ax.set_title('PSF Integrated Flux')
    ax.grid(True, alpha=0.3)
    
    # Stellar model
    ax = axes[0, 1]
    ax.semilogy(wvs, dataset.stellar_model_flux, 'r-', lw=2)
    ax.axvline(3.35, color='orange', ls='--', label='W1 band')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Stellar Flux (Jy)')
    ax.set_title(f'Stellar Model (W1 = {dataset.W1_mag:.2f} mag)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # dn_per_contrast
    ax = axes[1, 0]
    ax.semilogy(wvs, dataset.dn_per_contrast, 'g-', lw=2)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('DN per unit contrast')
    ax.set_title('dn_per_contrast')
    ax.grid(True, alpha=0.3)
    
    # Ratio check (should be flat if all consistent)
    ax = axes[1, 1]
    ratio = dataset.psf_flux / dataset.stellar_model_flux
    ax.plot(wvs, ratio / ratio.mean(), 'k-', lw=2)
    ax.axhline(1.0, color='gray', ls='--')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Normalized ratio')
    ax.set_title('PSF/Model ratio (chromatic throughput)')
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Flux Calibration', fontsize=14)
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, 'flux_calibration.png')
    plt.savefig(outpath, dpi=150)
    print(f"Saved flux calibration plot: {outpath}")
    plt.close()


def plot_extracted_spectrum(dataset, contrast, flux, output_dir):
    """Plot extracted spectrum in contrast and flux units."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    wvs = dataset.unique_wvs
    
    # Contrast spectrum
    ax = axes[0]
    ax.semilogy(wvs, contrast, 'b-', lw=2)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Planet/Star Contrast')
    ax.set_title('Contrast Spectrum')
    ax.grid(True, alpha=0.3)
    
    # Flux spectrum
    ax = axes[1]
    ax.plot(wvs, flux * 1e3, 'r-', lw=2)  # mJy
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Planet Flux (mJy)')
    ax.set_title('Calibrated Flux Spectrum')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Extracted Companion Spectrum', fontsize=14)
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, 'extracted_spectrum.png')
    plt.savefig(outpath, dpi=150)
    print(f"Saved spectrum plot: {outpath}")
    plt.close()


def create_mock_dataset():
    """Create a mock dataset for demonstration when no real data available."""
    
    from nales.analysis import ALESData
    from astropy.io import fits
    import tempfile
    
    print("Creating mock ALES datacube for demonstration...")
    
    # Create mock datacube
    n_wv = 50
    ny, nx = 63, 67
    
    # Wavelength grid (L-band)
    wavelengths = np.linspace(2.9, 4.1, n_wv)
    
    # Mock data with central PSF
    cube = np.random.randn(n_wv, ny, nx) * 10 + 100  # Noise + background
    
    # Add central star PSF
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    for i in range(n_wv):
        # PSF width increases with wavelength
        sigma = 2.0 * (wavelengths[i] / 3.5)
        psf = 10000 * np.exp(-r**2 / (2 * sigma**2))
        cube[i] += psf
    
    # Create temporary FITS file
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, 'mock_cube.fits')
    
    # Primary HDU with cube
    prihdr = fits.Header()
    prihdr['EXPTIME'] = 30.0
    prihdr['LBT_PARA'] = 45.0
    prihdr['OBJECT'] = 'MOCK_TARGET'
    primary = fits.PrimaryHDU(data=cube, header=prihdr)
    
    # Wavelength extension
    wv_col = fits.Column(name='WAVELENGTH', format='E', unit='micron',
                         array=wavelengths)
    wv_hdu = fits.BinTableHDU.from_columns([wv_col], name='WAVELENGTH')
    
    hdulist = fits.HDUList([primary, wv_hdu])
    hdulist.writeto(filepath, overwrite=True)
    
    # Load with ALESData
    dataset = ALESData(filepath, highpass=False)
    
    return dataset


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()
