#!/usr/bin/env python
"""
Example 01: Basic Cube Extraction with nales
============================================

This example demonstrates the simplest workflow for extracting
spectral cubes from ALES data.

Prerequisites
-------------
You need the following calibration data (already preprocessed):
- Narrow-band calibration images (nb29, nb33, nb35, nb39)
- A median sky image

If you're starting from raw data, see example 02 for how to
create these from raw FITS files.
"""

import numpy as np
from astropy.io import fits
import jfits

# Import nales
import nales


def main():
    # =========================================================================
    # Step 1: Load calibration data
    # =========================================================================
    
    # Load preprocessed narrow-band calibration images
    # These are dark-subtracted, bad-pixel corrected, and light-leak subtracted
    wave_cal_ims = {
        'nb29': fits.getdata('wavecal/nb29/6392.88/cdsp_nb29_median_ds_ll.fits'),
        'nb33': fits.getdata('wavecal/nb33/6392.88/cdsp_nb33_median_ds_ll.fits'),
        'nb35': fits.getdata('wavecal/nb35/6392.88/cdsp_nb35_median_ds_ll.fits'),
        'nb39': fits.getdata('wavecal/nb39/6392.88/cdsp_nb39_median_ds_ll.fits'),
    }
    
    # Load median sky image (defines trace profiles)
    median_sky = fits.getdata('science_target/median_sky.fits')
    
    print("Calibration data loaded.")
    
    # =========================================================================
    # Step 2: Build the Cubifier
    # =========================================================================
    
    # This is the main calibration step. The Cubifier will:
    # - Locate all microspectra on the detector
    # - Measure trace rotation angles
    # - Compute optimal extraction weights
    # - Fit wavelength solutions
    #
    # This takes a few minutes but only needs to be done once per night
    # (or whenever the lenslet array mechanism is moved).
    #
    # IMPORTANT: start_offsets must be determined by inspecting the NB39 image
    # to find the (x, y) position of the lower-left spot in the grid.
    
    # First, determine start_offsets by examining the NB39 calibration image:
    # >>> disp = jfits.InteractiveDisplay(wave_cal_ims['nb39'])
    # >>> disp.ax.set_title('Find lower-left spot (x, y)')
    # Then find the approximate pixel position of the lower-left spot.
    
    print("\nBuilding Cubifier...")
    cubifier = nales.Cubifier(
        wave_cal_ims,
        median_sky,
        sky_var=np.ones_like(median_sky),  # Uniform variance works well
        start_offsets=(73, 192),  # REQUIRED: (x, y) of lower-left spot
        # Optional: show final spot detection plot
        # gridplot=True,
    )
    print("Cubifier ready!")
    
    # =========================================================================
    # Step 3: Extract cubes from science frames
    # =========================================================================
    
    # Once the Cubifier is built, extracting cubes is fast
    science_frame = fits.getdata('science_target/raw/lmircam_0500.fits')
    
    # The __call__ method returns (cube, wavelengths)
    cube, wavelengths = cubifier(science_frame)
    
    print(f"\nCube extracted!")
    print(f"  Shape: {cube.shape}")  # (n_wavelengths, 63, 67)
    print(f"  Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} microns")
    
    # =========================================================================
    # Step 4: Save and visualize results
    # =========================================================================
    
    # Save the cube with wavelength table extension
    primary_hdu = fits.PrimaryHDU(cube)
    
    # Add wavelength as binary table extension (preferred over WCS for non-linear solutions)
    wl_col = fits.Column(name='WAVELENGTH', format='E', unit='um', array=wavelengths)
    wl_table = fits.BinTableHDU.from_columns([wl_col], name='WAVELENGTH')
    
    hdul = fits.HDUList([primary_hdu, wl_table])
    hdul.writeto('output_cube.fits', overwrite=True)
    print("\nCube saved to output_cube.fits")
    
    # Quick visualization of a wavelength slice
    mid_idx = len(wavelengths) // 2
    disp = jfits.InteractiveDisplay(cube[mid_idx])
    disp.ax.set_title(f'Cube slice at {wavelengths[mid_idx]:.2f} microns')
    
    # Extract and plot a spectrum at a specific position
    import matplotlib.pyplot as plt
    
    x_pos, y_pos = 30, 30  # Center of field
    spectrum = cube[:, y_pos, x_pos]
    
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, spectrum, 'b-', lw=1)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Counts')
    plt.title(f'Spectrum at spaxel ({x_pos}, {y_pos})')
    plt.grid(True, alpha=0.3)
    plt.savefig('spectrum.png', dpi=150)
    print("Spectrum saved to spectrum.png")
    plt.show()


if __name__ == '__main__':
    main()
