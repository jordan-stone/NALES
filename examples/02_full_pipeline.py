#!/usr/bin/env python
"""
Example 02: Full ALES Reduction Pipeline
========================================

This example demonstrates a complete ALES data reduction workflow
from raw data to extracted spectral cubes.

Workflow
--------
1. Process wavelength calibration (with light leak correction)
2. Build median sky using SkyBuilder
3. Build the Cubifier
4. Diagnostic inspection
5. Batch cube extraction using CubeExtractor
6. (Optional) Re-cubify with a second Cubifier for flexure tracking

Expected Directory Structure
----------------------------
Before running:
    wavecal/raw/           - Raw narrowband and dark FITS files
    science_target/raw/    - Raw science frames including sky nods
    cubes/                 - Empty directory for output

After running:
    wavecal/darks/         - Median dark frames
    wavecal/nb*/           - Organized narrowband data with *_ll.fits files
    wavecal/light_leak_median.fits
    wavecal/bad_and_neighbors_bpm_and_these_darks.pkl
    science_target/target_frames.pkl
    science_target/median_sky.fits
    science_target/Cubifier.pkl
    cubes/cube_*.fits      - Extracted spectral cubes
    cubes/bfsscdsp_*.fits  - Preprocessed 2D frames
"""

import os
import pickle
import glob
import numpy as np
from astropy.io import fits
from datetime import datetime
import jfits

import nales


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Number of sky frames to use for running background subtraction
    n_sky_frames = 50
    
    # chunk_size for median sky (smaller = less memory, 512 good for 16GB RAM)
    chunk_size = 512
    
    # Set to True to demonstrate flexure tracking with a second Cubifier
    demo_flexure_tracking = False
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    start_time = datetime.now()
    print("=" * 60)
    print(f"ALES Reduction Pipeline")
    print(f"Started: {start_time}")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs('cubes', exist_ok=True)
    
    # =========================================================================
    # Step 1: Process Wavelength Calibration
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 1: Wavelength Calibration")
    print("=" * 60)
    
    # Change to wavecal directory
    os.chdir('wavecal')
    
    # This handles:
    # - Sorting files by filter
    # - Creating median darks
    # - Building bad pixel mask
    # - Dark-subtracting narrow-band images
    # - Computing and subtracting light leak
    # - Saving light-leak-subtracted images (*_ll.fits)
    wave_cal_ims, light_leak = nales.organize_wavecal_frames('raw/')
    
    os.chdir('..')
    
    print(f"\nProcessed {len(wave_cal_ims)} narrow-band filters")
    for key, arr in wave_cal_ims.items():
        print(f"  {key}: shape {arr.shape}, range [{arr.min():.1f}, {arr.max():.1f}]")
    print(f"\nLight leak model saved to wavecal/light_leak_median.fits")
    
    # =========================================================================
    # Interactive: Determine start_offsets from NB39 image
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("INTERACTIVE: Determine start_offsets")
    print("=" * 60)
    
    print("\nDisplaying NB39 image. Find the lower-left spot at the intersection")
    print("of the lowest complete row and leftmost complete column.")
    print("Use the cursor to identify the (x, y) pixel coordinates.\n")
    
    disp = jfits.InteractiveDisplay(wave_cal_ims['nb39'])
    disp.ax.set_title('NB39: Find lower-left spot (x, y) for start_offsets')
    
    x = int(input("Enter x coordinate of lower-left spot: "))
    y = int(input("Enter y coordinate of lower-left spot: "))
    start_offsets = (x, y)
    print(f"\nUsing start_offsets = {start_offsets}")
    
    # =========================================================================
    # Step 2: Build Median Sky
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 2: Build Median Sky")
    print("=" * 60)
    
    # Create SkyBuilder and identify frames
    # Note: Relies on FLAG header keyword from standard ALES observing scripts.
    # If your data used non-standard observing, you may need to manually
    # adjust FLAG values in the FITS headers.
    builder = nales.SkyBuilder('science_target/raw/')
    builder.identify_frames()
    builder.summary()
    
    # Save frame lists
    builder.save('science_target/target_frames.pkl')
    
    # Find median dark (filename includes exposure time in ms)
    dark_files = glob.glob('wavecal/darks/median_*.fits')
    if not dark_files:
        raise FileNotFoundError("No median dark found in wavecal/darks/")
    dark_file = dark_files[0]
    print(f"Using dark: {dark_file}")
    
    # Build median sky with all corrections
    median_sky = builder.build_median_sky(
        dark_file=dark_file,
        bad_pixel_file='wavecal/bad_and_neighbors_bpm_and_these_darks.pkl',
        light_leak_file='wavecal/light_leak_median.fits',
        light_leak_scale=1.0,  # Adjust if sky/wavecal exposure times differ
        chunk_size=chunk_size,
        output='science_target/median_sky.fits'
    )
    
    print(f"\nMedian sky shape: {median_sky.shape}")
    print(f"Median sky range: [{median_sky.min():.1f}, {median_sky.max():.1f}]")
    
    # Visualize median sky
    disp = jfits.InteractiveDisplay(median_sky)
    disp.ax.set_title('Median Sky (corrected)')
    
    # =========================================================================
    # Step 3: Build Cubifier
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 3: Build Cubifier")
    print("=" * 60)
    
    print(f"Building Cubifier with start_offsets={start_offsets}")
    
    cubifier = nales.Cubifier(
        wave_cal_ims,
        median_sky,
        sky_var=np.ones_like(median_sky),  # Uniform variance works well
        start_offsets=start_offsets,
        dims=(63, 67),
        # gridplot=True,  # Uncomment to visualize spot detection
    )
    
    # Save cubifier
    cubifier_path = 'science_target/Cubifier.pkl'
    with open(cubifier_path, 'wb') as f:
        pickle.dump(cubifier, f)
    print(f"Cubifier saved to {cubifier_path}")
    
    # =========================================================================
    # Step 4: Diagnostic Inspection
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 4: Diagnostic Inspection")
    print("=" * 60)
    
    # Check wavelength calibration at center and edge of field
    print("\nWavelength calibration at center of field (30, 30):")
    cubifier.inspect_wavecal(spaxel=(30, 30))
    
    print("\nWavelength calibration at edge of field (5, 60):")
    cubifier.inspect_wavecal(spaxel=(5, 60))
    
    # Check narrowband overlays
    print("\nNarrowband overlays at center of field:")
    cubifier.inspect_wavecal_nbs(spaxel=(30, 30))
    
    # Check extraction weights
    print("\nExtraction weights at center vs edge:")
    cubifier.inspect_weights(spaxel=(30, 30))
    cubifier.inspect_weights(spaxel=(5, 60))
    
    # =========================================================================
    # Step 5: Extract Cubes
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 5: Extract Cubes")
    print("=" * 60)
    
    # Create CubeExtractor
    extractor = nales.CubeExtractor(
        cubifier=cubifier,
        frame_ids=builder,
        bad_pixel_file='wavecal/bad_and_neighbors_bpm_and_these_darks.pkl',
        raw_directory='science_target/raw/'
    )
    
    # Run extraction pipeline
    # For each science frame this:
    # - Applies CDS and overscan correction
    # - Median-combines closest sky frames for background subtraction
    # - Corrects bad pixels
    # - Extracts spectral cube
    extractor.run(
        output_dir='cubes/',
        n_sky_frames=n_sky_frames,
    )
    
    # =========================================================================
    # Step 6: (Optional) Flexure Tracking - Re-cubify with Second Cubifier
    # =========================================================================
    
    if demo_flexure_tracking:
        print("\n" + "=" * 60)
        print("STEP 6: Flexure Tracking Demo")
        print("=" * 60)
        
        # In a real scenario, you would have a second set of narrowband
        # calibrations taken later in the night. Here we simulate by
        # building a second Cubifier with slightly different start_offsets.
        
        print("\nBuilding second Cubifier (simulating flexure shift)...")
        
        # In practice, you'd load new wave_cal_ims from a second calibration
        # For demo purposes, we just shift start_offsets slightly
        start_offsets_v2 = (start_offsets[0] + 2, start_offsets[1] + 1)
        
        cubifier_v2 = nales.Cubifier(
            wave_cal_ims,  # In reality, load from second calibration
            median_sky,
            sky_var=np.ones_like(median_sky),
            start_offsets=start_offsets_v2,
            dims=(63, 67),
        )
        
        # Save second cubifier
        with open('science_target/Cubifier_v2.pkl', 'wb') as f:
            pickle.dump(cubifier_v2, f)
        
        # The key advantage: we can re-use the preprocessed frames
        # (CDS, sky subtraction, bad pixel correction already done)
        # and just re-extract cubes with the new wavelength solution
        
        print("\nRe-cubifying preprocessed frames with second Cubifier...")
        
        extractor_v2 = nales.CubeExtractor(cubifier=cubifier_v2)
        extractor_v2.cubify_preprocessed(
            preprocessed_dir='cubes/',  # Contains bfsscdsp_*.fits files
            output_dir='cubes_v2/'
        )
        
        n_cubes_v2 = len(glob.glob('cubes_v2/cube_*.fits'))
        print(f"Re-extracted {n_cubes_v2} cubes with updated wavelength solution")
    
    # =========================================================================
    # Visualize Results
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    # Load and display a sample cube
    cube_files = sorted(glob.glob('cubes/cube_*.fits'))
    if cube_files:
        sample_file = cube_files[0]
        with fits.open(sample_file) as hdu:
            sample_cube = hdu[0].data
            if len(hdu) > 1 and hdu[1].name == 'WAVELENGTH':
                wavelengths = hdu[1].data['WAVELENGTH']
            else:
                wavelengths = np.arange(sample_cube.shape[0])
        
        # Show middle wavelength slice
        mid_idx = len(wavelengths) // 2
        disp = jfits.InteractiveDisplay(sample_cube[mid_idx])
        disp.ax.set_title(f'Sample cube at {wavelengths[mid_idx]:.2f} um: {os.path.basename(sample_file)}')
        
        # Plot a spectrum from center of field
        import matplotlib.pyplot as plt
        
        spectrum = sample_cube[:, 30, 30]
        plt.figure(figsize=(10, 4))
        plt.plot(wavelengths, spectrum, 'b-', lw=1)
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Counts')
        plt.title(f'Spectrum at spaxel (30, 30)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 60)
    print("REDUCTION COMPLETE")
    print("=" * 60)
    print(f"Elapsed time: {elapsed}")
    
    n_cubes = len(glob.glob('cubes/cube_*.fits'))
    print(f"\nExtracted {n_cubes} spectral cubes")
    
    print("\nOutput files:")
    print("  wavecal/darks/median_*.fits      - Median dark frames")
    print("  wavecal/nb*/*/cdsp_*_ll.fits     - Light-leak subtracted narrowbands")
    print("  wavecal/light_leak_median.fits   - Light leak model")
    print("  wavecal/bad_and_neighbors_*.pkl  - Bad pixel data")
    print("  science_target/target_frames.pkl - Frame classifications")
    print("  science_target/median_sky.fits   - Median sky image")
    print("  science_target/Cubifier.pkl      - Cubifier calibration")
    print("  cubes/cube_*.fits                - Spectral cubes")
    print("  cubes/bfsscdsp_*.fits            - Preprocessed 2D frames")


if __name__ == '__main__':
    main()
