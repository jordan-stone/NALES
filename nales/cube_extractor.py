"""
CubeExtractor: Batch cube extraction pipeline for ALES data.

This module provides the CubeExtractor class for processing ALES science
frames into calibrated spectral cubes with running sky subtraction.
"""

import os
import pickle
from datetime import datetime

import numpy as np
from astropy.io import fits

from nales.organize import do_cds_plus


class CubeExtractor:
    """
    Batch cube extraction pipeline for ALES data.
    
    The CubeExtractor handles the complete workflow of:
    1. Loading a pre-built Cubifier
    2. For each science frame, finding temporally-close sky frames
    3. Building and subtracting a running sky background
    4. Applying preprocessing (CDS+, bad pixel correction)
    5. Extracting spectral cubes
    6. Saving with proper FITS headers and wavelength table
    
    The preprocessing and cubification steps can be run separately, which is
    useful when multiple Cubifiers are needed (e.g., for flexure tracking
    with multiple narrowband calibrations per night).
    
    Parameters
    ----------
    cubifier : Cubifier or str
        Either a Cubifier instance or path to a pickled Cubifier.
    frame_ids : SkyBuilder or str, optional
        Either a SkyBuilder instance or path to a saved SkyBuilder pickle.
        Required for automatic sky subtraction. Provides the frame 
        classifications (sky_files, sky_times, primary_files, primary_times).
    bad_pixel_file : str, optional
        Path to pickle file with pre-computed bad pixel corrections.
    raw_directory : str, optional
        Directory containing raw FITS files. If None, uses paths from
        frame_ids directly.
        
    Attributes
    ----------
    cubifier : Cubifier
        The cube extraction calibration.
    sky_files : numpy.ndarray
        Sky frame filenames.
    sky_times : numpy.ndarray
        Sky frame timestamps.
    primary_files : numpy.ndarray
        Science frame filenames.
    primary_times : numpy.ndarray
        Science frame timestamps.
    primary_parang : numpy.ndarray
        Parallactic angles of science frames.
    bad_and_neighbors : list
        Pre-computed bad pixel correction data.
        
    Examples
    --------
    Full pipeline using SkyBuilder:
    
    >>> from nales import CubeExtractor, SkyBuilder
    >>> 
    >>> # Load frame identification from SkyBuilder
    >>> frame_ids = SkyBuilder.load('target_frames.pkl')
    >>> 
    >>> # Create extractor
    >>> extractor = CubeExtractor(
    ...     cubifier='Cuber.pkl',
    ...     frame_ids=frame_ids,
    ...     bad_pixel_file='bad_and_neighbors.pkl'
    ... )
    >>> 
    >>> # Run full pipeline
    >>> extractor.run(output_dir='cubes/', n_sky_frames=50)
    
    Preprocessing only (for flexure tracking):
    
    >>> extractor.preprocess_only(
    ...     output_dir='preprocessed/',
    ...     n_sky_frames=50
    ... )
    >>> 
    >>> # Later, with a different Cubifier:
    >>> extractor2 = CubeExtractor(cubifier='Cuber_v2.pkl')
    >>> extractor2.cubify_preprocessed(
    ...     preprocessed_dir='preprocessed/',
    ...     output_dir='cubes_v2/'
    ... )
    
    Single-frame extraction:
    
    >>> cube, wavelengths = extractor.extract_single(
    ...     science_file='science_001.fits',
    ...     sky_files=['sky_001.fits', 'sky_002.fits'],
    ...     output_cube='cube_001.fits'
    ... )
    """
    
    def __init__(
        self,
        cubifier,
        frame_ids=None,
        bad_pixel_file=None,
        raw_directory=None,
    ):
        # Load cubifier if path provided
        if isinstance(cubifier, str):
            print(f"Loading Cubifier from {cubifier}")
            with open(cubifier, 'rb') as f:
                self.cubifier = pickle.load(f)
        else:
            self.cubifier = cubifier
            
        # Load frame_ids if path provided
        if frame_ids is not None:
            if isinstance(frame_ids, str):
                print(f"Loading frame IDs from {frame_ids}")
                # Import here to avoid circular imports
                from nales.sky_builder import SkyBuilder
                frame_ids = SkyBuilder.load(frame_ids)
            
            self.sky_files = frame_ids.sky_files
            self.sky_times = frame_ids.sky_times
            self.primary_files = frame_ids.primary_files
            self.primary_times = frame_ids.primary_times
            self.primary_parang = frame_ids.primary_parang
            self.raw_directory = raw_directory or frame_ids.raw_directory
        else:
            self.sky_files = None
            self.sky_times = None
            self.primary_files = None
            self.primary_times = None
            self.primary_parang = None
            self.raw_directory = raw_directory
            
        # Load bad pixel corrections if provided
        if bad_pixel_file is not None:
            print(f"Loading bad pixel file: {bad_pixel_file}")
            with open(bad_pixel_file, 'rb') as f:
                self.bad_and_neighbors = pickle.load(f)
        else:
            self.bad_and_neighbors = None
            
    def run(
        self,
        output_dir='cubes/',
        n_sky_frames=50,
        save_preprocessed=True,
        preprocessed_dir=None,
        use_reads=(0, -1),
        overwrite=False,
    ):
        """
        Run the full cube extraction pipeline on all primary frames.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory for output cube files. Default: 'cubes/'
        n_sky_frames : int, optional
            Number of closest sky frames to use for background.
            Default: 50
        save_preprocessed : bool, optional
            If True, also save the preprocessed (sky-subtracted, CDS+)
            2D images. Useful for re-cubifying with different Cubifiers.
            Default: True
        preprocessed_dir : str, optional
            Directory for preprocessed files. If None and save_preprocessed
            is True, uses output_dir. Default: None
        use_reads : tuple, optional
            Which reads to use from the ramp. Default: (0, -1)
        overwrite : bool, optional
            If True, overwrite existing files. Default: False
        """
        if self.primary_files is None:
            raise ValueError(
                "No primary files set. Provide a SkyBuilder or use "
                "extract_single() for individual frames."
            )
            
        os.makedirs(output_dir, exist_ok=True)
        if save_preprocessed:
            if preprocessed_dir is None:
                preprocessed_dir = output_dir
            os.makedirs(preprocessed_dir, exist_ok=True)
            
        print(f"Processing {len(self.primary_files)} science frames...")
        print(f"Using {n_sky_frames} closest sky frames for each")
        
        # Track last sky indices for potential reuse
        last_sky_inds = None
        last_sky_median = None
        
        for ii, primary_file in enumerate(self.primary_files):
            basename = os.path.basename(primary_file)
            cube_output = os.path.join(output_dir, f'cube_{basename}')
            
            if not overwrite and os.path.exists(cube_output):
                print(f"[{ii+1}/{len(self.primary_files)}] Skipping {basename} (exists)")
                continue
                
            print(f"[{ii+1}/{len(self.primary_files)}] Processing {basename}")
            
            # Get file path
            if self.raw_directory:
                filepath = os.path.join(self.raw_directory, basename)
            else:
                filepath = primary_file
                
            # Load science frame
            with fits.open(filepath) as hdu:
                header = hdu[0].header
                data = hdu[0].data[use_reads, :, :]
                
            # Find closest sky frames
            timestamp = self._get_timestamp(header)
            sky_inds, sky_files = self._get_closest_sky_frames(
                timestamp, n_sky_frames
            )
            
            # Build sky median (reuse if same sky frames)
            if last_sky_inds is not None and np.array_equal(sky_inds, last_sky_inds):
                print("  Re-using previous sky median")
                sky_median = last_sky_median
            else:
                print("  Computing sky median...")
                sky_median = self._build_sky_median(sky_files, use_reads)
                last_sky_inds = sky_inds
                last_sky_median = sky_median
                
            # Sky subtraction
            data_sky_sub = data - sky_median
            
            # CDS+ processing
            preprocessed = do_cds_plus(
                data_sky_sub, 
                bad_and_neighbors=self.bad_and_neighbors
            )
            
            # Save preprocessed if requested
            if save_preprocessed:
                preproc_output = os.path.join(
                    preprocessed_dir, f'bfsscdsp_{basename}'
                )
                self._save_preprocessed(
                    preprocessed, header, sky_files, preproc_output
                )
                
            # Extract cube
            cube, wavelengths = self.cubifier(preprocessed)
            
            # Save cube with full metadata
            self._save_cube(
                cube, wavelengths, header, primary_file, sky_files, cube_output
            )
            
        print("Cube extraction complete.")
        
    def preprocess_only(
        self,
        output_dir='preprocessed/',
        n_sky_frames=50,
        use_reads=(0, -1),
        overwrite=False,
    ):
        """
        Run only the preprocessing steps (no cube extraction).
        
        Useful when you need to re-cubify with different Cubifiers
        (e.g., for flexure tracking with multiple narrowband calibrations).
        
        Parameters
        ----------
        output_dir : str, optional
            Directory for preprocessed files. Default: 'preprocessed/'
        n_sky_frames : int, optional
            Number of closest sky frames for background. Default: 50
        use_reads : tuple, optional
            Which reads to use from the ramp. Default: (0, -1)
        overwrite : bool, optional
            If True, overwrite existing files. Default: False
        """
        if self.primary_files is None:
            raise ValueError(
                "No primary files set. Provide a SkyBuilder."
            )
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Preprocessing {len(self.primary_files)} science frames...")
        
        last_sky_inds = None
        last_sky_median = None
        
        for ii, primary_file in enumerate(self.primary_files):
            basename = os.path.basename(primary_file)
            output_file = os.path.join(output_dir, f'bfsscdsp_{basename}')
            
            if not overwrite and os.path.exists(output_file):
                print(f"[{ii+1}/{len(self.primary_files)}] Skipping {basename} (exists)")
                continue
                
            print(f"[{ii+1}/{len(self.primary_files)}] Preprocessing {basename}")
            
            # Get file path
            if self.raw_directory:
                filepath = os.path.join(self.raw_directory, basename)
            else:
                filepath = primary_file
                
            # Load science frame
            with fits.open(filepath) as hdu:
                header = hdu[0].header
                data = hdu[0].data[use_reads, :, :]
                
            # Find closest sky frames
            timestamp = self._get_timestamp(header)
            sky_inds, sky_files = self._get_closest_sky_frames(
                timestamp, n_sky_frames
            )
            
            # Build sky median
            if last_sky_inds is not None and np.array_equal(sky_inds, last_sky_inds):
                print("  Re-using previous sky median")
                sky_median = last_sky_median
            else:
                print("  Computing sky median...")
                sky_median = self._build_sky_median(sky_files, use_reads)
                last_sky_inds = sky_inds
                last_sky_median = sky_median
                
            # Sky subtraction and CDS+
            data_sky_sub = data - sky_median
            preprocessed = do_cds_plus(
                data_sky_sub,
                bad_and_neighbors=self.bad_and_neighbors
            )
            
            # Save
            self._save_preprocessed(preprocessed, header, sky_files, output_file)
            
        print("Preprocessing complete.")
        
    def cubify_preprocessed(
        self,
        preprocessed_dir,
        output_dir='cubes/',
        file_pattern='bfsscdsp_*.fits',
        overwrite=False,
    ):
        """
        Extract cubes from existing preprocessed files.
        
        Use this when re-cubifying with a different Cubifier (e.g., for
        flexure tracking).
        
        Parameters
        ----------
        preprocessed_dir : str
            Directory containing preprocessed files.
        output_dir : str, optional
            Directory for output cubes. Default: 'cubes/'
        file_pattern : str, optional
            Glob pattern for preprocessed files. Default: 'bfsscdsp_*.fits'
        overwrite : bool, optional
            If True, overwrite existing files. Default: False
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        preproc_files = sorted(glob.glob(
            os.path.join(preprocessed_dir, file_pattern)
        ))
        
        if len(preproc_files) == 0:
            raise ValueError(
                f"No files matching '{file_pattern}' in {preprocessed_dir}"
            )
            
        print(f"Cubifying {len(preproc_files)} preprocessed files...")
        
        for ii, preproc_file in enumerate(preproc_files):
            basename = os.path.basename(preproc_file)
            # Convert bfsscdsp_X.fits to cube_X.fits
            cube_basename = basename.replace('bfsscdsp_', 'cube_')
            cube_output = os.path.join(output_dir, cube_basename)
            
            if not overwrite and os.path.exists(cube_output):
                print(f"[{ii+1}/{len(preproc_files)}] Skipping {basename} (exists)")
                continue
                
            print(f"[{ii+1}/{len(preproc_files)}] Cubifying {basename}")
            
            # Load preprocessed data
            with fits.open(preproc_file) as hdu:
                header = hdu[0].header
                preprocessed = hdu[0].data
                
            # Extract cube
            cube, wavelengths = self.cubifier(preprocessed)
            
            # Get sky files from header comments if available
            sky_files = []
            for card in header['COMMENT']:
                if 'sky file:' in str(card):
                    sky_files.append(str(card).replace('sky file:', '').strip())
                    
            # Get original primary file from header if available
            primary_file = None
            for card in header['COMMENT']:
                if 'primary file:' in str(card):
                    primary_file = str(card).replace('primary file:', '').strip()
                    break
                    
            # Save cube
            self._save_cube(
                cube, wavelengths, header, primary_file, sky_files, cube_output
            )
            
        print("Cubification complete.")
        
    def extract_single(
        self,
        science_file,
        sky_files=None,
        n_sky_frames=50,
        output_cube=None,
        output_preprocessed=None,
        use_reads=(0, -1),
    ):
        """
        Extract a cube from a single science frame.
        
        Parameters
        ----------
        science_file : str
            Path to science FITS file.
        sky_files : list of str, optional
            List of sky file paths to use. If None and frame_ids was
            provided, finds closest sky frames automatically.
        n_sky_frames : int, optional
            Number of sky frames if using automatic selection. Default: 50
        output_cube : str, optional
            Path for output cube. If None, cube is not saved.
        output_preprocessed : str, optional
            Path for preprocessed file. If None, not saved.
        use_reads : tuple, optional
            Which reads to use. Default: (0, -1)
            
        Returns
        -------
        cube : numpy.ndarray
            3D spectral cube (n_wavelengths, n_rows, n_cols).
        wavelengths : numpy.ndarray
            1D wavelength array in microns.
        """
        # Load science frame
        with fits.open(science_file) as hdu:
            header = hdu[0].header
            data = hdu[0].data[use_reads, :, :]
            
        # Get sky files
        if sky_files is None:
            if self.sky_files is None:
                raise ValueError(
                    "No sky_files provided and no SkyBuilder available. "
                    "Either provide sky_files or initialize with a SkyBuilder."
                )
            timestamp = self._get_timestamp(header)
            _, sky_files = self._get_closest_sky_frames(timestamp, n_sky_frames)
            
        # Build sky median
        sky_median = self._build_sky_median(sky_files, use_reads)
        
        # Preprocess
        data_sky_sub = data - sky_median
        preprocessed = do_cds_plus(
            data_sky_sub,
            bad_and_neighbors=self.bad_and_neighbors
        )
        
        # Save preprocessed if requested
        if output_preprocessed is not None:
            self._save_preprocessed(
                preprocessed, header, sky_files, output_preprocessed
            )
            
        # Extract cube
        cube, wavelengths = self.cubifier(preprocessed)
        
        # Save cube if requested
        if output_cube is not None:
            self._save_cube(
                cube, wavelengths, header, science_file, sky_files, output_cube
            )
            
        return cube, wavelengths
        
    def _get_timestamp(self, header):
        """Extract timestamp from FITS header."""
        date_obs = header['DATE-OBS']
        time_obs = header['TIME-OBS']
        
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            ' %Y-%m-%dT %H:%M:%S:%f',
            ' %Y-%m-%dT %H:%M:%S.%f',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(f'{date_obs}T{time_obs}', fmt)
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse timestamp: {date_obs}T{time_obs}")
        
    def _get_closest_sky_frames(self, timestamp, n_closest):
        """Find N closest sky frames to a timestamp."""
        deltas = np.array([
            np.abs((t - timestamp).total_seconds())
            for t in self.sky_times
        ])
        indices = np.argsort(deltas)[:n_closest]
        return indices, self.sky_files[indices]
        
    def _build_sky_median(self, sky_files, use_reads):
        """Build median sky from a list of sky files."""
        # Get dimensions from first file
        sample_path = sky_files[0]
        if self.raw_directory:
            sample_path = os.path.join(
                self.raw_directory, os.path.basename(sky_files[0])
            )
            
        with fits.open(sample_path) as hdu:
            shape = hdu[0].data[use_reads, :, :].shape
            
        # Accumulate sky frames
        sky_arr = np.empty((len(sky_files),) + shape)
        
        for ii, sky_file in enumerate(sky_files):
            if self.raw_directory:
                filepath = os.path.join(
                    self.raw_directory, os.path.basename(sky_file)
                )
            else:
                filepath = sky_file
                
            with fits.open(filepath) as hdu:
                sky_arr[ii] = hdu[0].data[use_reads, :, :]
                
        return np.median(sky_arr, axis=0)
        
    def _save_preprocessed(self, data, header, sky_files, output_path):
        """Save preprocessed 2D image with metadata."""
        hdu = fits.PrimaryHDU(data)
        
        # Copy key header values
        for key in ['LBT_PARA', 'TIME-OBS', 'LBT_ALT', 'DATE-OBS', 
                    'EXPTIME', 'ITIME', 'OBJECT']:
            if key in header:
                hdu.header[key] = header[key]
                
        # Record sky files used
        for sky_file in sky_files:
            hdu.header['COMMENT'] = f'sky file: {sky_file}'
            
        hdu.writeto(output_path, overwrite=True)
        
    def _save_cube(self, cube, wavelengths, header, primary_file, sky_files, output_path):
        """
        Save cube with wavelength table extension and full metadata.
        
        The wavelength solution is stored in a binary table extension
        for precise wavelength information.
        """
        # Primary HDU with cube data
        primary_hdu = fits.PrimaryHDU(cube)
        
        # Copy key header values
        for key in ['LBT_PARA', 'TIME-OBS', 'LBT_ALT', 'DATE-OBS',
                    'EXPTIME', 'ITIME', 'OBJECT']:
            if key in header:
                primary_hdu.header[key] = header[key]
                
        # Add basic WCS-like keywords for wavelength axis
        primary_hdu.header['NAXIS3'] = len(wavelengths)
        primary_hdu.header['CRPIX3'] = 1
        primary_hdu.header['CRVAL3'] = float(wavelengths[0])
        primary_hdu.header['CDELT3'] = float(wavelengths[1] - wavelengths[0])
        primary_hdu.header['CTYPE3'] = 'WAVELENGTH'
        primary_hdu.header['CUNIT3'] = 'micron'
        
        # Also store wavelengths in header keywords (legacy format)
        for kk, wl in enumerate(wavelengths):
            primary_hdu.header[f'SLICE{kk:03d}'] = (
                float(wl), 'wavelength microns'
            )
            
        # Record provenance
        if primary_file is not None:
            primary_hdu.header['COMMENT'] = f'primary file: {primary_file}'
        for sky_file in sky_files:
            primary_hdu.header['COMMENT'] = f'sky file: {sky_file}'
            
        # Create wavelength table extension
        wavelength_col = fits.Column(
            name='WAVELENGTH',
            format='E',
            unit='micron',
            array=wavelengths
        )
        wavelength_table = fits.BinTableHDU.from_columns([wavelength_col])
        wavelength_table.header['EXTNAME'] = 'WAVELENGTH'
        
        # Write multi-extension FITS
        hdu_list = fits.HDUList([primary_hdu, wavelength_table])
        hdu_list.writeto(output_path, overwrite=True)
