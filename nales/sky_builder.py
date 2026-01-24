"""
SkyBuilder: Build median sky frames from nod sequences.

This module provides the SkyBuilder class for identifying sky frames,
building memory-efficient median sky images, and managing frame lists
for reproducible data reduction.
"""

import os
import gc
import pickle
from datetime import datetime

import numpy as np
from astropy.io import fits

from nales.organize import get_flags, do_cds_plus


class SkyBuilder:
    """
    Build median sky frames from ALES nod sequences.
    
    The SkyBuilder handles the complete workflow of:
    1. Identifying sky vs. science frames from FITS headers
    2. Building a median sky image using memory-efficient chunked processing
    3. Applying dark subtraction, bad pixel correction, and light leak removal
    4. Saving/loading frame lists for reproducibility
    
    The two-step workflow (identify frames, then build sky) allows for
    interactive inspection of frame classification before committing to
    the computationally expensive median combination.
    
    Parameters
    ----------
    data_directory : str
        Path to directory containing raw FITS files.
    raw_directory : str, optional
        Path to raw data files if different from data_directory.
        If None, uses data_directory.
        
    Attributes
    ----------
    data_directory : str
        Directory containing observation data.
    raw_directory : str
        Directory containing raw FITS files.
    all_files : numpy.ndarray
        All FITS filenames.
    all_times : numpy.ndarray
        Timestamps for all files.
    all_flags : numpy.ndarray
        Observation type flags for all files.
    all_parang : numpy.ndarray
        Parallactic angles for all files.
    sky_files : numpy.ndarray
        Filenames of identified sky frames.
    sky_times : numpy.ndarray
        Timestamps of sky frames.
    primary_files : numpy.ndarray
        Filenames of primary (science) frames.
    primary_times : numpy.ndarray
        Timestamps of primary frames.
    primary_parang : numpy.ndarray
        Parallactic angles of primary frames.
        
    Examples
    --------
    Basic workflow:
    
    >>> from nales import SkyBuilder
    >>> 
    >>> # Step 1: Identify frames
    >>> builder = SkyBuilder('/path/to/target_data/')
    >>> builder.identify_frames()
    >>> print(f"Found {len(builder.sky_files)} sky frames")
    >>> print(f"Found {len(builder.primary_files)} science frames")
    >>> 
    >>> # Save frame lists for later use
    >>> builder.save('target_frames.pkl')
    >>> 
    >>> # Step 2: Build median sky (can be run later)
    >>> builder = SkyBuilder.load('target_frames.pkl')
    >>> median_sky = builder.build_median_sky(
    ...     dark_file='median_dark.fits',
    ...     bad_pixel_file='bad_and_neighbors.pkl',
    ...     output='median_sky.fits'
    ... )
    
    With light leak correction:
    
    >>> median_sky = builder.build_median_sky(
    ...     dark_file='median_dark.fits',
    ...     bad_pixel_file='bad_and_neighbors.pkl',
    ...     light_leak_file='light_leak_median.fits',
    ...     output='median_sky.fits'
    ... )
    """
    
    def __init__(self, data_directory=None, raw_directory=None):
        self.data_directory = data_directory
        self.raw_directory = raw_directory if raw_directory else data_directory
        
        # Frame identification results
        self.all_files = None
        self.all_times = None
        self.all_flags = None
        self.all_parang = None
        
        # Separated frame lists
        self.sky_files = None
        self.sky_times = None
        self.primary_files = None
        self.primary_times = None
        self.primary_parang = None
        
    def identify_frames(self, sky_flag=3, primary_flag=0):
        """
        Identify sky and science frames from FITS headers.
        
        Reads all FITS files in the data directory and classifies them
        based on the FLAG header keyword.
        
        Parameters
        ----------
        sky_flag : int, optional
            Flag value indicating sky frames. Default: 3 (SKY/NOD_B)
        primary_flag : int, optional
            Flag value indicating primary/science frames. Default: 0 (PRI/NOD_A)
            
        Notes
        -----
        Flag values used by ALES:
        - 0: Primary (PRI) or NOD_A - on-source
        - 1: Dark (DRK) or blank
        - 2: Secondary (SEC)
        - 3: Sky (SKY) or NOD_B - off-source
        - 4: Science (SCI)
        """
        if self.data_directory is None:
            raise ValueError(
                "data_directory not set. Either pass it to __init__ or use load()."
            )
            
        print(f"Scanning directory: {self.data_directory}")
        fns, times, flags, paras = get_flags(self.data_directory)
        
        self.all_files = np.array(fns)
        self.all_times = np.array(times)
        self.all_flags = np.array(flags)
        self.all_parang = np.array(paras)
        
        # Separate sky and primary frames
        sky_mask = self.all_flags == sky_flag
        primary_mask = self.all_flags == primary_flag
        
        self.sky_files = self.all_files[sky_mask]
        self.sky_times = self.all_times[sky_mask]
        
        self.primary_files = self.all_files[primary_mask]
        self.primary_times = self.all_times[primary_mask]
        self.primary_parang = self.all_parang[primary_mask]
        
        print(f"Found {len(self.sky_files)} sky frames (flag={sky_flag})")
        print(f"Found {len(self.primary_files)} primary frames (flag={primary_flag})")
        
    def save(self, filename):
        """
        Save frame identification results to a pickle file.
        
        Parameters
        ----------
        filename : str
            Output pickle filename.
        """
        data = {
            'data_directory': self.data_directory,
            'raw_directory': self.raw_directory,
            'all_files': self.all_files,
            'all_times': self.all_times,
            'all_flags': self.all_flags,
            'all_parang': self.all_parang,
            'sky_files': self.sky_files,
            'sky_times': self.sky_times,
            'primary_files': self.primary_files,
            'primary_times': self.primary_times,
            'primary_parang': self.primary_parang,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved frame lists to {filename}")
        
    @classmethod
    def load(cls, filename):
        """
        Load a SkyBuilder from a saved pickle file.
        
        Parameters
        ----------
        filename : str
            Path to pickle file created by save().
            
        Returns
        -------
        builder : SkyBuilder
            Restored SkyBuilder instance.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        builder = cls(
            data_directory=data['data_directory'],
            raw_directory=data['raw_directory']
        )
        builder.all_files = data['all_files']
        builder.all_times = data['all_times']
        builder.all_flags = data['all_flags']
        builder.all_parang = data['all_parang']
        builder.sky_files = data['sky_files']
        builder.sky_times = data['sky_times']
        builder.primary_files = data['primary_files']
        builder.primary_times = data['primary_times']
        builder.primary_parang = data['primary_parang']
        
        print(f"Loaded frame lists from {filename}")
        print(f"  {len(builder.sky_files)} sky frames")
        print(f"  {len(builder.primary_files)} primary frames")
        
        return builder
        
    def build_median_sky(
        self,
        dark_file=None,
        bad_pixel_file=None,
        light_leak_file=None,
        light_leak_scale=1.0,
        output='median_sky.fits',
        chunk_size=512,
        use_reads=(0, -1),
    ):
        """
        Build a median sky image from identified sky frames.
        
        Uses memory-efficient chunked processing to handle large datasets
        that don't fit in memory.
        
        Parameters
        ----------
        dark_file : str, optional
            Path to median dark FITS file for dark subtraction.
        bad_pixel_file : str, optional
            Path to pickle file with pre-computed bad pixel corrections.
        light_leak_file : str, optional
            Path to light leak model FITS file for subtraction.
        light_leak_scale : float, optional
            Scale factor for light leak subtraction. Use if the light leak
            was measured at a different exposure time. Default: 1.0
        output : str, optional
            Output filename for median sky. Default: 'median_sky.fits'
        chunk_size : int, optional
            Size of image chunks for memory-efficient processing.
            The image is processed in chunk_size x chunk_size tiles.
            Default: 512
        use_reads : tuple, optional
            Which reads to use from the ramp, as indices.
            Default: (0, -1) for first and last reads (CDS).
            
        Returns
        -------
        median_sky : numpy.ndarray
            2D median sky image after CDS+ processing.
            
        Notes
        -----
        The processing steps are:
        1. For each chunk, load the relevant portion of all sky frames
        2. Compute the median along the frame axis
        3. Reassemble the full image from chunks
        4. Subtract median dark (if provided)
        5. Apply CDS+ processing (correlated double sampling + bad pixel fix)
        6. Subtract light leak (if provided)
        """
        if self.sky_files is None or len(self.sky_files) == 0:
            raise ValueError(
                "No sky frames identified. Run identify_frames() first."
            )
            
        print(f"Building median sky from {len(self.sky_files)} frames...")
        
        # Load dark if provided
        med_dark = None
        if dark_file is not None:
            print(f"Loading dark: {dark_file}")
            med_dark = fits.getdata(dark_file)
            
        # Load bad pixel corrections if provided
        bad_and_neighbors = None
        if bad_pixel_file is not None:
            print(f"Loading bad pixel file: {bad_pixel_file}")
            with open(bad_pixel_file, 'rb') as f:
                bad_and_neighbors = pickle.load(f)
                
        # Load light leak if provided
        light_leak = None
        if light_leak_file is not None:
            print(f"Loading light leak: {light_leak_file}")
            light_leak = fits.getdata(light_leak_file) * light_leak_scale
            
        # Determine image dimensions from first file
        with fits.open(self.sky_files[0]) as hdu:
            data_shape = hdu[0].data.shape
            
        # Handle different data shapes
        if len(data_shape) == 3:
            n_reads, ny, nx = data_shape
            n_use_reads = len(use_reads)
        else:
            ny, nx = data_shape
            n_use_reads = 1
            use_reads = None
            
        print(f"Image size: {ny} x {nx}, using {n_use_reads} reads")
        
        # Calculate number of chunks
        n_chunks_y = int(np.ceil(ny / chunk_size))
        n_chunks_x = int(np.ceil(nx / chunk_size))
        
        # Initialize output array
        if use_reads is not None:
            whole_arr = np.empty((n_use_reads, ny, nx))
        else:
            whole_arr = np.empty((ny, nx))
            
        # Process in chunks for memory efficiency
        print(f"Processing in {n_chunks_y}x{n_chunks_x} chunks of {chunk_size}x{chunk_size}...")
        
        for ky in range(n_chunks_y):
            for kx in range(n_chunks_x):
                y_start = ky * chunk_size
                y_end = min((ky + 1) * chunk_size, ny)
                x_start = kx * chunk_size
                x_end = min((kx + 1) * chunk_size, nx)
                
                chunk_ny = y_end - y_start
                chunk_nx = x_end - x_start
                
                print(f"  Chunk ({ky}, {kx}): [{y_start}:{y_end}, {x_start}:{x_end}]")
                
                # Allocate array for this chunk across all frames
                if use_reads is not None:
                    chunk_arr = np.empty((len(self.sky_files), n_use_reads, chunk_ny, chunk_nx))
                else:
                    chunk_arr = np.empty((len(self.sky_files), chunk_ny, chunk_nx))
                    
                # Load chunk from each sky frame
                for ii, sky_file in enumerate(self.sky_files):
                    # Handle path: use basename if raw_directory differs
                    if self.raw_directory != self.data_directory:
                        basename = os.path.basename(sky_file)
                        filepath = os.path.join(self.raw_directory, basename)
                    else:
                        filepath = sky_file
                        
                    with fits.open(filepath) as hdu:
                        if use_reads is not None:
                            chunk_arr[ii] = hdu[0].data[use_reads, y_start:y_end, x_start:x_end]
                        else:
                            chunk_arr[ii] = hdu[0].data[y_start:y_end, x_start:x_end]
                            
                # Compute median for this chunk
                med_chunk = np.median(chunk_arr, axis=0)
                
                if use_reads is not None:
                    whole_arr[:, y_start:y_end, x_start:x_end] = med_chunk
                else:
                    whole_arr[y_start:y_end, x_start:x_end] = med_chunk
                    
                # Free memory
                del chunk_arr
                gc.collect()
                
        # Apply dark subtraction
        if med_dark is not None:
            print("Subtracting dark...")
            if use_reads is not None:
                whole_arr = whole_arr - med_dark[use_reads, :, :]
            else:
                whole_arr = whole_arr - med_dark
                
        # Apply CDS+ processing
        print("Applying CDS+ processing...")
        out = do_cds_plus(whole_arr, bad_and_neighbors=bad_and_neighbors)
        
        # Apply light leak correction
        if light_leak is not None:
            print("Subtracting light leak...")
            out = out - light_leak
            
        # Save result
        print(f"Saving to {output}...")
        fits.writeto(output, out, overwrite=True)
        print(f"Median sky complete: {output}")
        
        return out
        
    def get_closest_sky_frames(self, timestamp, n_closest=20):
        """
        Find the N sky frames closest in time to a given timestamp.
        
        Parameters
        ----------
        timestamp : datetime or str
            Target timestamp. If string, will attempt to parse common formats.
        n_closest : int, optional
            Number of closest frames to return. Default: 20
            
        Returns
        -------
        indices : numpy.ndarray
            Indices into sky_files of the closest frames.
        files : numpy.ndarray
            Filenames of the closest sky frames.
        """
        if self.sky_files is None or len(self.sky_files) == 0:
            raise ValueError(
                "No sky frames identified. Run identify_frames() first."
            )
            
        # Parse timestamp if string
        if isinstance(timestamp, str):
            timestamp = self._parse_timestamp(timestamp)
            
        # Calculate time differences
        deltas = np.array([
            np.abs((t - timestamp).total_seconds()) 
            for t in self.sky_times
        ])
        
        # Get indices of closest frames
        indices = np.argsort(deltas)[:n_closest]
        
        return indices, self.sky_files[indices]
        
    def _parse_timestamp(self, tstr):
        """Parse a timestamp string in various formats."""
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            ' %Y-%m-%dT %H:%M:%S:%f',
            ' %Y-%m-%dT %H:%M:%S.%f',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(tstr, fmt)
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse timestamp: {tstr}")
        
    def get_exposure_time(self):
        """
        Get the exposure time from the first sky frame.
        
        Returns
        -------
        exptime : float
            Exposure time in seconds (EXPTIME header value).
            
        Notes
        -----
        This is useful for checking if light leak calibration was taken
        at the same exposure time as the sky frames.
        """
        if self.sky_files is None or len(self.sky_files) == 0:
            raise ValueError("No sky frames identified")
            
        with fits.open(self.sky_files[0]) as hdu:
            return float(hdu[0].header.get('EXPTIME', hdu[0].header.get('ITIME', 0)))
            
    def summary(self):
        """Print a summary of identified frames."""
        print("\n" + "=" * 50)
        print("SkyBuilder Summary")
        print("=" * 50)
        print(f"Data directory: {self.data_directory}")
        print(f"Raw directory:  {self.raw_directory}")
        
        if self.all_files is not None:
            print(f"\nTotal files scanned: {len(self.all_files)}")
            
            # Count by flag
            if self.all_flags is not None:
                flag_names = {0: 'Primary', 1: 'Dark', 2: 'Secondary', 3: 'Sky', 4: 'Science'}
                unique, counts = np.unique(self.all_flags, return_counts=True)
                print("\nFrame breakdown by type:")
                for flag, count in zip(unique, counts):
                    name = flag_names.get(flag, f'Unknown({flag})')
                    print(f"  {name}: {count}")
                    
        if self.sky_files is not None:
            print(f"\nSky frames selected: {len(self.sky_files)}")
            if len(self.sky_times) > 0:
                print(f"  Time range: {self.sky_times[0]} to {self.sky_times[-1]}")
                
        if self.primary_files is not None:
            print(f"Primary frames selected: {len(self.primary_files)}")
            if len(self.primary_times) > 0:
                print(f"  Time range: {self.primary_times[0]} to {self.primary_times[-1]}")
                
        print("=" * 50 + "\n")
