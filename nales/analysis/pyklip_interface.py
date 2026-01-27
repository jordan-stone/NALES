"""
pyklip instrument interface for ALES IFS datacubes.

This module provides the ALESData class, which inherits from
pyklip.instruments.Instrument.Data and implements the interface
required for pyKLIP PSF subtraction.

FITS Structure Expected (NALES output format):
    - HDU 0: Primary header + 3D datacube (n_wavelengths, y, x)
    - HDU 1: Binary table extension 'WAVELENGTH' with column 'WAVELENGTH' (microns)

Required FITS Keywords:
    - EXPTIME: Exposure time in seconds
    - LBT_PARA: Parallactic angle in degrees

Optional FITS Keywords:
    - OBJECT: Target name
    - DATE-OBS: Observation date

Example
-------
>>> from nales.analysis import ALESData
>>> import pyklip.parallelized as parallelized
>>> import glob
>>>
>>> # Load datacubes
>>> dataset = ALESData(glob.glob('cubes/cube_*.fits'), highpass=True)
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
"""

import os
import glob as glob_module
import warnings
import numpy as np
from astropy.io import fits
from astropy import wcs

from pyklip.instruments.Instrument import Data


class ALESData(Data):
    """
    pyKLIP-compatible dataset class for ALES IFS datacubes.
    
    Inherits from pyklip.instruments.Instrument.Data and implements
    all required attributes and methods for pyKLIP PSF subtraction.
    
    Parameters
    ----------
    filepaths : str or list of str
        Path(s) to NALES-reduced FITS datacubes. Can be a glob pattern,
        a list of paths, or a single path.
    
    highpass : bool or float, optional
        If True, apply high-pass filter with default size (image_size/10).
        If float, use as the filter size in pixels.
        Default is False (no filtering).
    
    skipslices : list of int, optional
        List of wavelength slice indices to skip (e.g., bad channels).
        Default is None (use all slices).
    
    IWA : float, optional
        Inner working angle in pixels. Default is 0.
    
    OWA : float, optional  
        Outer working angle in pixels. Default is None (use image size).
    
    plate_scale : float, optional
        Plate scale in arcseconds per pixel. Default is 0.0107 (ALES).
        Used for output metadata only, not for KLIP processing.
    
    Attributes
    ----------
    input : ndarray
        3D array of shape (N_frames, y, x) containing all science frames.
        For IFS data, N_frames = N_cubes * N_wavelengths.
    
    centers : ndarray
        Array of shape (N_frames, 2) with [x, y] center for each frame.
    
    wvs : ndarray
        Array of shape (N_frames,) with wavelength of each frame in microns.
    
    PAs : ndarray
        Array of shape (N_frames,) with parallactic angle of each frame in degrees.
    
    wcs : list
        List of WCS objects for each frame (or None if not available).
    
    IWA : float
        Inner working angle in pixels.
    
    OWA : float
        Outer working angle in pixels.
    
    filenums : ndarray
        Array indicating which file each frame came from.
    
    filenames : list
        List of input filenames.
    
    prihdrs : list
        List of primary headers from input files.
    
    exthdrs : list
        List of extension headers from input files.
    
    psfs : ndarray or None
        PSF cube after calling generate_psfs(). Shape (N_wavelengths, y, x).
    
    dn_per_contrast : ndarray or None
        Flux conversion factors after calling compute_dn_per_contrast().
    
    Notes
    -----
    ALES (Arizona Lenslets for Exoplanet Spectroscopy) is a lenslet-based
    integral field spectrograph at the Large Binocular Telescope. It provides
    diffraction-limited imaging spectroscopy in the L-band (2.9-4.1 µm) with
    R ~ 20-40 spectral resolution.
    
    The NALES pipeline produces datacubes with wavelengths stored in a
    binary table extension rather than WCS keywords, which this interface
    reads correctly.
    """
    
    # ALES instrument parameters
    lenslet_scale = 0.0107  # arcsec/pixel (plate scale)
    observatory_latitude = 32.7016  # LBT latitude in degrees
    
    def __init__(self, filepaths, highpass=False, skipslices=None,
                 IWA=0, OWA=None, plate_scale=None):
        """
        Initialize ALESData from NALES-reduced FITS files.
        """
        # Call parent class __init__
        super(ALESData, self).__init__()
        
        # Handle different input types for filepaths
        if isinstance(filepaths, str):
            # Could be a glob pattern or single file
            expanded = glob_module.glob(filepaths)
            if len(expanded) == 0:
                # Maybe it's a literal path
                if os.path.exists(filepaths):
                    filepaths = [filepaths]
                else:
                    raise FileNotFoundError(f"No files found matching: {filepaths}")
            else:
                filepaths = sorted(expanded)
        else:
            filepaths = sorted(list(filepaths))
        
        if len(filepaths) == 0:
            raise ValueError("No input files provided")
        
        # Store parameters
        self._highpass = highpass
        self._skipslices = skipslices if skipslices is not None else []
        self._IWA = IWA
        self._OWA = OWA
        self._plate_scale = plate_scale if plate_scale is not None else self.lenslet_scale
        
        # Initialize storage for PSF and flux calibration (set by later methods)
        self.psfs = None
        self.psf_flux = None
        self.psf_exptime_scale = 1.0
        self.dn_per_contrast = None
        self.stellar_model_flux = None
        
        # Read all files
        self._read_data(filepaths)
        
        # Apply high-pass filter if requested
        if highpass:
            self._apply_highpass_filter(highpass)
    
    def _read_data(self, filepaths):
        """
        Read all FITS files and populate dataset attributes.
        
        Parameters
        ----------
        filepaths : list of str
            List of paths to FITS files.
        """
        # Storage for data from all files
        all_data = []
        all_centers = []
        all_wvs = []
        all_pas = []
        all_wcs = []
        all_filenums = []
        
        self.filenames = []
        self.prihdrs = []
        self.exthdrs = []
        
        for filenum, filepath in enumerate(filepaths):
            # Read this file
            cube, wavelengths, center, pa, prihdr, exthdr, frame_wcs = \
                self._read_file(filepath, filenum)
            
            # Store headers
            self.filenames.append(filepath)
            self.prihdrs.append(prihdr)
            self.exthdrs.append(exthdr)
            
            # Get dimensions
            n_wv, ny, nx = cube.shape
            
            # Skip specified wavelength slices
            good_slices = [i for i in range(n_wv) if i not in self._skipslices]
            
            # Flatten cube: each wavelength slice becomes a separate "frame"
            for wv_idx in good_slices:
                all_data.append(cube[wv_idx])
                all_centers.append(center)
                all_wvs.append(wavelengths[wv_idx])
                all_pas.append(pa)
                all_wcs.append(frame_wcs)
                all_filenums.append(filenum)
        
        # Convert to arrays
        self.input = np.array(all_data)
        self.centers = np.array(all_centers)
        self.wvs = np.array(all_wvs)
        self.PAs = np.array(all_pas)
        self.wcs = all_wcs
        self.filenums = np.array(all_filenums)
        
        # Set IWA and OWA
        self.IWA = self._IWA
        if self._OWA is None:
            # Default OWA is half the image size
            ny, nx = self.input.shape[1:3]
            self.OWA = min(nx, ny) // 2
        else:
            self.OWA = self._OWA
        
        # Store number of wavelengths (from first file, assuming all same)
        self.numwvs = len([i for i in range(n_wv) if i not in self._skipslices])
        
        # Store unique wavelengths
        self._unique_wvs = np.unique(self.wvs)
    
    def _read_file(self, filepath, filenum):
        """
        Read a single NALES FITS file.
        
        Parameters
        ----------
        filepath : str
            Path to FITS file.
        filenum : int
            Index of this file in the dataset.
        
        Returns
        -------
        cube : ndarray
            3D datacube (n_wavelengths, y, x).
        wavelengths : ndarray
            1D array of wavelengths in microns.
        center : tuple
            (x, y) center coordinates.
        pa : float
            Parallactic angle in degrees.
        prihdr : fits.Header
            Primary header.
        exthdr : fits.Header or None
            Extension header.
        frame_wcs : WCS or None
            WCS object if available.
        """
        with fits.open(filepath) as hdulist:
            # Read primary data (3D cube)
            cube = hdulist[0].data.astype(np.float64)
            prihdr = hdulist[0].header
            
            # Read wavelengths from binary table extension
            if len(hdulist) > 1:
                exthdr = hdulist[1].header
                try:
                    wavelengths = hdulist[1].data['WAVELENGTH'].astype(np.float64)
                except KeyError:
                    # Try alternate column names
                    try:
                        wavelengths = hdulist[1].data['WAVE'].astype(np.float64)
                    except KeyError:
                        warnings.warn(
                            f"No WAVELENGTH column found in {filepath}. "
                            f"Using sequential indices."
                        )
                        wavelengths = np.arange(cube.shape[0], dtype=np.float64)
            else:
                exthdr = None
                warnings.warn(
                    f"No wavelength extension in {filepath}. "
                    f"Using sequential indices."
                )
                wavelengths = np.arange(cube.shape[0], dtype=np.float64)
            
            # Get parallactic angle
            try:
                pa = float(prihdr['LBT_PARA'])
            except KeyError:
                warnings.warn(
                    f"No LBT_PARA keyword in {filepath}. "
                    f"Setting parallactic angle to 0."
                )
                pa = 0.0
            
            # Get exposure time (for reference, stored in header)
            try:
                self._exptime = float(prihdr['EXPTIME'])
            except KeyError:
                self._exptime = None
            
            # Determine image center
            center = self._find_center(cube, prihdr)
            
            # Try to get WCS
            try:
                frame_wcs = wcs.WCS(prihdr)
            except Exception:
                frame_wcs = None
        
        return cube, wavelengths, center, pa, prihdr, exthdr, frame_wcs
    
    def _find_center(self, cube, header):
        """
        Determine the center of the image/PSF.
        
        Parameters
        ----------
        cube : ndarray
            3D datacube.
        header : fits.Header
            FITS header.
        
        Returns
        -------
        center : ndarray
            [x, y] center coordinates.
        """
        ny, nx = cube.shape[1:3]
        
        # Try header keywords first
        center_keywords = [
            ('PSFCENTX', 'PSFCENTY'),
            ('CRPIX1', 'CRPIX2'),
            ('CENX', 'CENY'),
        ]
        
        for xkey, ykey in center_keywords:
            if xkey in header and ykey in header:
                return np.array([float(header[xkey]), float(header[ykey])])
        
        # Default to geometric center
        return np.array([nx / 2.0, ny / 2.0])
    
    def _apply_highpass_filter(self, highpass):
        """
        Apply high-pass filter to all frames.
        
        Parameters
        ----------
        highpass : bool or float
            If True, use default filter size (image_size/10).
            If float, use as filter size in pixels.
        """
        from scipy.ndimage import gaussian_filter
        
        # Determine filter size
        if highpass is True:
            filtersize = self.input.shape[1] / 10.0
        else:
            filtersize = float(highpass)
        
        # Apply to each frame
        for i in range(self.input.shape[0]):
            lowpass = gaussian_filter(self.input[i], sigma=filtersize)
            self.input[i] -= lowpass
    
    # =========================================================================
    # Required pyklip.instruments.Instrument.Data methods
    # =========================================================================
    
    def savedata(self, filepath, data, klipparams=None, filetype='',
                 zaxis=None, more_keywords=None, center=None,
                 astr_hdr=None, fakePlparams=None):
        """
        Save processed data to FITS file.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        data : ndarray
            Data to save.
        klipparams : str, optional
            KLIP parameters string for header.
        filetype : str, optional
            Type of file being saved.
        zaxis : ndarray, optional
            Values for z-axis (e.g., KL mode numbers).
        more_keywords : dict, optional
            Additional FITS keywords.
        center : tuple, optional
            Center coordinates.
        astr_hdr : fits.Header, optional
            Astrometry header.
        fakePlparams : optional
            Fake planet parameters.
        """
        hdulist = fits.HDUList()
        
        # Use first file's header as template
        prihdr = self.prihdrs[0].copy()
        
        # Update header with processing info
        prihdr['PROCTYPE'] = ('KLIP', 'Processing type')
        if klipparams is not None:
            prihdr['KLIPPARS'] = (klipparams[:70], 'KLIP parameters')
        if filetype:
            prihdr['FILETYPE'] = (filetype, 'Output file type')
        
        # Add center info
        if center is not None:
            prihdr['PSFCENTX'] = (center[0], 'PSF center X')
            prihdr['PSFCENTY'] = (center[1], 'PSF center Y')
        elif hasattr(self, 'output_centers') and self.output_centers is not None:
            prihdr['PSFCENTX'] = (self.output_centers[0, 0], 'PSF center X')
            prihdr['PSFCENTY'] = (self.output_centers[0, 1], 'PSF center Y')
        
        # Add plate scale
        prihdr['PIXSCALE'] = (self._plate_scale, 'Plate scale [arcsec/pixel]')
        
        # Add any additional keywords
        if more_keywords is not None:
            for key, value in more_keywords.items():
                try:
                    prihdr[key[:8]] = value
                except Exception:
                    pass
        
        # Create primary HDU
        hdulist.append(fits.PrimaryHDU(data=data, header=prihdr))
        
        # Add wavelength extension if we have unique wavelengths
        if hasattr(self, '_unique_wvs'):
            wv_col = fits.Column(name='WAVELENGTH', format='E', unit='micron',
                                 array=self._unique_wvs)
            wv_hdu = fits.BinTableHDU.from_columns([wv_col], name='WAVELENGTH')
            hdulist.append(wv_hdu)
        
        # Add z-axis info if provided (e.g., KL modes)
        if zaxis is not None:
            zaxis_col = fits.Column(name='ZAXIS', format='E', array=zaxis)
            z_hdu = fits.BinTableHDU.from_columns([zaxis_col], name='ZAXIS')
            hdulist.append(z_hdu)
        
        # Write file
        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()
    
    def calibrate_output(self, img, spectral=False, units='contrast'):
        """
        Calibrate output image to physical units.
        
        Parameters
        ----------
        img : ndarray
            Image to calibrate.
        spectral : bool, optional
            If True, treat as spectral cube.
        units : str, optional
            Target units. Currently only 'contrast' is supported.
        
        Returns
        -------
        img : ndarray
            Calibrated image (modified in place).
        """
        if self.dn_per_contrast is None:
            warnings.warn(
                "dn_per_contrast not set. Run compute_dn_per_contrast() first. "
                "Returning uncalibrated data."
            )
            return img
        
        if units == 'contrast':
            if spectral:
                # Spectral cube: each slice gets its own calibration
                n_wv = img.shape[0]
                for i in range(n_wv):
                    if i < len(self.dn_per_contrast):
                        img[i] /= self.dn_per_contrast[i]
            else:
                # Broadband: use mean calibration
                img /= np.nanmean(self.dn_per_contrast)
        
        return img
    
    # =========================================================================
    # Convenience methods
    # =========================================================================
    
    def generate_psfs(self, boxrad=10, time_collapse='median',
                      aux_psf_files=None, aux_exptime=None, sci_exptime=None,
                      exptime_keyword='EXPTIME', normalize='none',
                      centroid_method='gaussian', background_subtract=True):
        """
        Generate instrumental PSF cube for KLIP-FM forward modeling.
        
        See nales.analysis.psf_utils.generate_psfs for full documentation.
        
        Parameters
        ----------
        boxrad : int
            Half-width of PSF stamp in pixels.
        time_collapse : str
            Method to combine PSFs: 'median', 'mean', 'weighted', or None.
        aux_psf_files : str or list, optional
            Auxiliary calibrator file(s). If None, uses central star.
        aux_exptime : float, optional
            Calibrator exposure time in seconds.
        sci_exptime : float, optional
            Science exposure time in seconds.
        exptime_keyword : str
            FITS keyword for exposure time.
        normalize : str
            Normalization: 'none', 'per_channel', or 'cube'.
        centroid_method : str
            Centroiding method: 'gaussian', 'com', or 'peak'.
        background_subtract : bool
            Whether to subtract background annulus.
        
        Returns
        -------
        psfs : ndarray
            PSF cube of shape (n_wavelengths, 2*boxrad+1, 2*boxrad+1).
        """
        from nales.analysis.psf_utils import generate_psfs as _generate_psfs
        return _generate_psfs(
            self, boxrad=boxrad, time_collapse=time_collapse,
            aux_psf_files=aux_psf_files, aux_exptime=aux_exptime,
            sci_exptime=sci_exptime, exptime_keyword=exptime_keyword,
            normalize=normalize, centroid_method=centroid_method,
            background_subtract=background_subtract
        )
    
    def compute_dn_per_contrast(self, W1_mag, spectral_type=None,
                                 Teff=None, logg=None):
        """
        Compute flux conversion factors using stellar models.
        
        See nales.analysis.flux_calibration.compute_dn_per_contrast
        for full documentation.
        
        Parameters
        ----------
        W1_mag : float
            WISE W1 magnitude of PSF reference star.
        spectral_type : str, optional
            Spectral type (e.g., 'G2V').
        Teff : float, optional
            Effective temperature in Kelvin.
        logg : float, optional
            Surface gravity log(g).
        
        Returns
        -------
        dn_per_contrast : ndarray
            Conversion factors for each wavelength.
        """
        from nales.analysis.flux_calibration import compute_dn_per_contrast as _compute
        return _compute(
            self, W1_mag=W1_mag, spectral_type=spectral_type,
            Teff=Teff, logg=logg
        )
    
    def calibrate_contrast_spectrum(self, contrast_spectrum):
        """
        Convert contrast spectrum to physical flux units.
        
        This is the final step after KLIP-FM extractSpec. Takes the
        extracted planet/star contrast ratio and multiplies by the
        stellar model flux to get planet flux in physical units.
        
        Parameters
        ----------
        contrast_spectrum : ndarray
            Planet/star contrast ratio at each wavelength.
            Output from pyklip.fmlib.extractSpec.
        
        Returns
        -------
        planet_flux : ndarray
            Planet flux in same units as stellar_model_flux
            (Jy by default, or W/m²/µm if specified).
        
        Raises
        ------
        ValueError
            If compute_dn_per_contrast() has not been called.
        
        Example
        -------
        >>> # After KLIP-FM extraction
        >>> planet_flux = dataset.calibrate_contrast_spectrum(contrast_spectrum)
        >>> # planet_flux is now in Jy
        """
        if self.stellar_model_flux is None:
            raise ValueError(
                "Stellar model not set. Run compute_dn_per_contrast() first."
            )
        
        return contrast_spectrum * self.stellar_model_flux
    
    def get_radial_extent(self):
        """
        Get the radial extent of the dataset.
        
        Returns
        -------
        IWA : float
            Inner working angle in pixels.
        OWA : float
            Outer working angle in pixels.
        """
        return self.IWA, self.OWA
    
    @property
    def unique_wvs(self):
        """Get unique wavelengths in the dataset."""
        return self._unique_wvs
    
    @property  
    def exptime(self):
        """Get exposure time from first file."""
        return getattr(self, '_exptime', None)
    
    def __repr__(self):
        """String representation."""
        n_files = len(self.filenames)
        n_frames = self.input.shape[0]
        ny, nx = self.input.shape[1:3]
        n_wv = len(self._unique_wvs)
        wv_range = f"{self._unique_wvs.min():.3f}-{self._unique_wvs.max():.3f}"
        
        return (
            f"ALESData(\n"
            f"  files: {n_files}\n"
            f"  frames: {n_frames} ({n_files} cubes × {n_wv} wavelengths)\n"
            f"  image size: {nx} × {ny} pixels\n"
            f"  wavelengths: {n_wv} channels, {wv_range} µm\n"
            f"  IWA/OWA: {self.IWA:.1f}/{self.OWA:.1f} pixels\n"
            f"  PSFs loaded: {self.psfs is not None}\n"
            f"  Flux calibrated: {self.dn_per_contrast is not None}\n"
            f")"
        )
