"""
Cubifier: Core IFS data cube extraction for ALES.

This module contains the Cubifier class, which handles the transformation
of 2D ALES detector images into 3D spectral datacubes.

The Cubifier performs:
1. Microspectrum location using narrow-band calibration images
2. Trace rotation angle measurement from sky frames  
3. Optimal extraction weight calculation
4. Wavelength calibration fitting
5. Spectral extraction and cube assembly
"""

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Use the new jfits package for visualization
import jfits

from nales.utils.wavecal import get_spots_grid, spoof_spots, w_func, onedmoments
from nales.utils.climb import climb, climb1d
from nales.utils.rotation import find_rotation_angle, rot
from nales.utils.bad_pixels import bfixpix


class Cubifier:
    """
    ALES IFS cube extractor.
    
    The Cubifier class transforms 2D ALES detector images into calibrated 3D 
    spectral datacubes. It learns the mapping from detector pixels to 
    (wavelength, spatial_x, spatial_y) coordinates using narrow-band calibration 
    frames and a median sky image.
    
    Once instantiated with calibration data, the Cubifier can be called on any 
    number of science frames taken with the same instrument configuration.
    
    Parameters
    ----------
    wave_cal_ims : dict
        Dictionary containing dark-subtracted narrow-band calibration images.
        Required keys: 'nb29', 'nb33', 'nb35', 'nb39' (for 2.9, 3.3, 3.5, 3.9 microns).
        Each value should be a 2D numpy array.
    sky_im : numpy.ndarray
        2D median sky image (dark-subtracted). Used to define the spatial 
        profile of each trace for optimal extraction.
    sky_var : numpy.ndarray
        2D variance image, typically np.ones_like(sky_im) works well in practice.
    start_offsets : tuple of int, REQUIRED
        Approximate pixel position (y, x) of the lower-left reference spot in
        the NB39 image. This seeds the automatic spot detection. Must be 
        determined by visual inspection - look for the spot at the intersection
        of the lowest complete row and leftmost complete column.
    dims : tuple of int, optional
        Output spatial dimensions (n_rows, n_cols) of the datacube. 
        Default: (63, 67)
    make_slice_bottoms : tuple of int, optional
        Extent below/left of reference point for trace slices (y, x). Defines
        the bounding box around each NB39 spot so that each microspectrum can
        be indexed from the full detector array. Default: (35, 35)
    gridplot : bool, optional
        If True, display a diagnostic plot showing where NB39 spots were 
        detected. Useful for debugging if the spot detection fails to find
        a regular grid pattern. Default: False
    spoof_nb29 : bool, optional
        If True, improve NB29 spot positions using NB39 as a reference. ALES
        optical quality is typically best near the frame center, so spots at
        the edges may be poorly detected in low S/N narrowband images. This
        option measures the offset between NB39 and NB29 in the high-quality
        region (defined by best_spots_slice), then applies that offset to 
        estimate NB29 positions across the full frame. Default: True
    spoof_nb33 : bool, optional
        If True, improve NB33 spot positions using NB39 as a reference (same
        technique as spoof_nb29). Default: False
    best_spots_slice : tuple of slice, optional
        Detector region with highest quality spots, used when spoofing 
        NB29/NB33 positions. Default: (slice(920, 1020), slice(900, 1000))
    use_middle_weights : bool, optional
        If True, use the median spatial profile from mid-band wavelengths for
        extraction at all wavelengths, rather than wavelength-dependent 
        profiles. Can improve stability. Default: True
    variance_weighted : bool, optional
        If True, use sky/variance for extraction weights (optimal extraction).
        If False, use sky image directly as weights, which often produces 
        better results in practice. Default: False
    trim_crosstalk : bool, optional
        If True, avoid extracting the right side of each microspectrum's 
        spatial profile in the blue third of wavelengths, reducing 
        contamination from adjacent spectra. Default: False
    zoom_factor : int, optional
        Oversampling factor for the wavelength direction. Default: 1
        
    Attributes
    ----------
    slices : dict
        Dictionary mapping (row, col) spaxel indices to detector slice objects.
    refs : dict
        Dictionary mapping spaxel indices to reference point coordinates.
    rot_angles : numpy.ndarray
        2D array of trace rotation angles in degrees.
    weights : dict
        Dictionary mapping spaxel indices to 2D extraction weight arrays.
    wavecals : dict
        Dictionary mapping spaxel indices to wavelength solution arrays.
    cal_cubes : dict
        Dictionary of extracted calibration cubes (for diagnostics).
        
    Examples
    --------
    Basic usage:
    
    >>> from nales import Cubifier
    >>> from astropy.io import fits
    >>> 
    >>> # Load calibration data
    >>> wave_cal_ims = {
    ...     'nb29': fits.getdata('nb29_dark_subtracted.fits'),
    ...     'nb33': fits.getdata('nb33_dark_subtracted.fits'),
    ...     'nb35': fits.getdata('nb35_dark_subtracted.fits'),
    ...     'nb39': fits.getdata('nb39_dark_subtracted.fits'),
    ... }
    >>> sky = fits.getdata('median_sky.fits')
    >>> 
    >>> # Build the cubifier (start_offsets from inspecting NB39 image)
    >>> cubifier = Cubifier(
    ...     wave_cal_ims, sky, np.ones_like(sky),
    ...     start_offsets=(77, 193)
    ... )
    >>> 
    >>> # Extract a cube
    >>> science = fits.getdata('science_frame.fits')
    >>> cube, wavelengths = cubifier(science)
    >>> print(f"Cube shape: {cube.shape}")  # (n_wavelengths, 67, 64)
    
    Processing multiple frames:
    
    >>> import glob
    >>> cubes = []
    >>> for f in glob.glob('science_*.fits'):
    ...     cube, wl = cubifier(fits.getdata(f))
    ...     cubes.append(cube)
    >>> stack = np.array(cubes)  # Shape: (n_frames, n_wavelengths, 67, 64)
    
    Notes
    -----
    The Cubifier is designed to be built once per observing night (or whenever
    the lenslet array mechanism is moved). The same Cubifier instance can then
    be used to process hundreds or thousands of science frames efficiently.
    
    The wavelength solution uses a functional form:
        wavelength = A * sqrt(pixel - B) + C
    where A, B, C are fitted for each spaxel using the four narrow-band
    calibration points.
    """
    
    def __init__(
        self,
        wave_cal_ims,
        sky_im,
        sky_var,
        start_offsets,
        make_slice_bottoms=(35, 35),
        gridplot=False,
        dims=(63, 67),
        zoom_factor=1,
        spoof_nb29=True,
        spoof_nb33=False,
        best_spots_slice=(slice(920, 1020), slice(900, 1000)),
        use_middle_weights=True,
        variance_weighted=False,
        trim_crosstalk=False,
    ):
        # Validate required parameters
        if start_offsets is None:
            raise ValueError(
                "start_offsets is required. Inspect your NB39 calibration image "
                "to find the approximate (y, x) pixel position of the lower-left "
                "spot in the grid (intersection of lowest complete row and "
                "leftmost complete column)."
            )
        
        # Store input parameters
        self.wave_cal_ims = wave_cal_ims
        self.make_slice_bottoms = make_slice_bottoms
        self.ref_to_top = 65
        self.ref_to_right = 20
        self.zoom_factor = zoom_factor
        self.sky_im = sky_im
        self.sky_var = sky_var
        self.use_middle_weights = use_middle_weights
        self.variance_weighted = variance_weighted
        self.trim_crosstalk = trim_crosstalk
        self.slices = None
        self.wavecals = None
        self.dims = dims
        
        # Calculate required stamp size to avoid clipping during rotation
        # Based on distance from reference point to trace corner
        self.sz = 2 * int(np.ceil((60**2 + 20**2)**0.5))

        # Optionally improve weak narrow-band spot positions
        if spoof_nb29:
            nb29_new = spoof_spots(
                self.wave_cal_ims['nb29'],
                self.wave_cal_ims['nb39'],
                best_spots_slice=best_spots_slice
            )
            self.wave_cal_ims['nb29_old'] = self.wave_cal_ims['nb29'].copy()
            self.wave_cal_ims['nb29'] = nb29_new

        if spoof_nb33:
            nb33_new = spoof_spots(
                self.wave_cal_ims['nb33'],
                self.wave_cal_ims['nb39'],
                best_spots_slice=best_spots_slice
            )
            self.wave_cal_ims['nb33_old'] = self.wave_cal_ims['nb33'].copy()
            self.wave_cal_ims['nb33'] = nb33_new

        # Detect spot grid positions using NB39 (brightest spots)
        self.nb39_spots = get_spots_grid(
            wave_cal_ims['nb39'],
            start_offsets=start_offsets,
            plot=gridplot,
            dims=self.dims
        )
        
        # Build the cubifier calibration
        print("Initializing Cubifier...")
        self._make_slices()
        self._make_trace_mask()
        self._find_rotation_angles()
        self._make_weights()
        self._make_cal_traces()
        self._wavecal()
        print("Cubifier ready.")

    def _make_slices(self):
        """
        Create detector slice objects for each spaxel.
        
        Uses the detected NB39 spot positions as reference points to define
        rectangular extraction regions on the detector for each spaxel.
        """
        print('Making slices for each trace...')
        self.slices = {}
        self.refs = {}
        
        for ii, row in enumerate(list(zip(*self.nb39_spots))):
            for ll, point in enumerate(list(zip(*row))):
                self.refs[(ii, ll)] = point
                p0 = int(np.rint(point[0]))
                p1 = int(np.rint(point[1]))
                self.slices[(ii, ll)] = (
                    slice(p0 - self.make_slice_bottoms[0], p0 + self.ref_to_top),
                    slice(p1 - self.make_slice_bottoms[1], p1 + self.ref_to_right)
                )

    def _make_trace_mask(self, slope=-3, offsets=(-19, 21)):
        """
        Create a mask to isolate each trace from its neighbors.
        
        The microspectra are dispersed at an angle across the detector.
        This mask defines the region containing only the target trace,
        excluding light from adjacent spectra.
        
        Parameters
        ----------
        slope : float
            Approximate slope of the trace dispersion direction.
        offsets : tuple
            Lower and upper bounds relative to reference point.
        """
        refy, refx = self.make_slice_bottoms
        self.mask = np.ones(
            (refy + self.ref_to_top, refx + self.ref_to_right), 
            dtype=int
        )
        indices = np.indices(self.mask.shape)
        self.mask[indices[0] > slope * (indices[1] - refx) + refy + offsets[1]] = 0
        self.mask[indices[0] < slope * (indices[1] - refx) + refy + offsets[0]] = 0

    def _find_rotation_angles(self, slope=-3, flip=True, leftof=5, rightof=9):
        """
        Measure the rotation angle of each trace.
        
        The microspectra are not perfectly horizontal on the detector.
        This method measures the angle of each trace relative to the
        detector rows, enabling proper alignment during extraction.
        """
        self.leftof = leftof
        self.rightof = rightof
        
        if self.mask is None:
            self._make_trace_mask()
            
        extra = 180.0 if flip else 0.0
        self.rot_angles = np.empty(self.dims)

        for spaxel in list(self.slices.keys()):
            stamp = self.mask * self.sky_im[self.slices[spaxel]]
            self.rot_angles[spaxel] = find_rotation_angle(
                stamp,
                self.make_slice_bottoms[::-1],
                slope=slope,
                leftof=leftof,
                rightof=rightof
            ) + extra

    def _weights_im(
        self, 
        spaxel, 
        smooth_kern_sky=1.4, 
        smooth_kern_var=2,
        plot=False, 
        flip=True, 
        use_middle_weights=False,
        variance_weighted=False, 
        trim_crosstalk=True
    ):
        """
        Compute optimal extraction weights for a single spaxel.
        
        Weights are computed as sky / variance (proportional to S/N^2),
        rotated to align with the trace, and normalized.
        
        Parameters
        ----------
        spaxel : tuple
            (row, col) index of the spaxel.
        smooth_kern_sky : float
            Gaussian smoothing kernel size for sky image.
        smooth_kern_var : float
            Gaussian smoothing kernel size for variance image.
        plot : bool
            If True, display diagnostic plots.
        flip : bool
            If True, flip the mask orientation.
        use_middle_weights : bool
            If True, use weights from trace center for all wavelengths.
        variance_weighted : bool
            If True, use sky/variance for weights. If False, use sky directly.
        trim_crosstalk : bool
            If True, mask crosstalk-affected regions.
            
        Returns
        -------
        weights : numpy.ndarray
            2D normalized extraction weight array.
        """
        stamp = gaussian_filter(
            self.sky_im[self.slices[spaxel]], 
            smooth_kern_sky
        )
        vstamp = gaussian_filter(
            self.sky_var[self.slices[spaxel]], 
            smooth_kern_var
        )
        
        # Rotate to align trace with rows
        rstamp = rot(
            stamp, 
            self.rot_angles[spaxel],
            self.make_slice_bottoms[::-1],
            out_size=(self.sz, self.sz)
        )
        rvstamp = rot(
            vstamp, 
            self.rot_angles[spaxel],
            self.make_slice_bottoms[::-1],
            out_size=(self.sz, self.sz)
        )
        
        # Compute weights (PSF^2/variance, proportional to PSF for optimal extraction)
        if variance_weighted:
            w0 = rstamp / rvstamp
        else:
            w0 = rstamp
            
        # Apply spatial mask
        mask = np.ones_like(w0)
        mask[:, :(self.sz // 2) - 4] = 0
        mask[:, (self.sz // 2) + 4:] = 0
        
        if trim_crosstalk:
            # Mask region where crosstalk from adjacent traces is significant
            mask[90:, :self.sz // 2] = 0
            
        if flip:
            mask = np.rot90(np.rot90(mask))
            
        w = mask * w0
        w[np.isnan(w)] = 0
        
        # Apply zoom factor for wavelength oversampling
        ww = zoom(w, (self.zoom_factor, 1))
        www = ww / np.sum(ww, axis=1)[:, None]
        www[np.isnan(www)] = 0
        
        if use_middle_weights:
            # Use profile from trace center for all wavelengths
            # This can be more stable when S/N varies along trace
            prof = np.mean(www[(self.sz // 2) - 8:(self.sz // 2) + 8, :], axis=0)
            prof /= prof.sum()
            www = mask * prof[None, :]
            www /= www.sum(1)[:, None]
            www[np.isnan(www)] = 0
            
        if plot:
            # Display diagnostic images using jfits
            d = jfits.InteractiveDisplay(self.sky_im[self.slices[spaxel]])
            d.ax.set_title('sky')
            d = jfits.InteractiveDisplay(stamp)
            d.ax.set_title('smooth sky')
            d = jfits.InteractiveDisplay(vstamp)
            d.ax.set_title('smooth sky variance')
            d = jfits.InteractiveDisplay(rstamp)
            d.ax.set_title('smooth rot sky')
            d = jfits.InteractiveDisplay(rvstamp)
            d.ax.set_title('smooth rot sky variance')
            d = jfits.InteractiveDisplay(w0)
            d.ax.set_title('weights')
            d = jfits.InteractiveDisplay(www)
            d.ax.set_title('trimmed weights')
            d.ax.contour(w0, colors='w')
            
        return www

    def _make_weights(self, smooth_kern_sky=1.4, smooth_kern_var=2):
        """
        Compute extraction weights for all spaxels.
        """
        print("Computing extraction weights...")
        self.weights = {}
        for spaxel in list(self.slices.keys()):
            self.weights[spaxel] = self._weights_im(
                spaxel,
                smooth_kern_sky=smooth_kern_sky,
                smooth_kern_var=smooth_kern_var,
                use_middle_weights=self.use_middle_weights,
                variance_weighted=self.variance_weighted,
                trim_crosstalk=self.trim_crosstalk
            )

    def _extract(self, arr, spaxel):
        """
        Extract a 1D spectrum from a 2D image for one spaxel.
        
        Parameters
        ----------
        arr : numpy.ndarray
            2D detector image.
        spaxel : tuple
            (row, col) index of the spaxel.
            
        Returns
        -------
        spec : numpy.ndarray
            1D extracted spectrum.
        """
        stamp0 = rot(
            arr[self.slices[spaxel]],
            self.rot_angles[spaxel],
            self.make_slice_bottoms[::-1],
            out_size=(self.sz, self.sz)
        )
        stamp0[np.isnan(stamp0)] = 0
        stamp = zoom(stamp0, (self.zoom_factor, 1))
        
        # Optimal extraction: weighted sum / sum of weights
        spec = (
            np.sum(self.weights[spaxel] * stamp, axis=1) / 
            np.sum(self.weights[spaxel], axis=1)
        )
        return spec

    def _make_cal_traces(self):
        """
        Extract calibration cubes from narrow-band images.
        
        These are used to fit the wavelength solution by measuring
        where the narrow-band spots fall in pixel coordinates.
        """
        cal_cubes = {}
        
        for key in self.wave_cal_ims.keys():
            print(f'Making cal cube: {key}')
            arr = self.wave_cal_ims[key]
            out_arr = np.empty((
                self.sz * self.zoom_factor,
                self.dims[0],
                self.dims[1]
            ))
            for spaxel in list(self.slices.keys()):
                spec = self._extract(arr, spaxel)
                out_arr[:, spaxel[0], spaxel[1]] = spec
            cal_cubes[key] = out_arr

        print('Making cal cube: sky')
        arr = self.sky_im
        out_arr = np.empty((
            self.sz * self.zoom_factor,
            self.dims[0],
            self.dims[1]
        ))
        for spaxel in list(self.slices.keys()):
            spec = self._extract(arr, spaxel)
            out_arr[:, spaxel[0], spaxel[1]] = spec
        cal_cubes['sky'] = out_arr
        
        self.cal_cubes = cal_cubes

    def _wavecal_coeffs(self):
        """
        Fit wavelength solution coefficients for all spaxels.
        
        Uses the four narrow-band calibration points to fit the
        wavelength function: wavelength = A * sqrt(pixel - B) + C
        """
        print("Fitting wavelength solutions...")
        self.wsoln_coeffs_nans = np.empty((3, self.dims[0], self.dims[1]))
        self.w_pixels = np.empty((4, self.dims[0], self.dims[1]))
        self.wsoln_coeffs = np.empty((3, self.dims[0], self.dims[1]))
        self.reuse = 0

        # Known wavelengths for each narrow-band filter (microns)
        wavelengths = [2.897, 3.360, 3.539, 3.874]
        
        for spaxel in self.slices.keys():
            # Find centroid of each narrow-band spot in extracted spectrum
            nb29_slices = (slice(1, 11),) + spaxel
            px29 = onedmoments(self.cal_cubes['nb29'][nb29_slices])[1] + 1
            
            nb33_slices = (slice(25, 35),) + spaxel
            px33 = onedmoments(self.cal_cubes['nb33'][nb33_slices])[1] + 25
            
            nb35_slices = (slice(35, 45),) + spaxel
            px35 = onedmoments(self.cal_cubes['nb35'][nb35_slices])[1] + 35
            
            nb39_slices = (slice(59, 69),) + spaxel
            px39 = onedmoments(self.cal_cubes['nb39'][nb39_slices])[1] + 59
            
            self.w_pixels[:, spaxel[0], spaxel[1]] = [px29, px33, px35, px39]

            try:
                popt, _ = curve_fit(
                    w_func,
                    [px29, px33, px35, px39],
                    wavelengths,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf])
                )
                self.wsoln_coeffs_nans[:, spaxel[0], spaxel[1]] = popt
            except (ValueError, RuntimeError):
                self.wsoln_coeffs_nans[:, spaxel[0], spaxel[1]] = [np.nan, np.nan, np.nan]

        # Fill in any failed fits using neighbors
        for ii in range(3):
            self.wsoln_coeffs[ii] = bfixpix(
                self.wsoln_coeffs_nans[ii],
                np.isnan(self.wsoln_coeffs_nans[ii]),
                retdat=True
            )

    def _wavecal(self):
        """
        Apply wavelength solution to create wavelength arrays for all spaxels.
        """
        self.wavecals = {}
        self._wavecal_coeffs()
        
        for spaxel in self.slices.keys():
            pixels = np.arange(len(self.cal_cubes['sky'][:, spaxel[0], spaxel[1]]))
            self.wavecals[spaxel] = w_func(
                pixels,
                *self.wsoln_coeffs[:, spaxel[0], spaxel[1]]
            )

    def __call__(self, arr, wave_anchor_spaxel=(30, 30)):
        """
        Extract a spectral cube from a 2D detector image.
        
        Parameters
        ----------
        arr : numpy.ndarray
            2D detector image to extract.
        wave_anchor_spaxel : tuple, optional
            Reference spaxel for the output wavelength grid. All spaxels 
            are interpolated onto this spaxel's wavelength solution.
            Default: (30, 30)
            
        Returns
        -------
        cube : numpy.ndarray
            3D spectral cube with shape (n_wavelengths, n_rows, n_cols).
            Each slice cube[i,:,:] is a monochromatic image at wavelengths[i].
        wavelengths : numpy.ndarray
            1D array of wavelengths in microns corresponding to cube axis 0.
            
        Examples
        --------
        >>> cube, wavelengths = cubifier(science_frame)
        >>> print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} microns")
        >>> 
        >>> # Extract spectrum at a specific spatial position
        >>> spectrum = cube[:, 30, 30]
        >>> plt.plot(wavelengths, spectrum)
        """
        # Determine output wavelength indices (valid range: 2.7-4.3 microns)
        anchor_wl = self.wavecals[wave_anchor_spaxel]
        out_inds = np.logical_and(anchor_wl > 2.7, anchor_wl < 4.3)
        
        out_arr = np.empty((np.sum(out_inds), self.dims[0], self.dims[1]))
        
        for spaxel in list(self.slices.keys()):
            # Extract spectrum
            spec = self._extract(arr, spaxel)
            
            # Interpolate onto common wavelength grid
            interpspec = interp1d(
                self.wavecals[spaxel],
                spec,
                kind='cubic',
                fill_value='extrapolate'
            )
            out_arr[:, spaxel[0], spaxel[1]] = interpspec(anchor_wl[out_inds])
            
        return out_arr, anchor_wl[out_inds]

    # =========================================================================
    # Diagnostic / Inspection Methods
    # =========================================================================
    
    def inspect_wavecal(self, spaxel):
        """
        Display wavelength calibration diagnostics for a spaxel.
        
        Shows the extracted narrow-band profiles and the fitted
        wavelength solution.
        
        Parameters
        ----------
        spaxel : tuple
            (row, col) index of the spaxel to inspect.
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        
        # Plot narrow-band profiles
        ax1.plot(
            self.cal_cubes['nb29'][1:11, spaxel[0], spaxel[1]],
            color='c', label='NB29 (2.9 microns)'
        )
        ax1.plot(
            self.cal_cubes['nb33'][25:35, spaxel[0], spaxel[1]],
            color='g', label='NB33 (3.3 microns)'
        )
        ax1.plot(
            self.cal_cubes['nb35'][35:45, spaxel[0], spaxel[1]],
            color='orange', label='NB35 (3.5 microns)'
        )
        ax1.plot(
            self.cal_cubes['nb39'][59:69, spaxel[0], spaxel[1]],
            color='r', label='NB39 (3.9 microns)'
        )

        # Mark centroids
        nb29_slices = (slice(1, 11),) + spaxel
        px29 = onedmoments(self.cal_cubes['nb29'][nb29_slices])[1]
        ax1.axvline(px29, color='c', linestyle='--')

        nb33_slices = (slice(25, 35),) + spaxel
        px33 = onedmoments(self.cal_cubes['nb33'][nb33_slices])[1]
        ax1.axvline(px33, color='g', linestyle='--')

        nb35_slices = (slice(35, 45),) + spaxel
        px35 = onedmoments(self.cal_cubes['nb35'][nb35_slices])[1]
        ax1.axvline(px35, color='orange', linestyle='--')

        nb39_slices = (slice(59, 69),) + spaxel
        px39 = onedmoments(self.cal_cubes['nb39'][nb39_slices])[1]
        ax1.axvline(px39, color='r', linestyle='--')
        
        ax1.set_xlabel('Relative Pixel')
        ax1.set_ylabel('Counts')
        ax1.set_title(f'Narrow-band profiles for spaxel {spaxel}')
        ax1.legend()

        # Plot wavelength solution
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(
            self.w_pixels[:, spaxel[0], spaxel[1]],
            [2.897, 3.360, 3.539, 3.874],
            'ko', markersize=10, label='Calibration points'
        )
        ax2.plot(
            np.arange(len(self.cal_cubes['sky'][:, spaxel[0], spaxel[1]])),
            self.wavecals[spaxel],
            'b-', label='Fitted solution'
        )
        ax2.set_xlabel('Pixel')
        ax2.set_ylabel('Wavelength (microns)')
        ax2.set_title(f'Wavelength solution for spaxel {spaxel}')
        ax2.legend()
        
        plt.show()

    def inspect_wavecal_nbs(self, spaxel, bands=('nb39', 'nb35', 'nb33', 'nb29')):
        """
        Display narrow-band calibration images overlaid on sky.
        
        Parameters
        ----------
        spaxel : tuple
            (row, col) index of the spaxel to inspect.
        bands : tuple of str, optional
            Which narrow-band filters to overlay.
        """
        colors = {'nb39': 'r', 'nb35': 'orange', 'nb33': 'g', 'nb29': 'c'}
        slc = self.slices[spaxel]
        
        disp = jfits.InteractiveDisplay(self.sky_im[slc])
        for wl in bands:
            if wl in self.wave_cal_ims:
                disp.ax.contour(
                    self.mask * self.wave_cal_ims[wl][slc],
                    colors=colors[wl]
                )
        disp.ax.set_title(f'Spaxel {spaxel}: Sky with NB overlays')
        plt.show()

    def inspect_rotation(self, spaxel, flip=True):
        """
        Display trace rotation angle measurement diagnostics.
        
        Parameters
        ----------
        spaxel : tuple
            (row, col) index of the spaxel to inspect.
        flip : bool, optional
            Whether to apply 180° flip to angle.
        """
        stamp = self.mask * self.sky_im[self.slices[spaxel]]
        extra = 180.0 if flip else 0.0
        
        angle = find_rotation_angle(
            stamp,
            self.make_slice_bottoms[::-1],
            plot=True,
            leftof=self.leftof,
            rightof=self.rightof
        )
        angle += extra
        
        # Show rotated result
        rotated = rot(stamp, angle, self.make_slice_bottoms[::-1])
        jfits.InteractiveDisplay(rotated)
        plt.show()

    def inspect_weights(self, spaxel, smooth_kern_sky=1.4, smooth_kern_var=2):
        """
        Display extraction weight diagnostics for a spaxel.
        
        Parameters
        ----------
        spaxel : tuple
            (row, col) index of the spaxel to inspect.
        smooth_kern_sky : float, optional
            Smoothing kernel for sky image.
        smooth_kern_var : float, optional
            Smoothing kernel for variance image.
        """
        _ = self._weights_im(
            spaxel,
            smooth_kern_sky=smooth_kern_sky,
            smooth_kern_var=smooth_kern_var,
            use_middle_weights=self.use_middle_weights,
            variance_weighted=self.variance_weighted,
            trim_crosstalk=self.trim_crosstalk,
            plot=True
        )
        plt.show()
