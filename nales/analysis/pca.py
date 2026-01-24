"""
Principal Component Analysis for high-contrast imaging post-processing.

This module provides annular PCA algorithms for removing stellar PSF
contributions from ALES datacubes, enabling detection of faint companions.
"""

import numpy as np
from multiprocessing import Pool

from nales.utils.rotation import rot


class PCA:
    """
    Annular PCA for high-contrast imaging.
    
    Implements Principal Component Analysis in annular regions for
    PSF subtraction in angular differential imaging (ADI) sequences.
    
    Parameters
    ----------
    data : numpy.ndarray
        3D array of images with shape (ny, nx, n_images).
        Note: axis order is different from typical (n_images, ny, nx).
    parallactic_angles : array_like
        Parallactic angles for each image in degrees.
    x_cen, y_cen : float, optional
        Center coordinates for annuli. Default: 150, 150
    reverse_rotate : bool, optional
        If True, negate parallactic angles. Default: False
        
    Attributes
    ----------
    data : numpy.ndarray
        Input data array.
    radius : numpy.ndarray
        2D array of distances from center.
    parallactic_angles : numpy.ndarray
        Parallactic angles (possibly negated).
        
    Examples
    --------
    >>> import numpy as np
    >>> from nales.analysis import PCA
    >>> 
    >>> # Load ADI sequence
    >>> cubes = np.array([fits.getdata(f) for f in cube_files])  # (n_images, n_waves, ny, nx)
    >>> parang = np.array([fits.getheader(f)['LBT_PARA'] for f in cube_files])
    >>> 
    >>> # Process one wavelength slice
    >>> data_slice = cubes[:, 10, :, :]  # (n_images, ny, nx)
    >>> data_reordered = np.rollaxis(data_slice, 0, 3)  # (ny, nx, n_images)
    >>> 
    >>> pca = PCA(data_reordered, parang, x_cen=150, y_cen=150)
    >>> result = pca((20, 5))  # radius=20, n_components=5
    """
    
    def __init__(self, data, parallactic_angles, x_cen=150, y_cen=150, reverse_rotate=False):
        self.data = data
        self.n_images = data.shape[2]
        self.X_DIM = data.shape[0]
        self.Y_DIM = data.shape[1]
        self.x_cen = x_cen
        self.y_cen = y_cen
        self.parallactic_angles = np.array(parallactic_angles)
        
        if reverse_rotate:
            self.parallactic_angles *= -1.0
            
        y, x = np.indices((self.Y_DIM, self.X_DIM))
        self.radius = np.abs((x - x_cen) + 1j * (y - y_cen))
        self.radius_1d = self.radius.flatten()

    def __call__(self, r_n_pca):
        """
        Perform PCA subtraction for one annulus.
        
        Parameters
        ----------
        r_n_pca : tuple
            (radius, n_components) where radius is the inner edge of the
            annulus and n_components is the number of PCA modes to subtract.
            
        Returns
        -------
        result : numpy.ndarray
            Median-combined, derotated residual image.
            
        Notes
        -----
        The annulus spans from r to r+1 pixels. The optimization region
        (used to compute PCA modes) extends ±2-3 pixels beyond this.
        """
        r, n_pca = r_n_pca
        
        # Define annular regions
        r_subt_in = r
        r_subt_out = r + 1.0
        r_opt_in = max(r_subt_in - 2.0, 1)
        r_opt_out = r_subt_in + 3.0

        # Create masks
        opt_mask = np.logical_and(self.radius > r_opt_in, self.radius <= r_opt_out)
        sub_mask = np.logical_and(self.radius > r_subt_in, self.radius <= r_subt_out)

        # Extract pixels in optimization region
        # A has shape (n_pix_opt, n_images)
        A = self.data[opt_mask, :]

        # Subtract median and mean for centering
        A -= np.median(A, axis=1)[:, None]
        A -= np.mean(A, axis=0)[None, :]

        # SVD to find principal components
        U1, S1, V1 = np.linalg.svd(A)
        eigenvalues = np.square(S1)
        eigenvectors = V1.T
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project eigenvectors back to pixel space
        eigenvectors = np.dot(A, eigenvectors)
        eigenvectors /= np.sqrt(np.sum(eigenvectors**2.0, axis=0))

        # Fit and subtract PCA model
        PCA_images_1d = np.zeros_like(A)
        for k in range(n_pca):
            coefficients = np.dot(eigenvectors[:, k], A)
            PCA_images_1d += eigenvectors[:, k, None] * coefficients

        PCA_subtracted_images = A - PCA_images_1d
        
        # Reconstruct 2D images
        output = np.zeros_like(self.data)
        output[opt_mask, :] = PCA_subtracted_images
        output *= sub_mask[:, :, None]
        
        # Derotate and median combine
        for image, q in enumerate(self.parallactic_angles):
            output[:, :, image] = rot(output[:, :, image], q, (self.x_cen, self.y_cen))
            
        return np.median(output, axis=2)


def parallel_annular_PCA(
    data,
    parallactic_angles,
    x_cen,
    y_cen,
    n_PCA,
    radii=None,
    reverse_rotate=False,
    ncpu=None
):
    """
    Perform annular PCA across all wavelengths using parallel processing.
    
    Parameters
    ----------
    data : numpy.ndarray
        4D array with shape (n_images, n_wavelengths, ny, nx).
    parallactic_angles : array_like
        Parallactic angles for each image.
    x_cen, y_cen : float
        Center coordinates for annuli.
    n_PCA : int
        Number of PCA components to subtract.
    radii : list, optional
        List of radii to process. Default: range(3, 44)
    reverse_rotate : bool, optional
        If True, negate parallactic angles. Default: False
    ncpu : int, optional
        Number of CPUs for parallel processing. Default: all available
        
    Returns
    -------
    result : numpy.ndarray
        3D output cube with shape (n_wavelengths, ny, nx).
        Each slice is the median-combined, PCA-subtracted result.
        
    Examples
    --------
    >>> import numpy as np
    >>> from nales.analysis import parallel_annular_PCA
    >>> 
    >>> # Load ADI sequence of cubes
    >>> cubes = np.array([fits.getdata(f) for f in cube_files])  # (n_images, n_waves, ny, nx)
    >>> parang = np.array([fits.getheader(f)['LBT_PARA'] for f in cube_files])
    >>> 
    >>> # Process with 5 PCA components
    >>> result = parallel_annular_PCA(
    ...     cubes, parang,
    ...     x_cen=150, y_cen=150,
    ...     n_PCA=5,
    ...     ncpu=4
    ... )
    >>> fits.writeto('pca_result.fits', result)
    
    Notes
    -----
    The two rot90 operations in the final step flip the upside-down 
    image to be right-side up (instrument-specific orientation correction).
    """
    if radii is None:
        radii = list(range(3, 44))
        
    nims, nwaves, ny, nx = data.shape
    out = np.zeros((nwaves, ny, nx), dtype=float)
    
    for lam_num in range(nwaves):
        data_lam = data[:, lam_num, :, :]
        print(f'Processing wavelength slice: {lam_num}')
        
        # Reorder axes for PCA: (ny, nx, n_images)
        do_PCA = PCA(
            np.rollaxis(data_lam, 0, 3),
            parallactic_angles,
            x_cen=x_cen,
            y_cen=y_cen,
            reverse_rotate=reverse_rotate
        )
        
        # Process annuli in parallel
        pool = Pool(ncpu)
        annuli = pool.map(
            do_PCA,
            list(zip(radii, np.zeros(len(radii), dtype=int) + n_PCA))
        )
        pool.close()
        pool.join()
        
        # Combine annuli
        combined_median = np.sum(annuli, axis=0)
        
        # Apply orientation correction
        out[lam_num, :, :] = np.rot90(np.rot90(combined_median))
        
    return out
