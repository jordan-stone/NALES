"""
Sub-pixel image registration for ALES calibration.

This module implements efficient sub-pixel image registration using
discrete Fourier transforms, based on the algorithm of 
Guizar-Sicairos, Thurman, and Fienup (2008).

Heritage
--------
This implementation was developed collaboratively between the nales and
MEAD pipelines. Original nales code was extended by Zackery Briesemeister
for the MEAD pipeline (Briesemeister et al. 2018, SPIE 10702), and those
improvements were subsequently incorporated back into nales.

References
----------
Briesemeister, Z., Skemer, A. J., Stone, J. M., et al. 2018, Proc. SPIE,
    10702, 107022Q. "MEAD: data reduction pipeline for ALES integral field
    spectrograph and LBTI thermal infrared calibration unit"
"""

import numpy as np


class SubPixelRegister:
    """
    Sub-pixel image registration using DFT upsampling.
    
    Implements efficient arbitrary-precision image registration
    (to 1/k pixel accuracy) using a single-step DFT approach.
    
    Parameters
    ----------
    g_a : numpy.ndarray
        Reference image (2D array).
        
    Attributes
    ----------
    G_a : numpy.ndarray
        FFT of the reference image.
    shape : numpy.ndarray
        Image dimensions.
        
    Notes
    -----
    Based on the second algorithm of:
    
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    "Efficient subpixel image registration algorithms,"
    Optics Letters 33, 156-158 (2008).
    
    Adapted from Zack Briesemeister's implementation.
    
    Examples
    --------
    >>> import numpy as np
    >>> ref_image = fits.getdata('reference.fits')
    >>> target_image = fits.getdata('target.fits')
    >>> 
    >>> register = SubPixelRegister(ref_image)
    >>> shift = register(target_image, k=100)  # 1/100 pixel precision
    >>> print(f"Shift: {shift} pixels")
    """
    
    def __init__(self, g_a):
        self.G_a = np.fft.fft2(g_a)
        self.shape = np.array(g_a.shape)
        self.range_x = np.arange(self.shape[1])
        self.range_y = np.arange(self.shape[0])

    def _build_exclusion_matrix(self, direction=None):
        """
        Build mask to exclude specific quadrants of the correlogram.
        
        Parameters
        ----------
        direction : str
            Which quadrant to search: '++', '--', '-+', or '+-'
            Corresponds to the sign of shift in (row, col).
            
        Returns
        -------
        mask : numpy.ndarray
            Boolean mask (True = excluded).
        """
        mask = np.zeros(self.shape, bool)
        mids = self.shape.astype(int) // 2
        
        if direction == '++':
            mask[mids[0]:] = True
            mask[:, mids[1]:] = True
        elif direction == '+-':
            mask[mids[0]:] = True
            mask[:, :mids[1]] = True
        elif direction == '-+':
            mask[:mids[0]] = True
            mask[:, mids[1]:] = True
        elif direction == '--':
            mask[mids[0]:] = True
            mask[:, mids[1]:] = True
        else:
            raise ValueError("Direction must be one of ['++', '--', '-+', '+-']")
            
        return mask

    def __call__(self, g_b, k, not_origin=True, direction=None):
        """
        Register an image to the reference with sub-pixel precision.
        
        Parameters
        ----------
        g_b : numpy.ndarray
            Image to register (same shape as reference).
        k : int
            Precision factor (achieves 1/k pixel accuracy).
        not_origin : bool, optional
            If True, exclude zero shift from search. Default: True
        direction : str, optional
            If known, specify which quadrant to search ('++', '--', '-+', '+-').
            Corresponds to the sign of the shift parameter for scipy.ndimage.shift.
            
        Returns
        -------
        shift : numpy.ndarray
            Estimated (row, col) shift to align g_b to reference.
            Apply this shift to g_b using scipy.ndimage.shift.
        """
        if self.G_a.shape != g_b.shape:
            raise ValueError('Incompatible image sizes')
            
        k = int(k)

        # Phase correlation (see Wikipedia)
        G_b = np.fft.fft2(g_b)
        R = self.G_a * G_b.conj()
        r = np.fft.ifft2(R)
        
        if not_origin:
            r[0, 0] = 0.0
            
        if direction is not None:
            mask = self._build_exclusion_matrix(direction=direction)
            r[mask] = 0

        # Find peak in correlogram (integer pixel precision)
        shift = np.asarray(
            np.where(np.abs(r) == np.abs(r).max()), 
            dtype=float
        ).flatten()
        shift[shift > self.shape / 2] -= self.shape[shift > self.shape / 2]

        if k == 1:
            pass
        else:
            # Refine to sub-pixel precision using DFT upsampling
            # in a 1.5 x 1.5 pixel neighborhood
            range_k = np.arange(1.5 * k)
            offset = np.fix(1.5 * k / 2.0) - shift * k

            # Calculate upsampling kernels
            kernel_x = np.exp(
                (-1j * 2 * np.pi / (self.shape[1] * k)) * 
                (np.fft.ifftshift(self.range_x)[:, None] - (self.shape[1] / 2)).dot(
                    range_k[None, :] - offset[1]
                )
            )
            kernel_y = np.exp(
                (-1j * 2 * np.pi / (self.shape[0] * k)) * 
                (range_k[:, None] - offset[0]).dot(
                    np.fft.ifftshift(self.range_y)[None, :] - (self.shape[0] / 2)
                )
            )

            # Upsampled DFT in neighborhood
            kdft = kernel_y.dot(np.conjugate(R)).dot(kernel_x) / (
                np.multiply(*self.shape) * k**2
            )

            # Find sub-pixel peak
            shift_k = np.asarray(
                np.where(np.abs(kdft) == np.abs(kdft).max()), 
                dtype=float
            ).flatten()
            shift_k -= np.fix(1.5 * k / 2.0)
            shift = shift + shift_k / k

        return shift
