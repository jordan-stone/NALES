import numpy as np
'''From Zack Briesemeister's mead'''

class SubPixelRegister:

    def __init__(self, g_a):
        """subpixel registration of arbitrary (1/k pixel) accuracy, single-step DFT approach
        Computational complexity of the inverse FFT is O(ny*nx*k)

        from second algorithm of
            Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
               "Efficient subpixel image registration algorithms,"

        Arguments:
        g_a : reference image, 2D array, (ny, nx)
        """
        self.G_a = np.fft.fft2(g_a)
        self.shape = np.array(g_a.shape)
        self.range_x = np.arange(self.shape[1])
        self.range_y = np.arange(self.shape[0])

    def _build_exclusion_matrix(self, direction=None):
        """for quadrants [['-+', '--'],
                          ['++', '+-']]
        of the correlogram, for known shift direction (row, col)"""
        mask = np.zeros(self.shape, bool)
        mids = self.shape.astype(int) // 2
        if direction == '++':
            mask[mids[0]:] = 1
            mask[:, mids[1]:] = 1
        elif direction == '+-':
            mask[mids[0]:] = 1
            mask[:, :mids[1]] = 1
        elif direction == '-+':
            mask[:mids[0]] = 1
            mask[:, mids[1]:] = 1
        elif direction == '--':
            mask[mids[0]:] = 1
            mask[:, mids[1]:] = 1
        else:
            raise ValueError('Direction must be element of [\'++\', \'--\', \'-+\', \'+-\']')
        return mask

    def __call__(self, g_b, k, not_origin=True, direction=None):
        """subpixel registration of arbitrary (1/k pixel) accuracy, single-step DFT approach

        Arguments:
        g_b : image to register to g_a, 2D array, (ny, nx)
        k   : int, arbitrary precision scaling factor
        not_origin : excludes null shift
        direction : isolates known direction in correlogram, ['++', '--', '-+', '+-']
                   and corresponds to the sign of the entries in the "shift" argument 
                   of the scipy.ndimage.interpolation.shift function.
        """
        if self.G_a.shape != g_b.shape:
            raise ValueError('Incompatible sizes')
        k = int(k)

        # perform phase correlation from https://en.wikipedia.org/wiki/Phase_correlation
        G_b = np.fft.fft2(g_b)
        R = self.G_a * G_b.conj()
        r = np.fft.ifft2(R)
        if not_origin:
            r[0, 0] = 0.
        if direction is not None:
            mask = self._build_exclusion_matrix(direction=direction)
            r[mask] = 0

        shift = np.asarray(np.where(np.abs(r) == np.abs(r).max()), dtype=float).flatten()
        shift[shift > self.shape / 2] -= self.shape[shift > self.shape / 2]

        if k == 1:
            pass
        else:
            # find 1.5 x 1.5 pixel neighborhood (in units of original pixels)
            range_k = np.arange(1.5 * k)
            offset = np.fix(1.5 * k / 2.0) - shift * k

            # calculate upsampling kernels for neighborhood
            kernel_x = np.exp((-1j * 2 * np.pi / (self.shape[1] * k)) * (np.fft.ifftshift(self.range_x)[:, None] - (self.shape[1] / 2)).dot(range_k[None, :] - offset[1]))
            kernel_y = np.exp((-1j * 2 * np.pi / (self.shape[0] * k)) * (range_k[:, None] - offset[0]).dot(np.fft.ifftshift(self.range_y)[None, :] - (self.shape[0] / 2)))

            # (1.5k, nx) . (nx, ny) . (ny, 1.5k)
            kdft = kernel_y.dot(np.conjugate(R)).dot(kernel_x) / (np.multiply(*self.shape) * k**2)

            shift_k = np.asarray(np.where(np.abs(kdft) == np.abs(kdft).max()), dtype=float).flatten()
            shift_k -= np.fix(1.5 * k / 2.0)
            shift = shift + shift_k / k

        return shift




