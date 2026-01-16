from scipy.ndimage import shift, zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import numpy as np
from . import jFits
from .wavecals import get_spots_grid, spoof_spots, w_func, onedmoments
from .climb import climb, climb1d
from .rotate import find_rotation_angle, rotate, rot
from .fix_pix import bfixpix


class Cubifier:
    def __init__(self, wave_cal_ims, sky_im, sky_var,
                 make_slice_bottoms=(35, 35), gridplot=False,
                 start_offsets=(200, 100), dims=(67,64),
                 zoom_factor=1,
                 spoof_nb29=True,
                 spoof_nb33=False,
                 best_spots_slice=(slice(920, 1020), slice(900, 1000)),
                 use_middle_weights=True,
                 fake_weights=False,
                 trim_crosstalk=True):
        """input wavecal frames and a median sky frame, return an object whose
        __call__ method will cubify any image from the associated data set.
        INPUTS:
        wave_cal_ims: dictionary, with keys ['nb29','nb33','nb35','nb39'] and
                      values the corresponding dark-subtracted and stacked
                      images of the spots.
        sky_im: 2d-array, the median sky image, dark subtracted
        sky_var: 2d-array, image (skyvar**2 + darkvar**2) for the dark
            subtracted image.
        make_slice_bottoms: a len=2 tuple, giving the extent below (left of)
            the reference point that the slice for trace will have. Important
            for identifying the position of the reference point within the
                stamps (trace positions are determined by the position of
                a reference point, usually the nb39 spots)
        gridplot: bool, if True plot the automatically determined reference
                positions on top of the nb39 image
        start_offsets: len=2 tuple, A position very near the lower left
                       reference spot this is important for finding all the
                       other spots.
        zoom_factor: a number, setting the factor to oversample the wavelength
                     direction of the traces
        """

        #############################
        # Assign inputs and constants
        ############################
        self.wave_cal_ims = wave_cal_ims
        self.make_slice_bottoms = make_slice_bottoms
        self.ref_to_top = 65
        self.ref_to_right = 20
        self.zoom_factor = zoom_factor
        self.sky_im = sky_im
        self.sky_var = sky_var
        self.use_middle_weights = use_middle_weights
        self.fake_weights = fake_weights
        self.trim_crosstalk = trim_crosstalk
        self.slices = None
        self.wavecals = None
        self.dims = dims  # output spatial dimensions
        # Below: from the reference location to the corner, this is the radius
        # that needs to not get cropped with rot...
        # 60 and 20 ref_to_[top|right] from make_slices...
        self.sz = 2*int(np.ceil((60**2 + 20**2)**0.5))

        if spoof_nb29:
            nb29_new = spoof_spots(self.wave_cal_ims['nb29'],
                                   self.wave_cal_ims['nb39'],
                                   best_spots_slice=best_spots_slice)
            self.wave_cal_ims['nb29_old'] = self.wave_cal_ims['nb29'].copy()
            self.wave_cal_ims['nb29'] = nb29_new

        if spoof_nb33:
            nb33_new = spoof_spots(self.wave_cal_ims['nb33'],
                                   self.wave_cal_ims['nb39'],
                                   best_spots_slice=best_spots_slice)
            self.wave_cal_ims['nb33_old'] = self.wave_cal_ims['nb33'].copy()
            self.wave_cal_ims['nb33'] = nb33_new

        self.nb39_spots = get_spots_grid(wave_cal_ims['nb39'],
                                         start_offsets=start_offsets,
                                         plot=gridplot,
                                         dims=self.dims)
        #############################
        # initialize cubifier
        ############################
        self._make_slices()
        self._make_trace_mask()
        self._find_rotation_angles()
        self._make_weights()
        self._make_cal_traces()
        self._wavecal()

    def _make_slices(self):
        """cycle through spot positions (found using .wavecals.get_spots_grid)
        and return a dictionary of reference points and slices.  The keys of
        the dictionary can be used as output x,y positions in a cube
        """
        print('making slices for each trace')
        self.slices = {}
        self.refs = {}
        for ii, row in enumerate(list(zip(*self.nb39_spots))):
            for ll, point in enumerate(list(zip(*row))):
                self.refs[(ii, ll)] = point
                p0 = int(np.rint(point[0]))
                p1 = int(np.rint(point[1]))
                self.slices[(ii, ll)] = (slice(p0-self.make_slice_bottoms[0],
                                               p0+self.ref_to_top),
                                         slice(p1-self.make_slice_bottoms[1],
                                               p1+self.ref_to_right))

    def _make_trace_mask(self, slope=-3, offsets=(-19, 21)):
        """use knowledge of where the trace of interest is positioned,
        and it's range of rotation angles to create a mask so that other
        traces in the stamp can be ignored
        """
        refy, refx = self.make_slice_bottoms
        # 60 and 20 from make_slices
        self.mask = np.ones((refy+self.ref_to_top, refx+self.ref_to_right), dtype=int)
        indices = np.indices(self.mask.shape)
        self.mask[indices[0] > slope*(indices[1]-refx)+refy+offsets[1]] = 0
        self.mask[indices[0] < slope*(indices[1]-refx)+refy+offsets[0]] = 0

    def _find_rotation_angles(self, slope=-3, flip=True, leftof=5, rightof=9):
        self.leftof=leftof
        self.rightof=rightof
        if self.mask is None:
            self.make_trace_mask()
        if flip:
            extra = 180.
        else:
            extra = 0.
        self.rot_angles = np.empty(self.dims)  # dims

        refy, refx = self.make_slice_bottoms
        for spaxel in list(self.slices.keys()):
            stamp = self.mask*self.sky_im[self.slices[spaxel]]
            self.rot_angles[spaxel] = find_rotation_angle(
                                                stamp,
                                                self.make_slice_bottoms[::-1],
                                                slope=slope,
                                                leftof=leftof,
                                                rightof=rightof) + extra

    def _weights_im(self, spaxel, smooth_kern_sky=1.4, smooth_kern_var=2,
                    plot=False, flip=True, use_middle_weights=False,
                    fake_weights=False, trim_crosstalk=True):

        stamp = gaussian_filter(self.sky_im[self.slices[spaxel]],
                                smooth_kern_sky)
        vstamp = gaussian_filter(self.sky_var[self.slices[spaxel]],
                                 smooth_kern_var)
        rstamp = rot(stamp, self.rot_angles[spaxel],
                     self.make_slice_bottoms[::-1],
                     out_size=(self.sz, self.sz))
        rvstamp = rot(vstamp, self.rot_angles[spaxel],
                      self.make_slice_bottoms[::-1],
                      out_size=(self.sz, self.sz))
        # usually psf squared but then divided by later,
        # so using just one power here
        if fake_weights:
            w0 = rstamp #p**2/sig**2 goes like p.
        else:
            w0 = rstamp / rvstamp
        mask = np.ones_like(w0)
        mask[:, :(self.sz//2)-4] = 0
        mask[:, (self.sz//2)+4:] = 0
        if trim_crosstalk:
            mask[90:, :self.sz//2] = 0 #90 by hand, where crosstalk is bad...
        if flip:
            mask = np.rot90(np.rot90(mask))
        w = mask*w0
        w[np.isnan(w)] = 0
        ww = zoom(w, (self.zoom_factor, 1))
        www = ww/np.sum(ww, axis=1)[:, None]
        www[np.isnan(www)] = 0
        if use_middle_weights:
            prof = np.mean(www[(self.sz//2)-8:(self.sz//2)+8,:],axis=0)
            prof /= prof.sum()
            www = mask*prof[None,:]
            www /= www.sum(1)
        if plot:
            d = jFits.jInteractive_Display(self.sky_im[self.slices[spaxel]])
            d.a.set_title('sky')
            d = jFits.jInteractive_Display(stamp)
            d.a.set_title('smooth sky')
            d = jFits.jInteractive_Display(vstamp)
            d.a.set_title('smooth sky variance')
            d = jFits.jInteractive_Display(rstamp)
            d.a.set_title('smooth rot sky')
            d = jFits.jInteractive_Display(rvstamp)
            d.a.set_title('smooth rot sky variance')
            d = jFits.jInteractive_Display(w0)
            d.a.set_title('weights')
            d = jFits.jInteractive_Display(www)
            d.a.set_title('trimmed_weights weights')
            d.a.contour(w0, colors='w')
        return www

    def _make_weights(self, smooth_kern_sky=1.4, smooth_kern_var=2):
        """divide the variance array into the sky array to make extraction
        weights
        """
        self.weights = {}
        for spaxel in list(self.slices.keys()):
            self.weights[spaxel] = self._weights_im(
                                      spaxel,
                                      smooth_kern_sky=smooth_kern_sky,
                                      smooth_kern_var=smooth_kern_var,
                                      use_middle_weights=self.use_middle_weights,
                                      fake_weights=self.fake_weights,
                                      trim_crosstalk=self.trim_crosstalk)

    def _extract(self, arr, spaxel):
        stamp0 = rot(arr[self.slices[spaxel]],
                     self.rot_angles[spaxel],
                     self.make_slice_bottoms[::-1],
                     out_size=(self.sz, self.sz))
        stamp0[np.isnan(stamp0)] = 0
        stamp = zoom(stamp0, (self.zoom_factor, 1))
        spec = np.sum(self.weights[spaxel]*stamp, axis=1) \
               / np.sum(self.weights[spaxel], axis=1)
        return spec

    def _make_cal_traces(self):
        cal_cubes = {}
        for key in self.wave_cal_ims.keys():
            print('making cal cube: {}'.format(key))
            arr = self.wave_cal_ims[key]
            out_arr = np.empty((self.sz*self.zoom_factor,
                                self.dims[0],
                                self.dims[1]))
            for spaxel in list(self.slices.keys()):
                spec = self._extract(arr, spaxel)
                out_arr[:, spaxel[0], spaxel[1]] = spec
            cal_cubes[key] = out_arr

        print('making cal cube: sky')
        arr = self.sky_im
        out_arr = np.empty((self.sz*self.zoom_factor,
                            self.dims[0],
                            self.dims[1]))
        for spaxel in list(self.slices.keys()):
            spec = self._extract(arr, spaxel)
            out_arr[:, spaxel[0], spaxel[1]] = spec
        cal_cubes['sky'] = out_arr
        self.cal_cubes = cal_cubes

    def _wavecal_coeffs(self):
        """run wavelength calibration"""
        self.wsoln_coeffs_nans = np.empty((3, self.dims[0], self.dims[1]))
        self.w_pixels = np.empty((4, self.dims[0], self.dims[1]))
        self.wsoln_coeffs = np.empty((3, self.dims[0], self.dims[1]))
        self.reuse = 0

        for spaxel in self.slices.keys():
            nb29_slices = (slice(1, 11),)+spaxel
            px29 = onedmoments(self.cal_cubes['nb29'][nb29_slices])[1] + 1
            nb33_slices = (slice(25, 35),)+spaxel
            px33 = onedmoments(self.cal_cubes['nb33'][nb33_slices])[1] + 25
            nb35_slices = (slice(35, 45),)+spaxel
            px35 = onedmoments(self.cal_cubes['nb35'][nb35_slices])[1] + 35
            nb39_slices = (slice(59, 69),)+spaxel
            px39 = onedmoments(self.cal_cubes['nb39'][nb39_slices])[1] + 59
            self.w_pixels[:, spaxel[0], spaxel[1]] = [px29, px33, px35, px39]

            try:
                popt, _ = curve_fit(w_func,
                                    [px29, px33, px35, px39],
                                    [2.897, 3.360, 3.539, 3.874],
                                    bounds=([-np.inf, -np.inf, -np.inf],
                                            [np.inf, 0, np.inf]))
                self.wsoln_coeffs_nans[:, spaxel[0], spaxel[1]] = popt
            except (ValueError, RuntimeError):
                self.wsoln_coeffs_nans[:, spaxel[0], spaxel[1]] = \
                                        np.array((np.nan, np.nan, np.nan))

        for ii in range(3):
            self.wsoln_coeffs[ii] = bfixpix(
                                        self.wsoln_coeffs_nans[ii],
                                        np.isnan(self.wsoln_coeffs_nans[ii]),
                                        retdat=True)

    def _wavecal(self):
        self.wavecals = {}
        self._wavecal_coeffs()
        for spaxel in self.slices.keys():
            out = w_func(
                np.arange(len(self.cal_cubes['sky'][:, spaxel[0], spaxel[1]])),
                *self.wsoln_coeffs[:, spaxel[0], spaxel[1]])
            self.wavecals[spaxel] = out

    def __call__(self, arr, wave_anchor_spaxel=(30, 30)):
        out_inds = np.logical_and(self.wavecals[wave_anchor_spaxel]>2.7,
                                self.wavecals[wave_anchor_spaxel]<4.3)
        out_arr = np.empty(
                        (np.sum(out_inds), self.dims[0], self.dims[1]))
        for spaxel in list(self.slices.keys()):
            #stamp0 = rot(arr[self.slices[spaxel]],
            #             self.rot_angles[spaxel],
            #             self.make_slice_bottoms[::-1],
            #             out_size=(self.sz, self.sz))
            #stamp0[np.isnan(stamp0)] = 0
            #stamp = zoom(stamp0, (self.zoom_factor, 1))
            spec = self._extract(arr, spaxel)
            interpspec = interp1d(self.wavecals[spaxel],
                                  spec,
                                  kind='cubic',
                                  fill_value='extrapolate')
            out_arr[:, spaxel[0], spaxel[1]] = interpspec(
                                             self.wavecals[wave_anchor_spaxel][out_inds])
        return out_arr, self.wavecals[wave_anchor_spaxel][out_inds]

    def inspect_wavecal(self, spaxel):
        f = jFits.mpl.figure()
        a = f.add_subplot(111)
        a.plot(self.cal_cubes['nb29'][1:11, spaxel[0], spaxel[1]],
               color='c')
        a.plot(self.cal_cubes['nb33'][25:35, spaxel[0], spaxel[1]],
               color='g')
        a.plot(self.cal_cubes['nb35'][35:45, spaxel[0], spaxel[1]],
               color='orange')
        a.plot(self.cal_cubes['nb39'][59:69, spaxel[0], spaxel[1]],
               color='r')

        nb29_slices = (slice(1, 11),)+spaxel
        px29 = onedmoments(self.cal_cubes['nb29'][nb29_slices])[1]
        a.axvline(px29, color='c')

        nb33_slices = (slice(25, 35),) + spaxel
        px33 = onedmoments(self.cal_cubes['nb33'][nb33_slices])[1]
        a.axvline(px33, color='g')

        nb35_slices = (slice(35, 45),) + spaxel
        px35 = onedmoments(self.cal_cubes['nb35'][nb35_slices])[1]
        a.axvline(px35, color='orange')

        nb39_slices = (slice(59, 69),) + spaxel
        px39 = onedmoments(self.cal_cubes['nb39'][nb39_slices])[1]
        a.axvline(px39, color='r')

        f2 = jFits.mpl.figure()
        a2 = f2.add_subplot(111)
        a2.plot(self.w_pixels[:, spaxel[0], spaxel[1]],
                [2.897, 3.360, 3.539, 3.874],
                'ko')
        a2.plot(np.arange(len(self.cal_cubes['sky'][:, spaxel[0], spaxel[1]])),
                self.wavecals[spaxel])
        jFits.mpl.show()

    def inspect_wavecal_nbs(self, spaxel,
                            bands=['nb39', 'nb35', 'nb33', 'nb29']):
        colors = {'nb39': 'r', 'nb35': 'orange', 'nb33': 'g', 'nb29': 'c'}
        slc = self.slices[spaxel]
        disp = jFits.jInteractive_Display(self.sky_im[slc])
        for wl in bands:
            disp.a.contour(self.mask*self.wave_cal_ims[wl][slc],
                           colors=colors[wl])
        jFits.mpl.show()

    def inspect_rotation(self, spaxel, flip=True):
        refy, refx = self.make_slice_bottoms
        stamp = self.mask * self.sky_im[self.slices[spaxel]]
        if flip:
            extra = 180.
        else:
            extra = 0.
        a = find_rotation_angle(stamp,
                                self.make_slice_bottoms[::-1],
                                plot=True,
                                leftof=self.leftof,
                                rightof=self.rightof)

        a += extra
        jFits.jInteractive_Display(rotate(stamp, a))

    def inspect_weights(self, spaxel, smooth_kern_sky=1.4, smooth_kern_var=2):
        __ = self._weights_im(spaxel,
                              smooth_kern_sky=smooth_kern_sky,
                              smooth_kern_var=smooth_kern_var,
                              use_middle_weights=self.use_middle_weights,
                              fake_weights=self.fake_weights,
                              trim_crosstalk=self.trim_crosstalk,
                              plot=True)
