from scipy import ndimage
from gaussfitter import gaussfit
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from astropy.io import fits as pyfits
import warnings
import os
import fnmatch

warnings.simplefilter("error", FutureWarning)

write=pyfits.writeto

def jslice(center, width):
    return slice(center-width,center+width+1)

def read_wcs_keys(fits_head):
    wcs_keys=[fits_head['CRPIX1'], 
              fits_head['CRPIX2'], 
              fits_head['CRVAL1'],
              fits_head['CRVAL2'],
              fits_head['CDELT1'],
              fits_head['CDELT2']]
    return wcs_keys

def get_pix2sky_wcs(wcs_keys):
    '''wcs_keys a list returned from jFits.read_wcs_keys
    The fits standard is not zero-based. Returns 2 functions'''
    transform1=lambda x:((x-wcs_keys[0]-1)*wcs_keys[4])+wcs_keys[2]
    transform2=lambda y:((y-wcs_keys[1]-1)*wcs_keys[5])+wcs_keys[3]
    return transform1, transform2

def get_sky2pix_wcs(wcs_keys):
    '''wcs_keys a list returned from jFits.read_wcs_keys
    The fits standard is not zero-based. Returns 2 functions'''
    transform1=lambda ra: ((ra-wcs_keys[2])*(1/wcs_keys[4]))+wcs_keys[0]-1
    transform2=lambda dec: ((dec-wcs_keys[3])*(1/wcs_keys[5]))+wcs_keys[1]-1
    return transform1,transform2

def make_format_coord_func(arr,scale=(1.,1.),xoffset=0,yoffset=0):
#scale is important here, since x and y are given in data coordinates, not array coordinates, so
#if for example the extent keyword is used in imshow, the x and y will need to be scaled to the 
#correct indices for indexing the array.
    def f(x,y):
        xout=np.rint(x)
        yout=np.rint(y)
        zout=arr[int(max(min(np.rint(scale[0]*y+yoffset),arr.shape[0]-1),0)),\
                 int(max(min(np.rint(scale[1]*x+xoffset),arr.shape[1]-1),0))]
        if np.isnan(zout):
            return "x=%i, y=%i, z=NAN" % (xout,yout)
        else:
            return "x=%f, y=%f, z=%05.4e" % (xout,yout,zout)
    return f

def get_fits_array(filename,quiet=True,fix=False):
    hdulist=pyfits.open(filename)
    if fix:
        hdulist.verify('silentfix')
    if not quiet:
        print(hdulist.info())
    if len(hdulist) > 1:
        return [hdu.header for hdu in hdulist], [hdu.data for hdu in hdulist]
    else:
        return hdulist[0].header, hdulist[0].data

def get_nods(basename,imnums):
    ds=[]
    for ii in imnums:
        fn=fnmatch.filter(os.listdir('.'),basename+('%05i'%ii)+'*')[0]
        h,d=get_fits_array(fn)
        ds.append(d)
    return np.sum(ds,axis=0)

def get_nods_start_stop(basename,startStop):
    ds=[]
    imnums=list(range(startStop[0],startStop[1]+1))
    print('%i images'%len(imnums))
    for ii in imnums:
        h,d=get_fits_array(basename+('%05i'%ii)+'.fit')
        ds.append(d)
    return np.sum(ds,axis=0)

def safer_log(arr):
    positive=arr[arr>0]
    if len(positive) > 0:
        min_pos=np.min(arr[arr>0])
        min_replace=np.log(min_pos)
    else:
        min_replace=0
    return np.where(arr>0,np.log(arr),min_replace)

def jDisplay(arr,figure=None,figsize=(8,8),subplot=111,log=False,show=True,**imshowargs):
    if figure is None:
        f=mpl.figure(figsize=figsize)
    else:
        f=figure
        print('figure supplied, ignoring figsize...')
    a=f.add_subplot(subplot)
    if log:
        out_arr=safer_log(arr)
    else:
        out_arr=arr
    if 'cmap' not in list(imshowargs.keys()) or imshowargs['cmap']=='median':
        gd_vals=np.isfinite(out_arr)
        black=0.85*(np.median(out_arr[gd_vals])-out_arr[gd_vals].min())/\
              (out_arr[gd_vals].max()-out_arr[gd_vals].min())
        print(out_arr[gd_vals].min())
        print(out_arr[gd_vals].max())
        print("HELLO",black)
        cdict={ 'red'  :((0.0,0,0),(black,0,0),(1.0,1,1)),
                'blue' :((0.0,0,0),(black,0,0),(1.0,1,1)),
                'green':((0.0,0,0),(black,0,0),(1.0,1,1))}
        imshowargs['cmap']=matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict)
        imshowargs['cmap'].set_bad('b')
    a.imshow(out_arr,origin='lower',interpolation='Nearest',**imshowargs)
    a.format_coord=make_format_coord_func(arr)
    if show:
        mpl.show()
    return a
########################################################################################################


class jInteractive_Display:
    def __init__(self,arr,figure=None,show=True,**imshowargs):
        self.arr=np.ma.array(arr,mask=np.isnan(arr))
        self.f=mpl.figure(figsize=(10,10))
        self.a=self.f.add_axes([0.25,0.25,0.65,0.65])
        cmap=cm.viridis
        cmap.set_bad('k')
        self.a.imshow(self.arr,origin='lower',interpolation='Nearest',cmap=cmap,**imshowargs)
        
        data_ext=self.a.images[0].get_extent()
        data_extents=(data_ext[1]-data_ext[0],data_ext[3]-data_ext[2])[::-1]#y first in list
        arr_size=arr.shape
        scale_factors=list(map(lambda de,xy:xy/de,data_extents,arr_size))
        offsets=list(map(lambda e,s:s*e+0.5,[data_ext[0],data_ext[2]],scale_factors))
        self.a.format_coord=make_format_coord_func(self.arr,scale_factors,xoffset=-offsets[1],yoffset=-offsets[0])

        self.logarr=safer_log(arr)
        self.data_dict={'linear':self.arr,'log':self.logarr}

        self.min0,self.max0=self.a.images[0].get_clim()
        self.vmaxax=self.f.add_axes([0.25, 0.08, 0.65, 0.03])
        self.vminax=self.f.add_axes([0.25,0.05, 0.65, 0.03])

        self.svmax=Slider(self.vmaxax,'Vmax',self.arr.min(),self.arr.max(),valinit=self.max0)
        self.svmin=Slider(self.vminax,'Vmin',self.arr.min(),self.arr.max(),valinit=self.min0)
        self.svmax.slidermin=self.svmin
        self.svmin.slidermax=self.svmax
        self.svmax.on_changed(self.update)
        self.svmin.on_changed(self.update)

        self.rax = self.f.add_axes([0.025, 0.75, 0.15, 0.15])
        self.radio = RadioButtons(self.rax, 
                                  ('viridis', 'gray','spectral','spring','Oranges','gray_r'), 
                                  active=0)
        self.radio.on_clicked(self.colorfunc)

        self.rax2 = self.f.add_axes([0.025, 0.5, 0.15, 0.15])
        self.radio2 = RadioButtons(self.rax2, ('log', 'linear'), active=1)
        self.radio2.on_clicked(self.log_linear)

        self.rax3 = self.f.add_axes([0.025, 0.25, 0.15, 0.15])
        self.rax3.set_title('mask color')
        self.radio3 = RadioButtons(self.rax3, ('w', 'k'), active=1)
        self.radio3.on_clicked(self.change_mask_color)

        self.rax4 = self.f.add_axes([0.025, 0.05, 0.15, 0.15])
        self.rax4.set_title('centroid')
        self.radio4 = RadioButtons(self.rax4, ('No', 'Yes','Stop'), active=0)
        self.radio4.on_clicked(self.centroid)

        self.cax=self.f.add_axes([0.25,0.18, 0.65, 0.03])
        formatter=matplotlib.ticker.ScalarFormatter()
        formatter.set_powerlimits((0,0))
        self.cbar=self.f.colorbar(self.a.images[0],ax=self.a,cax=self.cax,orientation='horizontal',format=formatter)

    def update(self,val):
        mx = self.svmax.val
        mn = self.svmin.val
        self.a.images[0].set_clim(mn,mx)
        mpl.draw()

    def colorfunc(self,label):
        self.a.images[0].set_cmap(label)
        mpl.draw()

    def log_linear(self,label):
        data=self.data_dict[label]
        self.a.images[0].set_data(data)
        self.svmax.valmax=data.max()
        self.svmin.valmin=data.min()
        mpl.draw()

    def change_mask_color(self,label):
        print('clicked me')
        cm=self.a.images[0].get_cmap()
        cm.set_bad(label)
        self.a.images[0].set_cmap(cm)
        mpl.draw()

    def onclicked(self,event):
        inx=event.xdata
        iny=event.ydata
        int_inx=int(np.rint(inx))
        int_iny=int(np.rint(iny))
        print('clicked point: ', inx, iny)
        print('using integers: ', int_inx, int_iny)
        box_size=8
        fitarr=self.arr[int_iny-box_size:int_iny+box_size,int_inx-box_size:int_inx+box_size].copy()
        fitarr-=np.median(fitarr)
        fitarr = fitarr.astype(float)
        z=self.arr[int_iny,int_inx]
        sig=3.4
        out,model=gaussfit(fitarr,params=[0,z,float(box_size),float(box_size),sig,sig,0],circle=1,returnfitimage=True)
        self.centroid_params=out
        centroid_model=np.zeros_like(self.arr)
        centroid_model[int_iny-box_size:int_iny+box_size,int_inx-box_size:int_inx+box_size]=model
        self.centroid_model=centroid_model
        amp=centroid_model.max()
        self.a.contour(self.centroid_model,[0.1*amp,0.5*amp,0.9*amp],colors='w')
        self.a.set_title('x=%5.2f y=%5.2f sig=%0.2f offset=%0.2e amp=%0.2e'%(out[2]+int_inx-box_size,out[3]+int_iny-box_size,out[4],out[0],out[1]))
        print('fit sigma=',out[4])
        print(out)
        mpl.show()

    def centroid(self,label):
        if label=='Yes':
            self.cid=self.f.canvas.mpl_connect('button_press_event',self.onclicked)
            print('yes!')
        elif label=='Stop':
            self.a.collections=[]
            self.f.canvas.mpl_disconnect(self.cid)
            print('disconnected centroid clicker')
        elif label=='No':
            pass

def smooth(image,kernel):
    if image.shape != kernel.shape:
        kern=np.zeros_like(image)
        ishp=image.shape
        kshp=kernel.shape
        if kshp[0] % 2 != 0:
            klower0=(kshp[0]/2.+0.5)
            kupper0=(kshp[0]/2.-0.5)
        else:
            klower0=kshp[0]/2.
            kupper0=kshp[0]/2.
        if kshp[1] %2 != 0:
            klower1=(kshp[1]/2.+0.5)
            kupper1=(kshp[1]/2.-0.5)
        else:
            klower1=kshp[1]/2.
            kupper1=kshp[1]/2.
        kern[ishp[0]/2-klower0:ishp[0]/2+kupper0,ishp[1]/2-klower1:ishp[1]/2+kupper1]=kernel
        kernel=kern
    smoothed=np.fft.fftshift(np.fft.ifft2(np.fft.fft2(kernel) * np.fft.fft2(image)))
    return smoothed.real

def pad_convolve(im,kernel,deconvolve=False):
    if im.shape != kernel.shape:
        print("im and kernal must have the same shape")
        return
    pad_im=np.zeros([3*sh for sh in im.shape])
    pad_kn=np.zeros([3*sh for sh in kernel.shape])
    pad_im[im.shape[0]:2*im.shape[0],im.shape[1]:2*im.shape[1]]=im
    pad_kn[kernel.shape[0]:2*kernel.shape[0],kernel.shape[1]:2*kernel.shape[1]]=kernel
    if deconvolve:
        return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(pad_im)/np.fft.fft2(pad_kn)))
    else:
        return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(pad_im)*np.fft.fft2(pad_kn)))

def edge_detect(arr,log=False):
    if log:
        positive=arr[arr>0]
        if len (positive) > 0:
            min_pos=np.min(arr[arr>0])
            min_replace=np.log(min_pos)
        else:
            min_replace=0
        out_arr=np.where(arr>0,np.log(arr),min_replace)
    dx = ndimage.sobel(out_arr, 0)  # horizontal derivative
    dy = ndimage.sobel(out_arr, 1)  # vertical derivative
    return np.hypot(dx, dy) # magnitudejk
