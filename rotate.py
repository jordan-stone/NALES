import matplotlib.pyplot as mpl
import numpy as np
from scipy.ndimage import rotate, map_coordinates
from gaussfitter import onedgaussian, onedgaussfit, multigaussfit
from . import jFits
from .climb import climb1d
    
def find_rotation_angle(im, ref_pos=None, plot=False, slope=-3, leftof=20, rightof=20,debug=False):
    #im is like sky_im[slices[spaxel]]
    x1 = []
    y1 = []
    x1.append(ref_pos[0])
    y1.append(climb1d(ref_pos[1],im[:,x1[-1]]))
    #loop forward from fiducial
    for xx in range(ref_pos[0]+1,ref_pos[0]+rightof):
        x1.append(xx)
        y1.append(climb1d(y1[-1]+slope,im[:,xx]))

    x2 = []
    y2 = []
    x2.append(ref_pos[0])
    y2.append(y1[0])
    #loop backward from fiducial
    for xx in range(ref_pos[0]-leftof,ref_pos[0])[::-1]:
        x2.append(xx)
        y2.append(climb1d(y2[-1]-slope,im[:,xx]))

    xs = x2[::-1]+x1
    ys = y2[::-1]+y1

    #crop
    pcs = np.polyfit(xs,ys,1)
    angle = 90 - (180/np.pi)*np.arctan(-1*pcs[0])
    if plot:
        disp = jFits.jInteractive_Display(im)
        disp.a.plot(xs,ys,'ro')
        disp.a.plot(xs,np.polyval(pcs,xs),'w-')
    if debug:
        disp = jFits.jInteractive_Display(im)
        disp.a.plot(xs,ys,'ro')
        disp.a.plot(xs,np.polyval(pcs,xs),'w-')
        mpl.show()
        1/0

    return angle

def rot(im, angle, axis, order=3, pivot=False, out_size=None):
    '''rotate an image clockwise by angle [degrees] about axis.
    if pivot is true the image will pivot about the axis. otherwise
    the axis will be centered in the output image''' 
    angle*=np.pi/180.#convert to radians
    if out_size is None:
        y, x = np.indices((2*max(im.shape),2*max(im.shape)))
        y-=(int(max(im.shape)/2))
        x-=(int(max(im.shape)/2))
    else:
        y, x = np.indices(out_size)
        y-=int((out_size[0]-im.shape[0])/2)
        x-=int((out_size[1]-im.shape[1])/2)
    
    #calculate how the axis moves when pivoting from bottom left corner
    theta_axis = np.arctan2(axis[1],axis[0])
    r_axis = np.abs(axis[0]+1j*axis[1])
    yoffset = r_axis*np.sin(theta_axis) - r_axis*np.sin(theta_axis-angle)
    xoffset = r_axis*np.cos(theta_axis) - r_axis*np.cos(theta_axis-angle)

    #put the axis in the middle? 
    ycenter_offset = (1-pivot) * ((im.shape[0]/2.)-axis[1])#pivot is a bool (i.e. 0 or 1)
    xcenter_offset = (1-pivot) * ((im.shape[1]/2.)-axis[0])
    yoffset += ycenter_offset
    xoffset += xcenter_offset

    #make rotation matrix elements
    ct = np.cos(angle)
    st = np.sin(angle)

    #do the rotation 
    new_x = (ct*(x-xoffset) - st*(y-yoffset)) 
    new_y = (st*(x-xoffset) + ct*(y-yoffset))
    return map_coordinates(im, [new_y, new_x], order=order, cval=np.nan)

def rot_and_interp(im, angle, axis, ys, order=3):
    '''Experimental''' 
    angle*=np.pi/180.#convert to radians
    y = np.empty((len(ys),20))+ys[:,None]
    x = np.empty((len(ys),20))+np.arange(20)[None,:]
    x-=(int(20/2))
    
    #calculate how the axis moves when pivoting from bottom left corner
    theta_axis = np.arctan2(axis[1],axis[0])
    r_axis = np.abs(axis[0]+1j*axis[1])
    yoffset = r_axis*np.sin(theta_axis) - r_axis*np.sin(theta_axis-angle)
    xoffset = r_axis*np.cos(theta_axis) - r_axis*np.cos(theta_axis-angle)

    #make rotation matrix elements
    ct = np.cos(angle)
    st = np.sin(angle)

    #do the rotation 
    new_x = (ct*(x-xoffset) - st*(y-yoffset)) 
    new_y = (st*(x-xoffset) + ct*(y-yoffset))
    return map_coordinates(im, [new_y, new_x], order=order, cval=np.nan)

