import gc
import os
import datetime
import fnmatch
import pickle
from multiprocessing import Pool

import numpy as np

from astropy.io import fits

from ALES.rotate import rot
from . import jFits

class PCA:
    def __init__(self, data, parallactic_angles, x_cen=150, y_cen=150, reverse_rotate=False):
        '''data needs to by shape (xdim,ydim,n_images). that's different than before'''
        self.data = data
        self.n_images = data.shape[2]
        self.X_DIM = data.shape[0]
        self.Y_DIM = data.shape[1]
        self.x_cen = x_cen
        self.y_cen = y_cen
        self.parallactic_angles = parallactic_angles
        if reverse_rotate:
            self.parallactic_angles *= -1.
        y,x = np.indices((self.Y_DIM, self.X_DIM))
        self.radius=np.abs((x-x_cen)+1j*(y-y_cen))
        self.radius_1d=self.radius.flatten()

    def __call__(self, r_n_pca):
        ''' r 6 or greater'''
        r, n_pca=r_n_pca
        r_subt_in = r
        r_subt_out = r+1.
        r_opt_in = max(r_subt_in - 2., 1)
        r_opt_out = r_subt_in +3.

        #////////////////////////////////////////////////////////
        #construct a masked region
        #////////////////////////////////////////////////////////
        opt_mask=np.logical_and(self.radius>r_opt_in,self.radius<=r_opt_out)#bool#
        n_pix_opt_masked=np.sum(opt_mask.astype(int))

        sub_mask = np.logical_and(self.radius>r_subt_in, self.radius<=r_subt_out)#bool#
        n_pix_sub_masked = np.sum(sub_mask.astype(int))

        #////////////////////////////////////////////////////////
        #make matrix of non-opt_masked pixels for PCA
        #////////////////////////////////////////////////////////
        A = self.data[opt_mask,:]#[xdim, ydim, nimages] converted to [opt_mask.sum(),nimages]

       #////////////////////////////////////////////////////////
       #Find the principal components (eigenvectors)
       #////////////////////////////////////////////////////////
       #linear algebra described on the wiki page for eigenfaces
       #https://en.wikipedia.org/wiki/Eigenface#Computing_the_eigenvectors
        A -= np.median(A, axis=1)[:,None] #subtract off the mean image
        A -= np.mean(A, axis=0)[None,:] #subtract off the mean image

        U1, S1, V1 = np.linalg.svd(A) 
        eigenvalues = np.square(S1)
        eigenvectors = V1.T
        #AT_A=np.dot(A.T,A)
        #eigenvalues, eigenvectors = np.linalg.eig(AT_A)
        idx=np.argsort(eigenvalues)[::-1]#python doesn't sort like idl...
        eigenvectors=eigenvectors[:,idx]
        eigenvectors=np.dot(A,eigenvectors)
        #normalize the eigenvectors
        eigenvectors /= (np.sum(eigenvectors**2.,axis=0)**0.5)

       #////////////////////////////////////////////////////////
       #Fit the individual images and reform to 2d
       #////////////////////////////////////////////////////////
        PCA_images_1d = np.zeros_like(A)
        for k in range(n_pca):
            coefficients = np.dot(eigenvectors[:,k],A)
            PCA_images_1d += (eigenvectors[:,k])[:,None] * coefficients

        PCA_subtracted_images = A - PCA_images_1d
        output = np.zeros_like(self.data)
        output[opt_mask,:] = PCA_subtracted_images
        output *= sub_mask[:,:,None]
        for image, q in enumerate(self.parallactic_angles):
            output[:,:,image] = rot(output[:,:,image], q, (self.x_cen, self.y_cen))
        return np.median(output, axis=2)

def parallel_annular_PCA(data,parallactic_angles,x_cen,y_cen,n_PCA,
                         radii=list(range(3,44)),
                         reverse_rotate=False,
                         ncpu=None):
    '''data is shape nims, nwaves,ny,nx
    two rot90's flip the upside down image to be right-side up'''
    nims, nwaves, ny, nx = data.shape
    out = np.zeros((nwaves,ny,nx),dtype=float)
    for lam_num in range(nwaves):
        data_lam = data[:,lam_num,:,:]
        print('wave_slice: ',lam_num)
        do_PCA = PCA(np.rollaxis(data_lam,0,3), parallactic_angles, x_cen=x_cen, y_cen=y_cen,reverse_rotate=reverse_rotate)
        pool=Pool(ncpu)
        annuli = pool.map(do_PCA,list(zip(radii,np.zeros(len(radii),dtype=int)+n_PCA)))
        combined_median=np.sum(annuli,axis=0)
        out[lam_num,:,:] = np.rot90(np.rot90(combined_median))
    return out
