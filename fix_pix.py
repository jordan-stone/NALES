import numpy as np

def get_bpm():
    import os
    from . import jFits
    h, d = jFits.get_fits_array(os.path.dirname(__file__)+'/bpm.fits')
    return d

def bfixpix(data, badmask, n=4, retdat=False):
    """
    Taken from Ian Crossfields LPL website, modified for bug...
    
    Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest four good neighboring pixels.

    :INPUTS:
    data : numpy array (two-dimensional)

    badmask : numpy array (same shape as data), nonzero values are "bad"

    :OPTIONAL_INPUTS:
    n : int
    number of nearby, good pixels to average over

    retdat : bool
    If True, return an array instead of replacing-in-place and do
    _not_ modify input array `data`.  This is always True if a 1D
    array is input!

    :RETURNS: 
    another numpy array (if retdat is True)

    :TO_DO:
    Implement new approach of Popowicz+2013 (http://arxiv.org/abs/1309.4224)
    """
    # 2010-09-02 11:40 IJC: Created
    # 2012-04-05 14:12 IJMC: Added retdat option
    # 2012-04-06 18:51 IJMC: Added a kludgey way to work for 1D inputs
    # 2012-08-09 11:39 IJMC: Now the 'n' option actually works.


    #if data.ndim==1:
    #    data = np.tile(data, (3,1))
    #    badmask = np.tile(badmask, (3,1))
    #    ret = bfixpix(data, badmask, n=2, retdat=True)
    #    return ret[1]


    nx, ny = data.shape

    badx, bady = np.nonzero(badmask)
    nbad = len(badx)
    print('bfixpix: %i bad'%nbad)

    if retdat:
        data = np.array(data, copy=True)

    print('looping over %i pixels'%nbad)
    for ii in range(nbad):
        thisloc = badx[ii], bady[ii]
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels<n:
            rad += 1
            xmin = max(0, badx[ii]-rad)
            xmax = min(nx, badx[ii]+rad)
            ymin = max(0, bady[ii]-rad)
            ymax = min(ny, bady[ii]+rad)
            x = np.arange(nx)[xmin:xmax+1]
            y = np.arange(ny)[ymin:ymax+1]
            yy,xx = np.meshgrid(y,x)
            #print ii, rad, xmin, xmax, ymin, ymax, badmask.shape

            rr = abs((xx-badx[ii]) + 1j*(yy-bady[ii])) * (1. - badmask[xmin:xmax+1,ymin:ymax+1])#BUG in Ian's code? fixed
            numNearbyGoodPixels = (rr>0).sum()

        closestDistances = np.unique(np.sort(rr[rr>0])[0:n])
        numDistances = len(closestDistances)
        localSum = 0.
        localDenominator = 0.
        for jj in range(numDistances):
            localSum += data[xmin:xmax+1,ymin:ymax+1][rr==closestDistances[jj]].sum()
            localDenominator += (rr==closestDistances[jj]).sum()

        #print badx[ii], bady[ii], 1.0 * localSum / localDenominator, data[xmin:xmax+1,ymin:ymax+1]
        data[badx[ii], bady[ii]] = 1.0 * localSum / localDenominator

    if retdat:
        ret = data
    else:
        ret = None

    return ret

def find_neighbors(badmask, n=4, retdat=False):
    """
    The longest step in bfixpix is looping over the bad
    pixels to find the best neighbors for correction. If the 
    Pixel mask doesn't change, the good neighbors don't change either.
    Use this function to identify the neigbors. The output of this
    function should be used as the input of fixpix_precomputed_neighbors
    """
    nx, ny = badmask.shape
    badx, bady = np.nonzero(badmask)
    nbad = len(badx)
    print('bfixpix: %i bad'%nbad)

    print('looping over %i pixels'%nbad)
    bad_and_neighbors=[]
    for ii in range(nbad):
        thisloc = badx[ii], bady[ii]
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels<n:
            rad += 1
            xmin = max(0, badx[ii]-rad)
            xmax = min(nx, badx[ii]+rad)
            ymin = max(0, bady[ii]-rad)
            ymax = min(ny, bady[ii]+rad)
            x = np.arange(nx)[xmin:xmax+1]
            y = np.arange(ny)[ymin:ymax+1]
            yy,xx = np.meshgrid(y,x)
            #print ii, rad, xmin, xmax, ymin, ymax, badmask.shape

            rr = abs((xx-badx[ii]) + 1j*(yy-bady[ii])) * (1. - badmask[xmin:xmax+1,ymin:ymax+1])#BUG in Ian's code? fixed
            numNearbyGoodPixels = (rr>0).sum()

        closestDistances = np.unique(np.sort(rr[rr>0])[0:n])
        numDistances = len(closestDistances)
        pixToTake=[]
        for jj in range(numDistances):
            pixToTake.append(rr==closestDistances[jj])

        bad_and_neighbors.append((badx[ii],bady[ii],xmin,xmax,ymin,ymax,np.any(pixToTake,axis=0)))
    return bad_and_neighbors

def correct_with_precomputed_neighbors(data,bad_and_neighbors):
    '''use this function to fix bad pixels in an image, after
    using fixpix_find_neighbors on the bad pixel mask to
    identify good neighbors. If fixing pixels in 5000 images with
    the same bad pixel mask, this function saves tons of time because 
    the neighbors only have to be found once.'''
    for badx, bady, xmin, xmax, ymin, ymax, take in bad_and_neighbors:
        mean_good=1.*data[xmin:xmax+1,ymin:ymax+1][take].sum()/take.sum()
        data[badx,bady]=mean_good
