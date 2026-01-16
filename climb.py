import numpy as np

def climb(arr,start,verbose=False):
    start = [int(np.rint(s)) for s in start]
    if verbose:
        print('start climb***************************************')
        print(start)
    this_val = arr[start[0], start[1]]
    neighbor_vals = np.r_[arr[start[0]-1, start[1]-2:start[1]+3],
                          arr[start[0]+1, start[1]-2:start[1]+3],
                          arr[start[0], start[1]-2:start[1]],
                          arr[start[0], start[1]+1:start[1]+3],
                          arr[start[0]-2, start[1]-2:start[1]+3],
                          arr[start[0]+2, start[1]-2:start[1]+3]]
    while np.any(neighbor_vals > this_val):
        stamp = arr[start[0]-2:start[0]+3, start[1]-2:start[1]+3]
        maxp = np.where((stamp-this_val)==np.max(stamp-this_val))
        new0 = start[0] + (maxp[0][0]-2)
        new1 = start[1] + (maxp[1][0]-2)
        start = (new0, new1)
        if verbose:
            print(start)

        this_val = arr[start[0], start[1]]
        neighbor_vals = np.r_[arr[start[0]-1,start[1]-2:start[1]+3],
                              arr[start[0]+1,start[1]-2:start[1]+3],
                              arr[start[0],start[1]-2:start[1]],
                              arr[start[0],start[1]+1:start[1]+3],
                              arr[start[0]-2,start[1]-2:start[1]+3],
                              arr[start[0]+2,start[1]-2:start[1]+3]]
    if verbose:
        print('end climb***************************************')
    return start

def climb1d(y,av=None):
    '''climb uphill along that column from a starting location
    Inputs:
    y          - [float] the initial y coordinate of the starting location
    av         - [1d array] the image column
    Returns:
    y_apex     - [int] the local maximum of the column
    closest to the input location
    '''
    inp=y
    y0=int(y)
    y=int(y)
    try:
        up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
        while up!=1 and y > 2 and y < len(av)-2:
            y+=(up-1)
            up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
        return y
    except ValueError:
        print('climb wandered off...\n Returning input')
        return y0

