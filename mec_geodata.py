# mec_geodata.py
# geographical utilities for the MEC algorithm

import numpy as np
from global_land_mask import globe

def land_matrix(min_lat, max_lat, min_lon, max_lon, resolution):
    """
    Return a boolean matrix L such that L[i, j] is true iff (lat(i, j), lon(i, j)) is a land point,
    where:
        lat(i, j) = min_lat + (i+0.5)*resolution
        lon(i, j) = min_lon + (j+0.5)*resolution
    The ranges are i \in [0,...,h] and j \in [0,...,k], where h (resp. k) is such that
    min_lat + (h+1)*resolution > max_lat (resp. min_lon + (k+1)*resolution > max_lon)
    """
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    lat_list = np.linspace(mlat, Mlat, nlats, endpoint=False)
    lon_list = np.linspace(mlon, Mlon, nlons, endpoint=False)
    geomap = np.fromfunction(lambda i, j: globe.is_land(lat_list[i]+0.5*resolution, lon_list[j]+0.5*resolution),
                             (nlats, nlons), dtype=int
                             )
    return geomap

def coast_distance(land):
    """
    Return a masked array D such that D[i, j] is a rough estimate of the point (lat(i, j), lon(i, j)) from the coast
    
    The lat, lon functions and the ranges are the same as the ones in the function land_matrix
    
    land is a boolean Numpy array that dictates the position of the land entries,
    such as the one returned by the function land_matrix
    
    In this implementation, this function is computed as:
    - if land[i, j] is True, then D[i, j] == 0
    - otherwise, D[i, j] = 1 + min{D[l, m] | (l, m) \in N(i, j) }
      where N(i, j) are the eight neighbouring entries of (i, j), i.e.
          N(i, j) = [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
    
    Please note that this function is NOT AWARE of the land outside the matrix provided
    as argument; in particular do NOT use this function with a False-only array,
    since it would enter in an infinite loop! Use coast_distance_with_coordinates
    with an appropriate value of "expand" instead.
    """
    # initialize coast_matrix with -1 on land and 0 on sea
    # 0 will mean "coast value currently undefined"
    coast = np.where(land, -1, 0)
    lvl = 1 # next coast distance
    while np.argwhere(coast==0).shape[0] != 0: # as long as there are zeros in coast...
        coast_temp = np.copy(coast) # create the new coast matrix
        for (i, j) in np.argwhere(coast==0): # for each value that is currently zero...
            # consider the neighbouring entries
            nb = [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
            # recover coast values in neighbourhood
            val = [] # recupera i valori dei punti dell'intorno
            for (k, l) in nb:
                if k >= 0 and l >= 0: # do not consider negative coordinates (REMEMBER: they do not rise an IndexError! -1 means "last entry")
                    try:
                        val.append(coast[k, l])
                    except IndexError:
                        pass
            if set(val) != set([0]): # if at least one of the neighbouring entry is either not undefined or land...
                coast_temp[i, j] = lvl # assign the current coast distance value to (i, j)
        lvl += 1 # finally update the current value...
        coast = coast_temp # ... and upgrade the map
    coast_matrix = np.ma.masked_where(coast==-1, coast) # mask land values
    return coast_matrix

def coast_distance_with_coordinates(min_lat, max_lat, min_lon, max_lon, resolution, expand=0):
    """
    Return a masked array D such that D[i, j] is a rough estimate of the point (lat(i, j), lon(i, j)) from the coast
    
    The lat, lon functions and the ranges are the same as the ones in the function land_matrix
    
    This function first computes the land matrix with the provided parameters using land_matrix,
    then calls coast_distance on the resulting array.
    
    In case there are not enough land points in the land matrix to have an acceptable
    result, the "expand" parameter can be used: if expand == n, the target area
    [min_lat, max_lat] x [min_lon, max_lon] is expanded in all directions by a
    quantity of n*resolution degrees before all computation, and the result is
    cut to the original area just before being returned.
    
    Uses either "coast_distance_integer" or "coast_distance_rational" depending on the
    value of the metod argument
    """
    n = expand
    mlat = float(min_lat) - n * resolution
    mlon = float(min_lon) - n * resolution
    Mlat = float(max_lat) + n * resolution
    Mlon = float(max_lon) + n * resolution
    res = float(resolution)    
    L = land_matrix(mlat, Mlat, mlon, Mlon, res)
    D = coast_distance(L)
    d0, d1 = D.shape
    return D[expand:d0-n, expand:d1-n]