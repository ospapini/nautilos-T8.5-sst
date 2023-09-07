# spaghetti_plot.py
# tool for the analysis and classification of mesoscale upwelling events

import numpy as np
import os
import re
import itertools
import pathlib
from datetime import datetime, timedelta
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
import cartopy.feature as cfeat

# remember to put sst_utils.py and mec_geodata.py in the same folder as this file
import sst_utils as sstu
import mec_geodata as geodata

### PARAMETERS ###

# assumed sampling resolution inside NetCDF/HDF files (in degrees of lat/lon)
file_resolution = 0.01

# assumed number of input images per day
temporal_resolution = 2

# threshold for final classification
heatmap_threshold = 0.6

# dictionary on keys {1,2,3,4} such that the value for key i is a list of dictionaries,
# where each dictionary, whose keys are {"latmin","latmax","lonmin","lonmax"},
# defines a geographical rectangle.
# the union of the rectangles of the value for key i represents the geographical zone
# where we expect an event of type Ei
event_zone = {
1: [{"latmin": 36.5, "latmax": 38.0, "lonmin": -11.0, "lonmax": -9.0}],
2: [{"latmin": 35.75, "latmax": 37.0, "lonmin": -9.5, "lonmax": -8.75}],
3: [{"latmin": 36.5, "latmax": 37.75, "lonmin": -9.5, "lonmax": -8.75},
    {"latmin": 36.5, "latmax": 37.25, "lonmin": -8.75, "lonmax": -7.5}
    ],
4: [{"latmin": 36.5, "latmax": 37.5, "lonmin": -9.5, "lonmax": -9.0},
    {"latmin": 36.75, "latmax": 37.5, "lonmin": -9.0, "lonmax": -8.75},
    {"latmin": 36.75, "latmax": 37.25, "lonmin": -8.75, "lonmax": -8.5}
    ]
}

### CUSTOM SST CLASSES ###

class SpaghettiData():
    """
    Class that defines a single plot in a spaghetti plot.
    
    A SpaghettiData object is initialized with (and has) the following attributes:
    latitude, longitude, resolution : reference coordinates for the spaghetti plot
                                      i.e. the SST values for a specific time are obtained as average in the square [lat,lat+res]x[lon,lon+res]
                                      (all values in degrees, lat N is positive and lat S is negative, lon E is positive and lon W is negative)
    data : numpy.array consisting of a list of pairs (date,temperature) (date: datetime object; temperature: float representing temperature in Celsius)
	"""
    def __init__(self, latitude, longitude, resolution, data):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.resolution = float(resolution)
        if len(data) > 0:
            data_array = np.array(data)
            data_array = data_array[data_array[:, 0].argsort()]
        else:
            data_array = np.empty((0, 2), dtype=object) # if data is the empty list, create an empty array
        self.data = data_array
    
    def __repr__(self):
        lat_string = f"{abs(self.latitude)}° {'N' if self.latitude>=0 else 'S'}"
        lon_string = f"{abs(self.longitude)}° {'E' if self.longitude>=0 else 'W'}"
        return "Spaghetti data relative to point (" + lat_string + ", " + lon_string + f") @ {self.resolution}°"

class SpaghettiPlot():
    """
    Class that defines a spaghetti plot.
    
    Input variables for initialization (in this order):
    min_lat, max_lat, min_lon, max_lon : minimum and maximum latitude and longitude of the target area
                                         (in degrees, positive for lat N and lon E, negative for lat S and lon W)
    resolution : dimension of the small squares of the grid (in degrees)
	"""
    def __init__(self, min_lat, max_lat, min_lon, max_lon, resolution):
        """
        Initialize a spaghetti plot. The following are defined:
        self.min_lat, self.max_lat, self.min_lon, self.max_lon, self.resolution : from the input values
        self.latitude, self.longitude : matrices such that position (i,j) contains the latitude (resp. longitude) of the (SW corner of the) small square
        self.color : matrix such that position (i,j) contains the (r,g,b) triple of the color used in the spaghetti plot
                     a linearly interpolated gradient such that the corners are:
                     SW - blue (0,0,1); SE - magenta (1,0,0.5); NW - green (0,1,0.5); NE - yellow (1,1,0)
        self.spaghetti : dictionary indexed by (i,j) initialized with empty lists
        """
        self.min_lat = float(min_lat)
        self.max_lat = float(max_lat)
        self.min_lon = float(min_lon)
        self.max_lon = float(max_lon)
        self.resolution = float(resolution)
        nlats = round((self.max_lat - self.min_lat) / self.resolution)
        nlons = round((self.max_lon - self.min_lon) / self.resolution)
        lats = np.linspace(self.min_lat, self.max_lat, nlats, endpoint=False)
        lons = np.linspace(self.min_lon, self.max_lon, nlons, endpoint=False)
        self.longitude, self.latitude = np.meshgrid(lons, lats)
        latspace = np.linspace(0, 1, num=nlats)
        lonspace = np.linspace(0, 1, num=nlons)
        self.color = np.stack(np.meshgrid(lonspace, latspace) + [0.5*sum(np.meshgrid(lonspace[::-1], latspace[::-1]))], axis=-1)
        self.spaghetti = {}
        for (i, j) in itertools.product(range(nlats), range(nlons)):
            self.spaghetti[i, j] = np.empty((0, 2), dtype=object)
    
    def __repr__(self):
        lat_string = f"[{abs(self.min_lat)}° {'N' if self.min_lat>=0 else 'S'}, {abs(self.max_lat)}° {'N' if self.max_lat>=0 else 'S'}]"
        lon_string = f"[{abs(self.min_lon)}° {'E' if self.min_lon>=0 else 'W'}, {abs(self.max_lon)}° {'E' if self.max_lon>=0 else 'W'}]"
        return "Spaghetti plot with target area " + lat_string + " x " + lon_string + f" @ {self.resolution}°"
    
    def _index_from_lat_and_lon(self, lat, lon):
        """
        Given two floats lat and lon, representing a latitude and longitude, return a tuple (i,j) such that self.latitude[i,j]==lat and self.longitude[i,j]==lon
        (equality tested as floats), and raises a ValueError if not present.
        """
        coordinates = np.argwhere(np.isclose(self.latitude, float(lat)) & np.isclose(self.longitude, float(lon)))
        if coordinates.shape[0] == 0:
            raise ValueError(f"({lat},{lon}) is not a valid point")
        return tuple(coordinates[0])
    
    def add_plot_data(self, plot_data):
        """
        Add data for a single plot to the spaghetti plot.
        plot_data is a SpaghettiData object, whose latitude and longitude values are used to assign the plot to the corresponding position in the SpaghettiPlot
        """
        if not np.isclose(self.resolution, plot_data.resolution): # inequality between Decimal objects
            raise ValueError("Different resolutions for the SpaghettiPlot and the SpaghettiData objects")
        self.spaghetti[self._index_from_lat_and_lon(plot_data.latitude, plot_data.longitude)] = plot_data.data
    
    def plot(self, time_range=None, temperature_range=None):
        """
        Produce a plot object of the spaghetti plot.
        time_range is an optional list [start_time, end_time] (of datetime objects) used to set the x range.
        temperature_range is an optional list [temp_min, temp_max] (each in Celsius) used to set the y range.
        """
        fig, ax = plt.subplots()
        for plot in self.spaghetti:
            if self.spaghetti[plot].size > 0: # add a plot only if it is not empty
                ax.plot(self.spaghetti[plot][:, 0], self.spaghetti[plot][:, 1], color=self.color[plot], alpha=0.8, lw=0.7)
        ax.set_facecolor('black')
        ax.grid(axis='x', color='white')
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        if time_range is not None:
            ax.set_xlim(time_range)
        if temperature_range is not None:
            ax.set_ylim(temperature_range)
        ax.xaxis_date() # in case of empty plot
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y\n%H:%M'))
        ax.xaxis.set_tick_params(rotation=90)
        fig.set_tight_layout(True)
        return fig
    
    def plot_reference_grid(self, geomap=None):
        """
        Produce a plot of the reference grid. If there is no data for a point, the corresponding square is transparent in the grid.
        If geomap is None, plot the grid only; otherwise, geomap is a list/tuple [map_min_lat, map_max_lat, map_min_lon, map_max_lon] with the boundaries of the map
        """
        fig, ax = plt.subplots()
        nlons, nlats = self.color.shape[0:2]
        alpha = np.ones((nlons, nlats))
        for plot in self.spaghetti:
            if self.spaghetti[plot].size == 0:
                alpha[plot] = 0.0
        rgba = np.dstack([self.color, alpha])
        if geomap is None:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xticks([0, nlons])
            ax.set_yticks([0, nlats])
            ax.set_xticklabels([f"{abs(self.min_lon)}° {'E' if self.min_lon>=0 else 'W'}", f"{abs(self.max_lon)}° {'E' if self.max_lon>=0 else 'W'}"])
            ax.set_yticklabels([f"{abs(self.min_lat)}° {'N' if self.min_lat>=0 else 'S'}", f"{abs(self.max_lat)}° {'N' if self.max_lat>=0 else 'S'}"])
            ax.imshow(rgba, origin='lower', extent=(0, nlons, 0, nlats))
        else:
            map_min_lat = float(geomap[0])
            map_max_lat = float(geomap[1])
            map_min_lon = float(geomap[2])
            map_max_lon = float(geomap[3])
            # create a GeoAxis in cartopy and set extent to the chosen window
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([map_min_lon, map_max_lon, map_min_lat, map_max_lat])
            # recover coast position
            land_50m = cfeat.NaturalEarthFeature('physical', 'coastline', '50m')
            ax.add_feature(land_50m, edgecolor='black', facecolor='gray', alpha=0.4)
            # put parallels and meridians
            grid = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=0.3, linestyle='--')
            grid.xlocator = mpl.ticker.MultipleLocator(0.5)
            grid.ylocator = mpl.ticker.MultipleLocator(0.5)
            ax.set_xticks(np.arange(map_min_lon, map_max_lon+1))
            ax.set_xticklabels([f"{abs(j)}° {'E' if j>=0 else 'W'}" for j in np.arange(map_min_lon, map_max_lon+1)])
            ax.set_yticks(np.arange(map_min_lat ,map_max_lat+1))
            ax.set_yticklabels([f"{abs(i)}° {'N' if i>=0 else 'S'}" for i in np.arange(map_min_lat, map_max_lat+1)])
            ax.imshow(rgba, origin='lower', extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat])
        return fig

### MEC SCRIPTS ###

# DATA SELECTION is performed with the scripts in sst_utils.py

# DATA ORGANIZATION

def create_spaghetti_data(filedirs,
                          start_time, end_time,
                          min_lat, max_lat, min_lon, max_lon, resolution,
                          lower_weight=None, discard_threshold=None,
                          verbose=False, save_data=False
                          ):
    """
    Return a dictionary indexed on the pairs (lat,lon) where
    
    lat \in { min_lat + k*resolution | k \in NN } \cap [min_lat, max_lat),
    lon \in { min_lon + k*resolution | k \in NN } \cap [min_lon, max_lon),
    
    such that the value relative to the key (lat,lon) is a SpaghettiData with latitude=lat, longitude=lon, resolution=resolution, and data computed using
    NetCDF/HDF files in filedirs starting from start_time and ending on end_time.
    
    Keys are actually STRINGS representing lat and lon, because using floats directly brings to bad results. The format of those strings is :.xf, where x is
    the maximum number of decimal places between the representations of min_lat, max_lat, min_lon, max_lon, resolution
    
    Parameters:
    - filedirs : list of strings each representing the path of a directory in which data files are stored
            	 (use a list with one element if there is a single path)
    - start_time, end_time : datetime objects representing the start and end times of the considered period
    - min_lat, max_lat, min_lon, max_lon, resolution : floats used to define the spatial range and resolution of the SpaghettiData
    - lower_weight : if None (default), compute the standard average for a single square in the grid
                     if not None, this is a float representing the weight assigned to low quality data when computing the average (good quality data have weight = 1)
    - discard_threshold : if not None, this is a pair (frac,min) containing the parameters depending on which we include data from a file in a single SpaghettiData,
                          discarding a file if it has too few data in the reference window (i.e. if the SpaghettiData has key (lat,lon), its reference window is [lat,lat+res]x[lon,lon+res])
                          
                          - frac : float in [0.0, 1.0] such that a file is discarded if #(unmasked data in window)/#(total expected data in window) < frac,
                                   where we expect that data is sampled every 0.01°, independently of the presence of land
                                   Example: if we have chosen a SpaghettiData with resolution=0.05, we expect 25 data points in the window anywhere, even on land
                                   (in fact, this will probably always discard near-shore data, which are unreliable in any case)
                          - min : integer such that if #(unmasked data in window) < min, the file is discarded independently from the value of frac
                          
                          if None, it defaults with frac = 0.0 and min = 1, i.e. discard only if there is no (unmasked) data in the reference window
    - verbose : if True, print on terminal the status
    - save_data : if True, produce in the current directory a pickled object called "SpaghettiData_YYYYmmdd_HHMMSS.pickle" (with the current time) containing the returned dictionary, together with a text file called
                  "SpaghettiData_YYYYmmdd_HHMMSS.txt" describing the parameters used to produce it
    """
    file_dict = {} # this is a dictionary indexed on the paths in filedirs such that each entry is the list of the NetCDF/HDF files in that path in the considered time period
    file_time_list = [] # we don't need a dictionary here: we put all the times of the files together, independently on their paths
    if verbose:
        print("Selecting files in the chosen time period...", end='')
    for filepath in filedirs:
        all_the_files = [filename for filename in os.listdir(filepath) if filename.endswith(".nc") or filename.endswith(".hdf")] # this contains all the NetCDF/HDF in filepath
        file_dict[filepath] = []
        for filename in all_the_files: # select only files relative to the considered time period, with time of file taken from filename
            if "METOP" in filename:
                regex_match = re.search(r"[0-9]{8}_[0-9]{6}", filename)
                time_of_image = datetime.strptime(regex_match[0], "%Y%m%d_%H%M%S")
            else: # it is Aqua
                regex_match = re.search(r"[0-9]{13}", filename)
                time_of_image = datetime.strptime(regex_match[0], "%Y%j%H%M%S")
            if start_time <= time_of_image <= end_time:
                file_dict[filepath].append(filename)
                file_time_list.append(time_of_image)
    if verbose:
        print(" \033[32mDone.\033[00m\n")
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    # determine number of decimal places
    precision = max([len(str(parameter).split(".")[1]) for parameter in [mlat, Mlat, mlon, Mlon, res]])
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    lat_list = np.linspace(mlat, Mlat, nlats, endpoint=False)
    lon_list = np.linspace(mlon, Mlon, nlons, endpoint=False)
    # default values for discard_threshold
    if discard_threshold is None:
        discard_threshold = (0.0, 1)
    # number of data expected in a (res)x(res) window, assuming a sampling of file_resolution (defined on top of this file)
    expected_data = (res / file_resolution) ** 2
    temp_data = {} # temporary dictionary in which the (unsorted) lists of (time,sst) are stored
    for lat in lat_list:
        for lon in lon_list:
            key = (f"{lat:.{precision}f}", f"{lon:.{precision}f}")
            temp_data[key] = []
    file_counter = 0
    total_file_number = sum([len(file_dict[filepath]) for filepath in filedirs])
    for filepath in filedirs:
        for datafile in file_dict[filepath]:
            file_counter += 1
            if verbose:
                print(f"\033[F\033[2KExtracting data... file {file_counter} of {total_file_number}")
            if lower_weight is None:
                d = sstu.get_sst_data_from_file(os.path.join(filepath, datafile), ["lat", "lon", "sst", "time", "timedelta"])
            else:
                d = sstu.get_sst_data_from_file(os.path.join(filepath, datafile), ["lat", "lon", "sst", "quality", "time", "timedelta"])
            for lat in lat_list:
                for lon in lon_list:
                    key = (f"{lat:.{precision}f}", f"{lon:.{precision}f}") # key used in output dictionary
                    target_window = np.argwhere((lat < d['lat']) & (d['lat'] < lat+res) & (lon < d['lon']) & (d['lon'] < lon+res))
                    # if there are no points in the small window, skip file for this coordinates
                    if target_window.shape[0] == 0:
                        continue
                    # retrieve only non-masked values
                    unmasked_target_window = np.array([p for p in target_window if not d['sst'].mask[p[0], p[1]]])
                    # if there is not enough non-masked data  w.r.t. discard_threshold, skip file for this coordinates
                    # (this should be equivalent to (unmasked_target_window.shape[0] < discard_threshold[1]) OR (unmasked_target_window.shape[0] < discard_threshold[0]*expected_data)
                    if unmasked_target_window.shape[0] < max(discard_threshold[1], discard_threshold[0]*expected_data):
                        continue
                    # otherwise, first compute the time
                    target_time = d['time'] + timedelta(seconds=np.min(d['timedelta'][unmasked_target_window[:, 0], unmasked_target_window[:, 1]]))
                    # then compute the temperature
                    target_sst_values = d['sst'].data[unmasked_target_window[:, 0], unmasked_target_window[:, 1]]
                    target_sst_weight = np.ones(target_sst_values.shape) # give weight one to everything
                    if lower_weight is not None: # if we want to reduce the weight of low-quality values...
                        target_sst_quality = d['quality'][unmasked_target_window[:, 0], unmasked_target_window[:, 1]] # extract the information
                        target_sst_weight[target_sst_quality] = lower_weight # and set the weight of those values
                    try:
                        target_sst = np.average(target_sst_values, weights=target_sst_weight)
                    except ZeroDivisionError: # if we set lower_weight=0 (i.e. discard low-quality values), this happens when all values in the square have low quality (i.e. all weights are 0)
                        continue # in this case skip file for this point
                    # finally append the result to the temporary list
                    temp_data[key].append((target_time, target_sst))
    if verbose:
        print("\033[F\033[2KExtracting data... \033[32mDone.\033[00m")
        print("Populating dictionary of SpaghettiData...", end='')
    output_data = {}
    for lat in lat_list:
        for lon in lon_list:
            key = (f"{lat:.{precision}f}", f"{lon:.{precision}f}")
            output_data[key] = SpaghettiData(lat, lon, resolution, temp_data[key])
    if verbose:
        print(" \033[32mDone.\033[00m")
    if save_data:
        if verbose:
            print("Saving dictionary of SpaghettiData...", end='')
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open("SpaghettiData_" + current_time + ".txt", "w") as metadata_file:
            metadata_file.write(f"Target window: [{mlat},{Mlat}] x [{mlon},{Mlon}] @ {res}\n")
            metadata_file.write(f"Dictionary keys formatting: .{precision}f\n")
            metadata_file.write("Time coverage: from " + start_time.strftime("%Y-%m-%d %H:%M:%S") + " to " + end_time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            metadata_file.write(f"Discard threshold: {float(discard_threshold[0])} (minimum {discard_threshold[1]} value(s))\n")
            if lower_weight is None:
                metadata_file.write("No lower weight given to bad quality data\n")
            else:
                metadata_file.write(f"Weight given to bad quality data: {lower_weight}\n")
            filelist = sorted([datafile for filepath in filedirs for datafile in file_dict[filepath]])
            metadata_file.write(f"{len(filelist)} file(s) used:\n")
            for datafile in filelist:
                metadata_file.write(datafile + "\n")
        with open("SpaghettiData_" + current_time + ".pickle", "wb") as dict_file:
            pickle.dump(output_data, dict_file, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(" \033[32mDone.\033[00m")
    return output_data

def create_spaghetti_plot(filedirs,
                          start_time, end_time,
                          min_lat, max_lat, min_lon, max_lon, resolution,
                          lower_weight=None, discard_threshold=None,
                          verbose=False, save_data=False, load_data=None
                          ):
    """
    Produce a SpaghettiPlot object using data from NetCDF/HDF files in a directory.
    
    Parameters:
    - filedirs : list of strings each representing the path of a directory in which data files are stored
                 (use a list with one element if there is a single path)
    - start_time, end_time : datetime objects representing the start and end times of the considered period
    - min_lat, max_lat, min_lon, max_lon, resolution : floats used to define the SpaghettiPlot grid (see docstring for the SpaghettiPlot class)
    - lower_weight : if None (default), compute the standard average for a single square in the grid
                     if not None, this is a float representing the weight assigned to low quality data when computing the average (good quality data have weight = 1)
    - discard_threshold : if not None, this is a pair (frac,min) containing the parameters depending on which we include data from a file in a single SpaghettiData,
                          discarding a file if it has too few data in the reference window (i.e. if the SpaghettiData has key (lat,lon), its reference window is [lat,lat+res]x[lon,lon+res])
                          
                          - frac : float in [0.0, 1.0] such that a file is discarded if #(unmasked data in window)/#(total expected data in window) < frac,
                                   where we expect that data is sampled every 0.01°, independently of the presence of land
                                   Example: if we have chosen a SpaghettiData with resolution=0.05, we expect 25 data points in the window anywhere, even on land
                                   (in fact, this will probably always discard near-shore data, which are unreliable in any case)
                          - min : integer such that if #(unmasked data in window) < min, the file is discarded independently from the value of frac
                          
                          if None, it defaults with frac = 0.0 and min = 1, i.e. discard only if there is no (unmasked) data in the reference window
    - verbose : if True, print on terminal the status
    - save_data : if True, save the SpaghettiData dictionary (see docstring of create_spaghetti_data, as this is passed to it)
    - load_data : if not None, it is either a dictionary of SpaghettiData, as produced by create_spaghetti_data, or the path (string) of a pickled object representing
                  a dictionary of SpaghettiData, as produced by create_spaghetti_data with save_data=True
    """
    output_plot = SpaghettiPlot(min_lat, max_lat, min_lon, max_lon, resolution)
    if load_data is None:
        # create the SpaghettiData objects
        spaghetti_data = create_spaghetti_data(filedirs, start_time, end_time, min_lat, max_lat, min_lon, max_lon, resolution, lower_weight, discard_threshold, verbose, save_data)
    else:
        if isinstance(load_data, str):
            if verbose:
                print(f"\033[33mUsing {pathlib.Path(load_data).name} as data source.\033[00m")
            with open(load_data, "rb") as dict_file:
                spaghetti_data = pickle.load(dict_file)
        else: # in this case we assume that load_data is already a SpaghettiData dictionary
            if verbose:
                print(f"\033[33mUsing provided dictionary as data source.\033[00m")
            spaghetti_data = load_data
    # now populate the actual SpaghettiPlot
    if verbose:
        print("Populating SpaghettiPlot...", end='')
    # deduce key formatting
    # recover a key with next(iter(spaghetti_data.keys())), pick the latitude, split on the dot and take the length of the decimal part
    precision = len(next(iter(spaghetti_data.keys()))[0].split(".")[1])
    for key in output_plot.spaghetti.keys():
        square_lat = output_plot.latitude[key]
        square_lon = output_plot.longitude[key]
        spaghetti_key = (f"{square_lat:.{precision}f}", f"{square_lon:.{precision}f}")
        data_points = spaghetti_data[spaghetti_key]
        cut_data = data_points.data[np.where((start_time <= data_points.data[:, 0]) & (data_points.data[:, 0] <= end_time))]
        new_data_points = SpaghettiData(data_points.latitude, data_points.longitude, data_points.resolution, cut_data)
        output_plot.add_plot_data(new_data_points)
    if verbose:
        print(" \033[32mDone.\033[00m")
    return output_plot

def regularize_spaghetti_dict(sdict, method="discard", discard_threshold=1.5):
    """
    Given a SpaghettiData dictionary as created with create_spaghetti_data, return another SpaghettiData dictionary with the same keys, where data has been regularized
    
    Two methods are available:
    - "discard" (default): for each SpaghettiData, and for each pair (time[i],SST[i]) in its data attribute, compute the median of the set [SST[i-2],SST[i-1],SST[i],SST[i+1],SST[i+2]]
      (for i=0, use only [SST[0],SST[1],SST[2]]; for i=1, use only [SST[0],SST[1],SST[2],SST[3]]; similarly at the other end)
      and remove the pair (time[i],SST[i]) if SST[i] does NOT belong to the interval [median - discard_threshold*std, median + discard_threshold*std], where std is the standard deviation of
      the SpaghettiData as computed through extract_spaghetti_stats
    - "replace": compute the median of [SST[i-1],SST[i],SST[i+1]] instead (for i=0, use [SST[0],SST[1],SST[2]], and similarly at the other end) and replace the value SST[i] with this median
      (discard_threshold is ignored in this case)
    
    this method returns a new SpaghettiData dictionary and does not modify the input one
    """
    output_dict = {}
    for key in sdict.keys():
        lat = sdict[key].latitude
        lon = sdict[key].longitude
        res = sdict[key].resolution
        new_data = []
        ndata = sdict[key].data.shape[0]
        if ndata > 2: # if the SpaghettiData has too few values, do not proceed further and return an empty SpaghettiData for that key
            if method == "discard":
                stime = sdict[key].data[0, 0] # first time present in the SpaghettiData
                etime = sdict[key].data[-1, 0] # last time present in the SpaghettiData
                # compute std with extract_spaghetti_stats just for this SpaghettiData
                # extract_spaghetti_stats should return a 1x1 matrix in this case
                std = extract_spaghetti_stats(sdict, stime, etime, lat, lat+res, lon, lon+res, res, ["std"])["std"][0, 0]
                if ndata == 3: # separate the "short" case
                    med = np.median(sdict[key].data[:, 1])
                    for i in range(3):
                        if med - discard_threshold * std <= sdict[key].data[i, 1] <= med + discard_threshold * std:
                            new_data.append([sdict[key].data[i, 0], sdict[key].data[i, 1]])
                else: # here ndata >= 4
                    # we build the list of all the medians, adding the first ones and the last ones manually
                    med = [np.median(sdict[key].data[0:3, 1]), np.median(sdict[key].data[0:4, 1])]
                    for i in range(2, ndata-2):
                        med.append(np.median(sdict[key].data[i-2:i+3, 1]))
                    med += [np.median(sdict[key].data[ndata-4:ndata, 1]), np.median(sdict[key].data[ndata-3:ndata, 1])]
                    # now iterate over all the values in the SpaghettiData
                    for i in range(ndata):
                        if med[i] - discard_threshold * std <= sdict[key].data[i, 1] <= med[i] + discard_threshold * std:
                            new_data.append([sdict[key].data[i, 0], sdict[key].data[i, 1]])
            elif method == "replace":
                med = np.median(sdict[key].data[0:3, 1]) # first value
                new_data.append([sdict[key].data[0, 0], med])
                med = np.median(sdict[key].data[ndata-3:ndata, 1]) # last value
                new_data.append([sdict[key].data[ndata-1, 0], med])
                for i in range(1, ndata-1): # all the others
                    med = np.median(sdict[key].data[i-1:i+2, 1])
                    new_data.append([sdict[key].data[i, 0], med])
            else:
                raise RuntimeError("method not supported")
        output_dict[key] = SpaghettiData(lat, lon, res, new_data)
    return output_dict

def cut_spaghetti_dict(sdict, start_time, end_time):
    """
    Given a SpaghettiData dictionary as computed with create_spaghetti_data, return another SpaghettiData dictionary with the same keys where every SpaghettiData is "cut" to
    the given time peiod. In other words, the new data attributes contain only the pairs (time,SST) of the original data attributes with start_time <= time <= end_time.
    
    Both start_time and end_time are datetime.datetime objects.
    """
    output_dict = {}
    for key in sdict.keys():
        lat = sdict[key].latitude
        lon = sdict[key].longitude
        res = sdict[key].resolution
        new_data = [pair for pair in sdict[key].data if start_time <= pair[0] <= end_time]
        output_dict[key] = SpaghettiData(lat, lon, res, new_data)
    return output_dict

# STATISTICS COMPUTATION

def extract_spaghetti_stats(spaghetti_dict,
                            start_time, end_time,
                            min_lat, max_lat, min_lon, max_lon, resolution,
                            stats_list,
                            discard_threshold=0
                            ):
    """
    Given a dictionary of SpaghettiData obtained as in create_spaghetti_data, return a dictionary containing some statistics on the SpaghettiData in it.
    
    In particular, stats_list is a list that can contain the strings:
    - "mean" : produce the mean of a single SpaghettiData in the considered time period
    - "std" : produce the standard deviation of a single SpaghettiData in the considered time period
    - "regr" : produce the angular coefficient of the regression line for a single SpaghettiData in the considered time period
    - "quad" : produce the coefficient of the degree two term of the best-fitting parabola for a single SpaghettiData in the considered time period
    - "ndata" : produce the number of points in a single SpaghettiData in the considered time period
    The returned dictionary has these strings as keys.
    
    For each key, its value in the output dictionary is a 2-dimensional Numpy masked array such that its (i,j)-th entry corresponds to the statistic of the SpaghettiData with coordinates (lat,lon)
    where i (resp. j) is the index of lat (resp. lon) in numpy.arange(min_lat,max_lat,resolution) (resp. numpy.arange(min_lon,max_lon,resolution)). No statistic is computed for SpaghettiData
    containing less than or equal to discard_threshold points, and the corresponding entry in the matrix is masked.
    
    The provided resolution MUST BE THE SAME of all resolutions of SpaghettiData in the spaghetti_dict.
    
    The returned dictionary has one more key, "discard_threshold", containing the value of discard_threshold
    """
    # check resolution compatibility
    # we assume that spaghetti_dict has been well-constructed so that all SpaghettiData in it have the same resolution
    # therefore we only check a random one
    # recover a key with next(iter(spaghetti_dict.keys()))
    key = next(iter(spaghetti_dict.keys()))
    spaghetti_res = spaghetti_dict[key].resolution
    if not np.isclose(spaghetti_res, float(resolution)): # using np.isclose for floats
        raise RuntimeError("SpaghettiData resolution inconsistent with chosen resolution")
    # recover also the key formatting, i.e. the number of decimal places in the keys
    precision = len(key[0].split(".")[1])
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    lat_list = np.linspace(mlat, Mlat, nlats, endpoint=False)
    lon_list = np.linspace(mlon, Mlon, nlons, endpoint=False)
    key_list = [key for key in stats_list if key in ["mean", "std", "regr", "quad", "ndata"]] # remove extra items from stats_list
    output_dict = {}
    for key in key_list:
        output_dict[key] = np.ma.zeros((nlats, nlons))
        output_dict[key].mask = np.full((nlats, nlons), False)
    # now populate the matrices
    for (i, j) in itertools.product(range(nlats), range(nlons)):
        spaghetti_key = (f"{lat_list[i]:.{precision}f}", f"{lon_list[j]:.{precision}f}")
        data = np.array([point for point in spaghetti_dict[spaghetti_key].data if start_time <= point[0] <= end_time])
        if data.shape[0] <= discard_threshold: # if there are too few data
            for key in key_list:
                output_dict[key].mask[i, j] = True
            continue
        if "mean" in key_list:
            output_dict["mean"][i, j] = np.mean(data[:, 1])
        if "std" in key_list:
            output_dict["std"][i, j] = np.std(data[:, 1])
        if "regr" in key_list:
            times = mdates.date2num(data[:, 0])
            m, q = np.polyfit(times, data[:, 1].astype(float), 1)
            output_dict["regr"][i, j] = m
        if "quad" in key_list:
            times = mdates.date2num(data[:, 0])
            a, b, c = np.polyfit(times, data[:, 1].astype(float), 2)
            output_dict["quad"][i, j] = a
        if "ndata" in key_list:
            output_dict["ndata"][i, j] = data.shape[0]
    # finally add the "discard_threshold" key
    output_dict["discard_threshold"] = discard_threshold
    return output_dict

# CLASSIFICATION RULES

def classify_events(stats_dict,
                    min_lat, max_lat, min_lon, max_lon, resolution,
                    landmass_mask=None, coast_distance=None
                    ):
    """
    Return a dictionary indexed on the integers {1,2,3,4} such that the n-th entry is a 2-dimensional Numpy masked array such that its (i,j)-th entry corresponds to
    the score (normalized to [0;1]) given for event En by our rules to the point with coordinates (lat,lon) where i (resp. j) is the index of lat (resp. lon) in
    numpy.arange(min_lat,max_lat,resolution) (resp. numpy.arange(min_lon,max_lon,resolution)).
    
    stats_dict is a dictionary as returned from extract_spaghetti_stats, which must contain "mean", "std" and "regr" as keys. min_lat, max_lat, min_lon, max_lon, resolution
    define the boundary of the target area: this MUST agree with the values in the stats dictionary.
    
    landmass_mask is a Numpy boolean array containing the mask of the landmass; its shape must be consistent with min_lat, max_lat, min_lon, max_lon and resolution
    if it is None (default), it is computed through the functions of the module mec_geodata.
    
    coast_distance is a Numpy array containing an estimate of the distance from the coast for every square of the grid, in resolution units;
    its shape must be consistent with min_lat, max_lat, min_lon, max_lon and resolution. If it is None (default), it is computed through the functions of the module mec_geodata.
    """
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    # check agreement between stats_dict and provided grid info
    key = next(iter(stats_dict.keys())) # choose a random key
    frame_shape = stats_dict[key].shape # recover the shape in stats_dict
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    provided_shape = (nlats, nlons)
    if provided_shape != frame_shape:
        raise RuntimeError("Shape of the statistics dictionary inconsistent with the one computed from provided lat/lon and resolution")
    # prepare the dictionary of events
    event_dict = {}
    stats_mask = stats_dict[key].mask # recover the same mask of data in stats_dict
    if landmass_mask is not None:
        if landmass_mask.shape != provided_shape: # check consistency
            raise RuntimeError("Shape of the landmass mask inconsistent with the one computed from provided lat/lon and resolution")
    else:
        landmass_mask = geodata.land_matrix(mlat, Mlat, mlon, Mlon, res) # be sure to remove land points
    global_mask = stats_mask | landmass_mask
    if coast_distance is not None:
        if coast_distance.shape != provided_shape: # check consistency
            raise RuntimeError("Shape of the coast distance array inconsistent with the one computed from provided lat/lon and resolution")
        coast = coast_distance
    else:
        coast = geodata.coast_distance(landmass_mask)
    for n in range(1, 5):
        event_dict[n]=np.ma.zeros(frame_shape)
    ### BEGIN RULES ###
    keys_list = [(i, j) for (i, j) in itertools.product(range(nlats), range(nlons)) if not global_mask[i, j]]
    mn = stats_dict["mean"] # this and the following just for brevity
    sd = stats_dict["std"]
    rg = stats_dict["regr"]
    # compute the maximum score that a square can obtain depending on the rules
    # for E1 and E2, these are "global" values i.e. they do not depend on the point
    # for E3 and E4, there is a "global" part and a "local" part, depending on the neighbourhood of the square
    # in particular, rules (E[34].1) and (E[34].3.*) are global, whereas rules (E[34].2.*) are local: we assume that EVERY square in nbhd_sea (see below) may contribute to them (even if the squares do not belong to nhbd_coast)
    # here max_value[i] contains the maximum score for event Ei
    # max_value[0] is unused, max_value[1] and [2] are scalars (since they're global), and max_value[3] and [4] are arrays of shape (nlats, nlons), initially containing only the global part
    # max_value[3] and [4] will be updated after the rules; we initialize arrays local_max_value_E3 and local_max_value_E4 that will contain the local part
    max_value = [1, 22.0, 22.0] + 2 * [np.full((nlats, nlons), 16, dtype='float64')]
    local_max_value_E3 = np.zeros((nlats, nlons))
    local_max_value_E4 = np.zeros((nlats, nlons))
    for (i, j) in keys_list:
        nbhd = [(i-2, j), (i-1, j), (i+1, j), (i+2, j), (i, j-2), (i, j-1), (i, j+1), (i, j+2), (i+1, j-1), (i-1, j+1)] # the eight squares around (i,j) and the squares at distance 2 directly to N, S, E, W, minus the two squares at SW (i-1,j-1) and NE (i+1,j+1)
        nbhd_sea = [p for p in nbhd if 0 <= p[0] < nlats and 0 <= p[1] < nlons and not landmass_mask[p]] # only the squares on the sea, to be used for local_max_value_E3 and local_max_value_E4; also removes possible out-of-range values
        nbhd_coast = [p for p in nbhd_sea if abs(coast[p] - coast[i, j]) <= 1] # this is a list of the neighbours that have about the same distance from the coast as (i,j)
        nbhd_NW = [p for p in nbhd_coast if p[0] >= i and p[1] <= j] # divide in NW part and SE part, to be used in rules (E[34].2.*) and (E[34].3.*)
        nbhd_SE = [p for p in nbhd_coast if p[0] <= i and p[1] >= j]
        # here compute the local maximum for E3 and E4
        for p in nbhd_sea:
            if p == (i+1, j) or p == (i, j-1): # points directly to N and W
                local_max_value_E3[i, j] += 2
                local_max_value_E4[i, j] += 3
            elif p == (i-1, j) or p == (i, j+1): # points directly to S and E
                local_max_value_E3[i, j] += 3
                local_max_value_E4[i, j] += 2
            else:
                local_max_value_E3[i, j] += 2
                local_max_value_E4[i, j] += 2
        if rg[i, j] < -0.05 and sd[i, j] > 0.2: # apply only if there is some variation
            ### RULES FOR E1 ###
            east_check = False
            east_further_check = False
            # (E1.1.1) (i,j) decreases in temperature after the square on its east, i.e. its regression coefficient is less negative
            try:
                if rg[i, j] > rg[i, j+1]:
                    east_check = True
                    event_dict[1][i, j] += 6
            except IndexError: pass
            # (E1.1.2) same as (E1.1.1) but two squeares on its east
            try:
                if rg[i, j] > rg[i, j+2]:
                    east_further_check = True
                    event_dict[1][i, j] += 3
            except IndexError: pass
            # (E1.1.3) bonus of 2 pts if both (E1.1) and (E1.2) are verified and also rg[i,j]>rg[i,j+1]>rg[i,j+2]
            if east_check and east_further_check and rg[i, j+1] > rg[i, j+2]:
                event_dict[1][i, j] += 2
            # (E1.2) (i,j) is colder than the average of the three squares on its north and south. Awards (E1.2.1) 4 pts for north, (E1.2.2) 4 pts for south and (E1.2.3) an additional 2 bonus pts if both are satisfied
            north_check = False
            south_check = False
            try:
                north_mean = np.average(mn[i+1, j-1:j+2])
                if north_mean is not np.ma.masked:
                    if mn[i, j] < north_mean:
                        north_check = True
                        event_dict[1][i, j] += 4
            except IndexError: pass
            try:
                south_mean = np.average(mn[i-1, j-1:j+2])
                if south_mean is not np.ma.masked:
                    if mn[i, j] < south_mean:
                        south_check = True
                        event_dict[1][i, j] += 4
            except IndexError: pass
            if north_check and south_check:
                event_dict[1][i, j] += 2
            ### RULES FOR E2 ###
            north_check = False
            north_further_check = False
            # (E2.1.1) (i,j) decreases in temperature after the square on its north, i.e. its regression coefficient is less negative
            try:
                if rg[i, j] > rg[i+1, j]:
                    north_check = True
                    event_dict[2][i, j] += 6
            except IndexError: pass
            # (E2.1.2) same as (E2.1.1) but two squeares on its north
            try:
                if rg[i, j] > rg[i+2, j]:
                    north_further_check = True
                    event_dict[2][i, j] += 3
            except IndexError: pass
            # (E2.1.3) bonus of 2 pts if both (E2.1) and (E2.2) are verified and also rg[i,j]>rg[i+1,j]>rg[i+2,j]
            if north_check and north_further_check and rg[i+1, j] > rg[i+2, j]:
                event_dict[2][i, j] += 2
            # (E2.2) (i,j) is colder than the average of the three squares on its east and west. Awards (E2.2.1) 4 pts for east, (E2.2.2) 4 pts for west and (E2.2.3) an additional 2 bonus pts if both are satisfied
            east_check = False
            west_check = False
            try:
                east_mean = np.average(mn[i-1:i+2, j+1])
                if east_mean is not np.ma.masked:
                    if mn[i, j] < east_mean:
                        east_check = True
                        event_dict[2][i, j] += 4
            except IndexError: pass
            try:
                west_mean = np.average(mn[i-1:i+2, j-1])
                if west_mean is not np.ma.masked:
                    if mn[i, j] < west_mean:
                        west_check = True
                        event_dict[2][i, j] += 4
            except IndexError: pass
            if east_check and west_check:
                event_dict[2][i, j] += 2
            ### RULES FOR E3 ###
            # (E3.1) the neighbours at the same distance from the coast also decrease in temperature
            if max([rg[p] for p in nbhd_coast], default=0) < 0: # default to 0 so that if nbhd_coast is empty the condition is False
                event_dict[3][i, j] += 5
            # (E3.2.1) neighbours on the north and west decrease more in temperature: +2 points for each square in neighbourhood
            for p in nbhd_NW:
                if rg[i, j] > rg[p]:
                    event_dict[3][i, j] += 2
            # (E3.2.2) neighbours on the south decrease less in temperature
            for p in nbhd_SE:
                if rg[i, j] < rg[p]:
                    event_dict[3][i, j] += 4 - (abs(i-p[0]) + abs(j-p[1])) # 3 points if directly in the south/east, 2 if SE or two squares away
            # (E3.3.1) neighbours on the NW are colder on average and (E3.3.2) on the SE are warmer; (E3.3.3) bonus points if both are satisfied
            NW_check = False
            SE_check = False
            if len(nbhd_NW) > 0:
                NW_mean = np.average(np.array([mn[p] for p in nbhd_NW]))
                if NW_mean is not np.ma.masked:
                    if mn[i, j] > NW_mean:
                        NW_check = True
                        event_dict[3][i, j] += 4
            if len(nbhd_SE) > 0:
                SE_mean = np.average(np.array([mn[p] for p in nbhd_SE]))
                if SE_mean is not np.ma.masked:
                    if mn[i, j] < SE_mean:
                        SE_check = True
                        event_dict[3][i, j] += 4
            if NW_check and SE_check:
                event_dict[3][i, j] += 2
        if rg[i, j] > 0.05 and sd[i, j] > 0.2: # apply only if there is some variation
            ### RULES FOR E4 ###
            # (E4.1) the neighbours at the same distance from the coast also increase in temperature
            if min([rg[p] for p in nbhd_coast], default=0) > 0: # default to 0 so that if nbhd_coast is empty the condition is False
                event_dict[4][i, j] += 5
            # (E4.2.1) neighbours on the north and west increase less in temperature
            for p in nbhd_NW:
                if rg[i, j] > rg[p]:
                    event_dict[4][i, j] += 4 - (abs(i-p[0]) + abs(j-p[1])) # 3 points if directly in the north/west, 2 if NW or two squares away
            # (E4.2.2) neighbours on the south increase more in temperature: +2 points for each square in neighbourhood
            for p in nbhd_SE:
                if rg[i, j] < rg[p]:
                    event_dict[4][i, j] += 2
            # (E4.3.1) neighbours on the NW are warmer on average and (E4.3.2) on the SE are colder; (E4.3.3) bonus points if both are satisfied
            NW_check = False
            SE_check = False
            if len(nbhd_NW) > 0:
                NW_mean = np.average(np.array([mn[p] for p in nbhd_NW]))
                if NW_mean is not np.ma.masked:
                    if mn[i, j] > NW_mean:
                        NW_check = True
                        event_dict[4][i, j] += 4
            if len(nbhd_SE) > 0:
                SE_mean = np.average(np.array([mn[p] for p in nbhd_SE]))
                if SE_mean is not np.ma.masked:
                    if mn[i, j] < SE_mean:
                        SE_check = True
                        event_dict[4][i, j] += 4
            if NW_check and SE_check:
                event_dict[4][i, j] += 2
        # (HV) bonus point for all types of event if the temperature variation is significant (HV="high variation")
        if rg[i, j] < -0.1 and sd[i, j] > 1:
            event_dict[1][i, j] += 1
            event_dict[2][i, j] += 1
            event_dict[3][i, j] += 1
        if rg[i, j] > 0.1 and sd[i, j] > 1:
            event_dict[4][i, j] += 1
    ### END RULES ###
    # compute the global+local max_value for E3 and E4
    max_value[3] = max_value[3] + local_max_value_E3
    max_value[4] = max_value[4] + local_max_value_E4
    # now mask all the event_dict[n] and normalize to [0;1]
    for n in range(1, 5):
        event_dict[n].mask = global_mask
        event_dict[n] /= max_value[n]
    return event_dict

# CLASSIFICATION OUTPUT

def classification_heatmap(event_dict,
                           min_lat, max_lat, min_lon, max_lon, resolution,
                           percent_of_data_sources=None, event_zones=False
                           ):
    """
    Return the heatmap resulting from the classification process.
    
    event_dict is the dictionary returned by classify_events. The returned heatmap is a numpy array of the same shape built in the following way: for each (i,j),
    consider the set M(i,j):={m | event_dict[m][i,j]=max{event_dict[k][i,j] | k=1,2,3,4} and event_dict[m][i,j]>=0.6}, which is a subset of {1,2,3,4}. Then heatmap[i,j]=sum([2**(m-1) for m in M]).
    In other words, the result is a numpy array with integer entries in {0,...,15} that gives us the type of event(s) that has received the highest score(s), provided that this maximum score
    is greater than a threshold (currently 0.6). In particular we have the following values: 0 -> no event, 1 -> E1, 2 -> E2, 4 -> E3, 8 -> E4.
    
    If percent_of_data_sources is None, return only the heatmap array; otherwise percent_of_data_sources is a triple (N,S,E) where N a numpy array with the abundance of data in the (i,j)-th SpaghettiPlot
    (i.e. stats["ndata"] as returned by extract_spaghetti_stats) and S, E are datetime.datetime objects representing the starting and ending times respectively. In this case, the returned object is a pair
    (heatmap,percentages) where heatmap is the same as above and percentages is a numpy array of the same shape such that the (i,j)-th entry represent the percentage of data in the (i,j)-th square
    of the grid w.r.t. the expected data (as float numbers). For example, if the time period is 15 days long, we expect 30 data points, and if (i,j) is a SpaghettiPlot with 24 data points then percentages[i,j]=0.8.
    
    If event_zones is true, do not include a label m in M(i,j) if the coordinates of (i,j) are outside the area specified for event Em in the dictionary event_zone.
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
    # recover shape and mask of data in one random map of event_dict (e.g. the one for E1)
    provided_shape = (nlats, nlons)
    frame_shape = event_dict[1].shape
    if provided_shape != frame_shape:
        raise RuntimeError("Shape of the events dictionary inconsistent with the one computed from provided lat/lon and resolution")
    global_mask = event_dict[1].mask
    keys_list = [(i, j) for (i, j) in itertools.product(range(nlats), range(nlons)) if not global_mask[i, j]]
    heatmap = np.zeros((nlats, nlons), dtype=int)
    for (i, j) in keys_list:
        values = [event_dict[n][i, j] for n in range(1, 5)]
        candidate = max(values)
        if candidate >= heatmap_threshold: # defined above
            candidate_events = [n+1 for n, x in enumerate(values) if np.isclose(x, candidate)]
            if event_zones:
                candidate_events_filtered = []
                for n in candidate_events:
                    for rect in event_zone[n]:
                        if rect["latmin"] <= lat_list[i] < rect["latmax"] and rect["lonmin"] <= lon_list[j] < rect["lonmax"]:
                            candidate_events_filtered.append(n)
                            break # do not test further rectangles in event_zone[n]
            else:
                candidate_events_filtered = candidate_events
            heatmap[i, j] = sum([2 ** (n - 1) for n in candidate_events_filtered], 0)
    if percent_of_data_sources is None:
        return heatmap
    else:
        ndata = percent_of_data_sources[0]
        if provided_shape != ndata.shape:
            raise RuntimeError("Shape of the ndata array inconsistent with the one computed from provided lat/lon and resolution")
        start_time = percent_of_data_sources[1]
        end_time = percent_of_data_sources[2]
        image_delta = 24 / temporal_resolution # average interval between images, in hours (temporal_resolution defined above)
        expected_number = (end_time - start_time) / timedelta(hours=image_delta)
        percentages = ndata / expected_number
        return (heatmap, np.where(heatmap!=0, percentages, 0.0))

### PLOTTING UTILITIES ###

def plot_stats(stats_dict,
               start_time, end_time,
               min_lat, max_lat, min_lon, max_lon
               ):
    """
    Return a plot of the statistics of a SpaghettiPlot.
    
    stats_dict is a dictionary as returned from extract_spaghetti_stats. start_time and end_time are used only to compose the title of the plot (please use the same values
    as the ones given to extract_spaghetti_stats for consistency). min_lat, max_lat, min_lon, max_lon are the boundary of the target area.
    
    AT THE MOMENT the ranges of the plots are:
    - for the mean, all the values from stats_dict['mean']
    - for the standard deviation, the values from 0 to 0.99*max(stats_dict['std'])
    - for the regression coefficient, the values from -0.99*max(abs(stats_dict['regr'])) to +0.99*max(abs(stats_dict['regr']))
    - for the quadratic coefficient, the values from -0.99*max(abs(stats_dict['quad'])) to +0.99*max(abs(stats_dict['quad']))
    These last three limits are chosen in order to reduce noise from outliers (e.g. squares near the coast)
    - for the number of data, the values from 0.0 to 1.0 as the fraction of theoretical number of data (i.e. one SST value every 12 hours)
    """
    nstats = len(stats_dict.keys()) - 1 # remove the "discard_threshold" key
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    # prepare the figure
    fig, axs = plt.subplots(1, nstats, figsize=(6*nstats, 6), subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig.suptitle("From: " + start_time.strftime("%d-%b-%Y %H:%M:%S") + "\nTo: " + end_time.strftime("%d-%b-%Y %H:%M:%S"), fontsize=14)
    next_ax = 0
    # this is common to all the stat plots
    land_50m = cfeat.NaturalEarthFeature('physical', 'coastline', '50m')
    for ax in axs:
        ax.set_extent([mlon, Mlon, mlat, Mlat])
        ax.add_feature(land_50m, edgecolor='black', facecolor='gray', alpha=0.4)
        # put parallels and meridians
        grid = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=0.3, linestyle='--')
        grid.xlocator = mpl.ticker.MultipleLocator(0.5)
        grid.ylocator = mpl.ticker.MultipleLocator(0.5)
        ax.set_xticks(np.arange(mlon, Mlon+1))
        ax.set_xticklabels([f"{abs(j)}° {'E' if j>=0 else 'W'}" for j in np.arange(mlon, Mlon+1)])
        ax.set_yticks(np.arange(mlat, Mlat+1))
        ax.set_yticklabels([f"{abs(i)}° {'N' if i>=0 else 'S'}" for i in np.arange(mlat, Mlat+1)])
        ax.xaxis.tick_top()
    # now plot each stat
    img = {}
    cbax = {}
    if "mean" in stats_dict.keys():
        img[next_ax] = axs[next_ax].imshow(stats_dict["mean"], origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=mpl.cm.get_cmap('viridis'))
        divider = make_axes_locatable(axs[next_ax])
        cbax[next_ax] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[next_ax], label="SST mean (°C)", cax=cbax[next_ax], orientation='horizontal')
        next_ax += 1
    if "std" in stats_dict.keys():
        std_max = np.quantile(stats_dict["std"], 0.99)
        img[next_ax] = axs[next_ax].imshow(stats_dict["std"], origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=mpl.cm.get_cmap('Purples'), norm=mcolors.Normalize(vmin=0, vmax=std_max))
        divider = make_axes_locatable(axs[next_ax])
        cbax[next_ax] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[next_ax], label="SST standard deviation (°C)", cax=cbax[next_ax], orientation='horizontal', extend='max', extendfrac=0.02)
        next_ax += 1
    if "regr" in stats_dict.keys():
        regr_max = np.quantile(np.abs(stats_dict["regr"]), 0.99)
        img[next_ax] = axs[next_ax].imshow(stats_dict["regr"], origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=mpl.cm.get_cmap('RdBu_r'), norm=mcolors.Normalize(vmin=-regr_max, vmax=regr_max))
        divider = make_axes_locatable(axs[next_ax])
        cbax[next_ax] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[next_ax], label="SST linear regression coeff. (°C/day)", cax=cbax[next_ax], orientation='horizontal', extend='both', extendfrac=0.02)
        next_ax += 1
    if "quad" in stats_dict.keys():
        quad_max = np.quantile(np.abs(stats_dict["quad"]), 0.99)
        img[next_ax] = axs[next_ax].imshow(stats_dict["quad"], origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=mpl.cm.get_cmap('PiYG_r'), norm=mcolors.Normalize(vmin=-quad_max, vmax=quad_max))
        divider = make_axes_locatable(axs[next_ax])
        cbax[next_ax] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[next_ax], label="SST quadratic regression coeff. (°C/day²)", cax=cbax[next_ax], orientation='horizontal', extend='both', extendfrac=0.02)
        next_ax += 1
    if "ndata" in stats_dict.keys():
        image_delta = 24 / temporal_resolution # average interval between images, in hours (temporal_resolution defined above)
        expected_number = (end_time - start_time) / timedelta(hours=image_delta)
        cividis_cmap = mpl.cm.get_cmap('cividis')
        ndata_colors = cividis_cmap(np.linspace(0, 1, 256))
        discarded_cut = round(256 * stats_dict["discard_threshold"] / expected_number)
        ndata_colors[:discarded_cut, :] = np.array([1, 1, 1, 0])
        ndata_cmap = mcolors.ListedColormap(ndata_colors)
        img[next_ax] = axs[next_ax].imshow(stats_dict["ndata"]/expected_number, origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=ndata_cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
        divider = make_axes_locatable(axs[next_ax])
        cbax[next_ax] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[next_ax], label="Fraction of expected data", cax=cbax[next_ax], orientation='horizontal', extend='max', extendfrac=0.02)
        next_ax += 1
    fig.savefig(end_time.strftime("%Y%m%d")+"_stats.png", bbox_inches='tight')
    plt.close(fig)

def plot_classification_maps(event_dict,
                             start_time, end_time,
                             min_lat, max_lat, min_lon, max_lon, resolution,
                             tag=None
                             ):
    """
    Return four heatmaps, one for each event type, highlighting the areas in which we identify the presence of an event depending on the values of event_dict, which is the dictionary
    returned by classify_events. At the moment, the maps are saved into a PNG named "YYYYmmdd.png", where YYYY, mm, dd are the year, month and day of end_time.
    
    start_time and end_time are used to compose the title of the plot (please use the same values as the ones used during the computation of the statistics dictionary provided to classify_events for consistency).
    
    If tag is not None, it is a string that is attached to the name of the output PNG; in particular, if tag=<tag>, the name of the file is "YYYYmmdd_<tag>.png".
    """
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    # check agreement between event_dict and provided grid info
    frame_shape = event_dict[1].shape # recover the shape in one random map of event_dict (e.g. the one for E1)
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    provided_shape = (nlats, nlons)
    if provided_shape != frame_shape:
        raise RuntimeError("Shape of the events dictionary inconsistent with the one computed from provided lat/lon and resolution")
    # prepare the output figure
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig.suptitle("From: " + start_time.strftime("%d-%b-%Y %H:%M:%S") + "\nTo: " + end_time.strftime("%d-%b-%Y %H:%M:%S"), fontsize=14)
    # this is common to all the plots
    land_50m = cfeat.NaturalEarthFeature('physical', 'coastline', '50m')
    for ax in axs:
        ax.set_extent([mlon, Mlon, mlat, Mlat])
        ax.add_feature(land_50m, edgecolor='black', facecolor='gray', alpha=0.4)
        # put parallels and meridians
        grid = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=0.3, linestyle='--')
        grid.xlocator = mpl.ticker.MultipleLocator(0.5)
        grid.ylocator = mpl.ticker.MultipleLocator(0.5)
        ax.set_xticks(np.arange(mlon, Mlon+1))
        ax.set_xticklabels([f"{abs(j)}° {'E' if j>=0 else 'W'}" for j in np.arange(mlon, Mlon+1)])
        ax.set_yticks(np.arange(mlat, Mlat+1))
        ax.set_yticklabels([f"{abs(i)}° {'N' if i>=0 else 'S'}" for i in np.arange(mlat, Mlat+1)])
        ax.xaxis.tick_top()
    # now prepare each axis
    img = {}
    cbax = {}
    for n in range(4):
        img[n] = axs[n].imshow(event_dict[n+1], origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=mpl.cm.get_cmap('viridis_r'), norm=mcolors.Normalize(vmin=0, vmax=1))
        axs[n].set_title(f"E{n+1}", fontsize=18)
        divider = make_axes_locatable(axs[n])
        cbax[n] = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
        fig.colorbar(img[n], cax=cbax[n], orientation='horizontal')
    if tag is not None:
        outfilename = end_time.strftime("%Y%m%d-%H%M") + "_" + tag
    else:
        outfilename = end_time.strftime("%Y%m%d-%H%M")
    fig.savefig(outfilename+".png", bbox_inches='tight')
    plt.close(fig)

def plot_classification_heatmap(heatmap,
                                start_time, end_time,
                                min_lat, max_lat, min_lon, max_lon, resolution,
                                percentage_map=None, tag=None
                                ):
    """
    Return a plot of the heatmap returned by classification_heatmap, where each square is coloured with:
    - white: no event
    - green: E1
    - yellow: E2
    - blue: E3
    - red: E4
    - gray: more than one event 
    At the moment, the map is saved into a PNG named "YYYYmmdd_heat.png", where YYYY, mm, dd are the year, month and day of end_time.
    
    start_time and end_time are used to compose the title of the plot (please use the same values as the ones used during the computation of the statistics dictionary provided to classify_events for consistency).
    
    If percentage_map is not None, it is the array obtained from classification_heatmap with percent_of_data_sources!=None; in this case, each square reports also the percentage of data in that array
    (it is displayed as an integer number that represents the percentage).
    
    If tag is not None, it is a string that is attached to the name of the output PNG; in particular, if tag=<tag>, the name of the file is "YYYYmmdd_<tag>.png".
    """
    mlat = float(min_lat)
    mlon = float(min_lon)
    Mlat = float(max_lat)
    Mlon = float(max_lon)
    res = float(resolution)
    # check agreement between heatmap and provided grid info
    frame_shape = heatmap.shape
    nlats = round((Mlat - mlat) / res)
    nlons = round((Mlon - mlon) / res)
    lat_list = np.linspace(mlat, Mlat, nlats, endpoint=False)
    lon_list = np.linspace(mlon, Mlon, nlons, endpoint=False)
    provided_shape = (nlats, nlons)
    if provided_shape != frame_shape:
        raise RuntimeError("Shape of the heatmap inconsistent with the one computed from provided lat/lon and resolution")
    # prepare figure 
    fig_heat, ax_heat = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig_heat.suptitle("From: " + start_time.strftime("%d-%b-%Y %H:%M:%S") + "\nTo: " + end_time.strftime("%d-%b-%Y %H:%M:%S"), fontsize=14)
    ax_heat.set_extent([mlon, Mlon, mlat, Mlat])
    land_50m = cfeat.NaturalEarthFeature('physical', 'coastline', '50m')
    ax_heat.add_feature(land_50m, edgecolor='black', facecolor='gray', alpha=0.4)
    # put parallels and meridians
    grid_heat = ax_heat.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=0.3, linestyle='--')
    grid_heat.xlocator = mpl.ticker.MultipleLocator(0.5)
    grid_heat.ylocator = mpl.ticker.MultipleLocator(0.5)
    ax_heat.set_xticks(np.arange(mlon, Mlon+1))
    ax_heat.set_xticklabels([f"{abs(j)}° {'E' if j>=0 else 'W'}" for j in np.arange(mlon, Mlon+1)])
    ax_heat.set_yticks(np.arange(mlat, Mlat+1))
    ax_heat.set_yticklabels([f"{abs(i)}° {'N' if i>=0 else 'S'}" for i in np.arange(mlat, Mlat+1)])
    # define colormap:
    # 0 -- nothing (white)
    # 1 -- E1 (green)
    # 2 -- E2 (yellow)
    # 3 -- E3 (blue)
    # 4 -- E4 (red)
    # 5 -- more than one event (gray)
    heat_colors=np.array([[1.0,1.0,1.0,1.0],
                          [0.0,0.5,0.0,1.0],
                          [1.0,0.9,0.15,1.0],
                          [0.0,0.0,1.0,1.0],
                          [1.0,0.0,0.0,1.0],
                          [0.5,0.5,0.5,1.0]
                          ])
    heat_cmap = mcolors.ListedColormap(heat_colors)
    # change values of heatmap according to the colormap above
    heatmap_normalized = np.select([heatmap==0, heatmap==1, heatmap==2, heatmap==4, heatmap==8], [0, 1, 2, 3, 4], default=5) 
    ax_heat.imshow(heatmap_normalized, origin='lower', extent=[mlon, Mlon, mlat, Mlat], cmap=heat_cmap, norm=mcolors.Normalize(vmin=0, vmax=5))
    # add ndata information
    if percentage_map is not None:
        heat_labels = []
        for (i, j) in itertools.product(range(nlats), range(nlons)):
            if heatmap[i, j] > 0:
                heat_labels.append(ax_heat.text(lon_list[j] + 0.5 * res,
                                                lat_list[i] + 0.5 * res,
                                                f"{int(100 * percentage_map[i, j])}",
                                                fontsize="xx-small",
                                                ha="center",
                                                va="center",
                                                color="w" if heatmap[i, j]!=2 else "k",
                                                transform=ccrs.PlateCarree()
                                                ))
        # prepare legend
        # we have to define a handler specifically to put a string as a legend key
        # see https://stackoverflow.com/questions/27174425/how-to-add-a-string-as-the-artist-in-matplotlib-legend
        class TextHandler(mpl.legend_handler.HandlerBase):
            def create_artists(self, legend, text, xdescent, ydescent, width, height, fontsize, trans):
                tx = mpl.text.Text(width/2.0, height/2.0, text, fontsize="xx-small", color="k", ha="center", va="center")
                return [tx]
        heat_legend_keys = [mpl.lines.Line2D([0], [0], color='k', lw=0, marker='s', markerfacecolor=heat_colors[n], markeredgecolor=(0, 0, 0, 0)) for n in range(1, 6)] + ["NN"]
        ax_heat.legend(heat_legend_keys, ["E1", "E2", "E3", "E4", "multiple", "% of data"], handlelength=0.7, handler_map={str:TextHandler()})
    else:
        # in this case we don't need a key for the % of data in the legend
        heat_legend_keys = [mpl.lines.Line2D([0], [0], color='k', lw=0, marker='s', markerfacecolor=heat_colors[n], markeredgecolor=(0, 0, 0, 0)) for n in range(1, 6)]
        ax_heat.legend(heat_legend_keys, ["E1", "E2", "E3", "E4", "multiple"], handlelength=0.7)
    if tag is not None:
        outfilename = end_time.strftime("%Y%m%d-%H%M") + "_" + tag
    else:
        outfilename = end_time.strftime("%Y%m%d-%H%M")
    fig_heat.savefig(outfilename+"_heat.png", bbox_inches='tight', dpi=150)
    plt.close(fig_heat)