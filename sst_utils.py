# sst_utils.py
# utilities for information extracting from SST NetCDF/HDF files

import numpy as np
import netCDF4
import pyhdf.SD as SD
from pyhdf.SD import SDC as SDconst
import os
import re
import pathlib
import shutil
from datetime import datetime, timedelta

# the following module is used in discard_files to recover the land mask independently from the files
# install with pip install global-land-mask
from global_land_mask import globe

# assumed sampling resolution inside NetCDF/HDF files (in degrees of lat/lon)
# we assume that the resolution of data in the files is about 0.01°
file_resolution = 0.01

def get_sst_data_from_file(filepath, infolist):
	"""
	Return a Python dictionary containing the information from a SST file data (either HDF or NetCDF)
	
	filepath is the file path, infolist is a list containing the information that will be extracted from the file
	
	Acceptable values in infolist:
	- "lat": matrix containing the latitudes of the points
	- "lon": matrix containing the longitudes of the points
	- "sst": matrix containing the values of the sst of the points
	- "quality": boolean matrix containing the quality mask, where True == bad quality and False == good quality
	- "clouds": boolean matrix containing the cloud mask, where True == cloud detected
	- "time": datetime object representing the date and time of image
	- "timedelta": matrix containing the time shift (in seconds) of the time of reading w.r.t. the time of image
	- "land": boolean matrix containing the land mask, where True == pixel on land
	
	The values in infolist are used as keys in the output dictionary
	"""
	output_dict = {}
	# choose file type
	file_pathobj = pathlib.Path(filepath)
	filetype = file_pathobj.suffix
	if filetype == ".hdf": # older Aqua files use this format
		ds = SD.SD(filepath, SDconst.READ)
		if "lat" in infolist:
			output_dict["lat"] = ds.select("latitude").get()
		if "lon" in infolist:
			output_dict["lon"] = ds.select("longitude").get()
		if "sst" in infolist:
			ds_data = ds.select("sst")
			try:
				dtemp_mask = ds_data.get() == ds_data.bad_value_scaled
			except AttributeError:
				dtemp_mask = ds_data.get() == -32767
			output_dict["sst"] = np.ma.masked_where(dtemp_mask, ds_data.slope*ds_data.get()+ds_data.intercept)
		if "quality" in infolist:
			output_dict["quality"] = ds.select("qual_sst").get() > 2 # this has True where the pixel is low-quality
		if "clouds" in infolist:
			output_dict["clouds"] = ds.select("l2_flags").get() & (1<<9) # set cloud where correspondent bit is set in the l2_flags array
		if "time" in infolist:
			regex_match = re.search(r"[0-9]{13}", file_pathobj.stem)
			output_dict["time"] = datetime.strptime(regex_match[0], "%Y%j%H%M%S") # for Aqua files, recover time info from filename
		if "timedelta" in infolist:
			regex_match = re.search(r"[0-9]{13}", file_pathobj.stem)
			ref_date = datetime.strptime(regex_match[0], "%Y%j%H%M%S")
			timeinfo = zip(ds.select("year").get(), ds.select("day").get(), ds.select("msec").get()) # recover pixel time info from relevant datasets
			dates = [datetime(int(dt[0]), 1, 1) + timedelta(days=int(dt[1])-1, milliseconds=int(dt[2])) for dt in timeinfo] # create array of datetime objects
			delta = np.array([(dt-ref_date).total_seconds() for dt in dates]) # compute difference with reference date
			output_dict["timedelta"] = np.tile(delta, (ds.attributes()["Pixels per Scan Line"], 1)).transpose()
		if "land" in infolist:
			output_dict["land"] = (ds.select("l2_flags").get() & 2) != 0 # 2 is the 'land' flag
	else: # filetype == ".nc"
		ds = netCDF4.Dataset(filepath)
		if "METOP" in file_pathobj.stem: # it is a METOP NetCDF file
			tzero = datetime(1981, 1, 1)
			if "lat" in infolist:
				output_dict["lat"] = ds["lat"][:]
			if "lon" in infolist:
				output_dict["lon"] = ds["lon"][:]
			if "sst" in infolist:
				output_dict["sst"] = ds["sea_surface_temperature"][0, :] - 273.15 # METOP SSTs are in Kelvin
			if "quality" in infolist:
				try:
					output_dict["quality"] = ds["quality_level"][0, :] < 3 # this has True where the pixel is low-quality
				except IndexError:
					output_dict["quality"] = np.full(ds["sea_surface_temperature"].shape, False) # older METOP files do not have quality data, so create a fake mask
			if "clouds" in infolist:
				output_dict["clouds"] = np.full(ds["sea_surface_temperature"].shape, False) # since METOP does not have cloud data, create a fake mask
			if "time" in infolist:
				output_dict["time"] = tzero + timedelta(seconds=int(ds["time"][0]))
			if "timedelta" in infolist:
				output_dict["timedelta"] = ds["sst_dtime"][0, :]
			if "land" in infolist:
				try:
					output_dict["land"] = ((ds["l2p_flags"][0, :] & 2) != 0).data # 2 is the 'land' flag
				except IndexError: # older METOP files do not have a 'l2p_flags' variable: land data is stored as bit 6 of the 'rejection_flag' variable
					output_dict["land"] = ((ds["rejection_flag"][0, :] & (1<<6)) !=0 ).data
		else: # it is an Aqua NetCDF file
			if "lat" in infolist:
				output_dict["lat"] = ds["navigation_data"]["latitude"][:]
			if "lon" in infolist:
				output_dict["lon"] = ds["navigation_data"]["longitude"][:]
			if "sst" in infolist:
				try:
					output_dict["sst"] = ds["geophysical_data"]["sst"][:]
				except IndexError:
					output_dict["sst"] = ds["geophysical_data"]["sst4"][:]
			if "quality" in infolist:
				try:
					output_dict["quality"] = ds["geophysical_data"]["qual_sst"][:] > 2 # this has True where the pixel is low-quality
				except IndexError:
					output_dict["quality"] = ds["geophysical_data"]["qual_sst4"][:] > 2 # this has True where the pixel is low-quality
			if "clouds" in infolist:
				output_dict["clouds"] = ds["geophysical_data"]["l2_flags"][:] & (1<<9) # set cloud where correspondent bit is set in the l2_flags array
			if "time" in infolist:
				regex_match = re.search(r"[0-9]{13}", file_pathobj.stem)
				output_dict["time"] = datetime.strptime(regex_match[0], "%Y%j%H%M%S") # for Aqua files, recover time info from filename
			if "timedelta" in infolist:
				regex_match = re.search(r"[0-9]{13}", file_pathobj.stem)
				ref_date = datetime.strptime(regex_match[0], "%Y%j%H%M%S")
				timeinfo = zip(ds["scan_line_attributes"]["year"][:], ds["scan_line_attributes"]["day"][:], ds["scan_line_attributes"]["msec"][:]) # recover pixel time info from relevant datasets
				dates = [datetime(int(dt[0]), 1, 1) + timedelta(days=int(dt[1])-1, milliseconds=int(dt[2])) for dt in timeinfo] # create array of datetime objects
				delta = np.array([(dt-ref_date).total_seconds() for dt in dates]) # compute difference with reference date
				output_dict["timedelta"] = np.tile(delta, (ds.dimensions["pixels_per_line"].size, 1)).transpose()
			if "land" in infolist:
				output_dict["land"] = ((ds["geophysical_data"]["l2_flags"][:] & 2) != 0).data # 2 is the 'land' flag; we remove the (empty) mask to obtain a regular numpy.array
	return output_dict

def discard_files(filedirs,
                  min_lat, max_lat, min_lon, max_lon,
                  discard_threshold=0.1,
                  verbose=False, log=False, simulate=False):
	"""
	Discard NetCDF/HDF files if they have too few values in them.
	In particular, a file is discarded if #(unmasked data in window)/#(total expected data in window) < discard_threshold
	
	Discarded files are moved to a subdirectory "Discarded" in each path contained in filedirs, which is created if it doesn't exist
	
	Parameters:
	filedirs : list of paths (string) of the directories containing the files to be selected
	min_lat,max_lat,min_lon,max_lon : floats representing the latitude/longitude boundaries of the target window
	discard_threshold : float representing the minimal fraction of data that has to be present in a file (default: 0.1)
	verbose : if True, print the status onscreen
	log : if True, for each path in filedirs produce a file called "discard_files_log_YYYYmmdd_HHMMSS.txt" (with the current time) in the Discarded subdirectory,
	      reporting which files have been discarded and why
	simulate : if True, do not actually move the files
	           PLEASE NOTE: if simulate is True, the logfile will be produced even if log=False
	"""
	for filedir in filedirs:
		target_path = pathlib.Path(filedir).joinpath("Discarded")
		try:
			os.mkdir(target_path)
		except FileExistsError: # if a folder named "Discarded" already exists in filedir, it will be used
			pass
		if log or simulate:
			logfile = open(target_path.joinpath("discarded_files_log_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".txt"), 'w')
			logfile.write(f"Target window: [{float(min_lat)},{float(max_lat)}]x[{float(min_lon)},{float(max_lon)}]\nDiscard threshold: {float(discard_threshold)}\n\n")
		all_the_files = sorted([filename for filename in os.listdir(filedir) if filename.endswith(".nc") or filename.endswith(".hdf")]) # this contains all the NetCDF/HDF in filepath
		total_files_number = len(all_the_files)
		# we estimate the number of data that we expect in the target window
		lats = np.arange(min_lat, max_lat, file_resolution)
		lons = np.arange(min_lon, max_lon, file_resolution)
		lon_grid, lat_grid = np.meshgrid(lons, lats)
		# recover land mask
		land_data = globe.is_land(lat_grid, lon_grid)
		# count number of False (i.e. sea points) in land_data
		expected_data = np.size(land_data) - np.count_nonzero(land_data)
		# now iterate
		# counters for the log file
		accepted_counter = 0
		discarded_counter = 0
		for filename in all_the_files:
			# exctract data
			d = get_sst_data_from_file(pathlib.Path(filedir).joinpath(filename), ["lat", "lon", "sst", "time"])
			# select the non-masked data in the target window
			target_data = np.argwhere((float(min_lat) <= d["lat"]) & (d["lat"] <= float(max_lat)) & (float(min_lon) <= d["lon"]) & (d["lon"] <= float(max_lon)) & ~d["sst"].mask)
			# if there is too few non-masked data in the window, discard the file
			if target_data.shape[0]==0 or target_data.shape[0] < discard_threshold * expected_data: # test the case "equal to 0" separately to avoid a false "0 < 0" in case discard_threshold is 0
				discarded_counter += 1
				if verbose:
					print(d["time"].strftime("%Y-%m-%d %H:%M:%S") + f" -- \033[31mDiscarded\033[00m: too few data (data coverage: {float(target_data.shape[0])/expected_data:.4f})")
				if log or simulate:
					logfile.write(d["time"].strftime("%Y-%m-%d %H:%M:%S") + f" -- Discarded: too few data (data coverage: {float(target_data.shape[0])/expected_data:.4f})\n")
				if not simulate:
					# for Python 3.9+:
					#shutil.move(pathlib.Path(filedir).joinpath(filename),target_path)
					# for older Python, shutil.move does not accept pathlib.Path objects
					shutil.move(os.fspath(pathlib.Path(filedir).joinpath(filename)), os.fspath(target_path))
			else: # continue without further actions (apart from the verbose and log checks)
				accepted_counter += 1
				if verbose:
					print(d["time"].strftime("%Y-%m-%d %H:%M:%S") + f" -- \033[32mAccepted\033[00m (data coverage: {float(target_data.shape[0])/expected_data:.4f})")
				if log or simulate:
					logfile.write(d["time"].strftime("%Y-%m-%d %H:%M:%S") + f" -- Accepted (data coverage: {float(target_data.shape[0])/expected_data:.4f})\n")
		if log or simulate:
			logfile.write(f"\nTotal accepted: {accepted_counter} of {total_files_number}\nTotal discarded: {discarded_counter} of {total_files_number}")
			logfile.close()