# -*- coding: utf-8 -*-
#
#  get_wrf.py

"""Functions for extraction of wind speed time series from WRF @lat,lon,h

1) Wrfouts are read based on dates defined and processed per day.

2) Data is extracted as iris cubes and concatenated in time.

3) Cubes are cleaned and prepared for processing.

4) U and V velocities are interpolated to a horizontal location.

5) Wind speed is calculated at that location and then interpolated to the desired height.

6) Bilinear for horizontal interpolation (lat,lon) and semi-logartihmic for vertical interpolation (model level numbers)

"""


import argparse
import datetime
import sys

import numpy as np
import iris
import iris.pandas
import scipy


def wd_data_func(u_data, v_data):
	""" Calculate wind direction from u and v velocities in degrees.
		`u_data` and `v_data` must have same dimensions

		Args:
			u_data (iris.cube.Cube): Cube that contains u velocity
			v_data (iris.cube.Cube): Cube that contains v velocity

		Returns:
			iris.cube.Cube

		Notes:
			See https://www.eol.ucar.edu/content/wind-direction-quick-reference
			for more info
	"""
	return (np.arctan2(u_data,v_data)*(180/np.pi))+180
def wd_units_func(u_cube, v_cube):
    if u_cube.units != getattr(v_cube, 'units', u_cube.units):
        raise ValueError('Units do not match')
    u_cube.units = 'degrees'
    return u_cube.units
wd_ifunc = iris.analysis.maths.IFunc(wd_data_func, wd_units_func)


def ws_data_func(u_data, v_data):
	""" Calculate wind speed from u and v velocities.
		`u_data` and `v_data` must have same dimensions

		Args:
			u_data (iris.cube.Cube): Cube that contains u velocity
			v_data (iris.cube.Cube): Cube that contains v velocity

		Returns:
			iris.cube.Cube
	"""
	return np.sqrt( u_data**2 + v_data**2 )
def ws_units_func(u_cube, v_cube):
    if u_cube.units != getattr(v_cube, 'units', u_cube.units):
        raise ValueError('Units do not match')
    return u_cube.units
ws_ifunc = iris.analysis.maths.IFunc(ws_data_func, ws_units_func)


def concat_list(cube_list):
	""" Concatenate a list of WRF cubes through time coordinates and
		make "nice" time coordinates with fix_time().

	Args:
		cube_list (): List of cubes to concatenate

	Returns:
		iris.cube.Cube
	"""
	#Fix WRF time coordinates
	def fix_time(cube1):
		iris.util.promote_aux_coord_to_dim_coord(cube1, 'XTIME')
		iris.coords.DimCoord.rename(cube1.coord('XTIME'),'time')
		cube1.coord('time').attributes = {}
		return cube1
	list_map = map(fix_time,cube_list)
	c_list = iris.cube.CubeList(list(list_map))
	iris.util.equalise_attributes(c_list)		#Fix problems with DT = n (n changes with WRF in adaptive timestep) and other different attributes
	iris.util.unify_time_units(c_list)
	cube = c_list.concatenate()[0]
	return cube

#Lat Lon Vert coordinates fix
def fix_coordinates(cube):
	#Check if staggered
	stag = str(cube.attributes['stagger'])

	#Remove particular characteristics of coordinates
	cube.coord('latitude').attributes = {}
	cube.coord('longitude').attributes = {}
	cube.coord('latitude').var_name = str('lat')
	cube.coord('longitude').var_name = str('lon')

	#Cube is X (west-east) staggered
	if stag == 'X' and cube.ndim == 4:
		#Read characteristics and attributes of cubes
		n_lat = cube.attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
		n_lon = cube.attributes['WEST-EAST_PATCH_END_STAG']
		lat_coord = cube[0,0,:,0].coord('latitude')
		lon_coord = cube[0,0,0,:].coord('longitude')
		lats = lat_coord.points
		lons = lon_coord.points

		#Brief check it is reading in order
		if len(lats) != n_lat:
			print('check iris is reading WRF coordinates in order')
			sys.exit()
		if len(lons) != n_lon:
			print('check iris is reading WRF coordinates in order')
			sys.exit()

		#Modify latitude and longitude
		iris.cube.Cube.remove_coord(cube,'latitude')
		iris.cube.Cube.remove_coord(cube,'longitude')
		new_lat = iris.coords.AuxCoord.copy(lat_coord)
		new_lon = iris.coords.AuxCoord.copy(lon_coord)
		iris.cube.Cube.add_aux_coord(cube,new_lat,2)
		iris.cube.Cube.add_aux_coord(cube,new_lon,3)
		iris.util.promote_aux_coord_to_dim_coord(cube,'latitude')
		iris.util.promote_aux_coord_to_dim_coord(cube,'longitude')

		#Modify height
		height_n = len(cube[0,:,0,0].data)
		points = np.linspace(1, height_n, num=height_n, endpoint=True, retstep=False, dtype=int, axis=0)
		height_coord = iris.coords.DimCoord(points, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
		iris.cube.Cube.add_dim_coord(cube,height_coord,1)

	#Cube is Y (south-north) staggered
	elif stag == 'Y' and cube.ndim == 4:
		#Read characteristics and attributes of cubes
		n_lat = cube.attributes['SOUTH-NORTH_PATCH_END_STAG']
		n_lon = cube.attributes['WEST-EAST_PATCH_END_UNSTAG']
		lat_coord = cube[0,0,:,0].coord('latitude')
		lon_coord = cube[0,0,0,:].coord('longitude')
		lats = lat_coord.points
		lons = lon_coord.points

		#Brief check it is reading in order
		if len(lats) != n_lat:
			print('check iris is reading WRF coordinates in order')
			sys.exit()
		if len(lons) != n_lon:
			print('check iris is reading WRF coordinates in order')
			sys.exit()

		#Modify latitude and longitude
		iris.cube.Cube.remove_coord(cube,'latitude')
		iris.cube.Cube.remove_coord(cube,'longitude')
		new_lat = iris.coords.AuxCoord.copy(lat_coord)
		new_lon = iris.coords.AuxCoord.copy(lon_coord)
		iris.cube.Cube.add_aux_coord(cube,new_lat,2)
		iris.cube.Cube.add_aux_coord(cube,new_lon,3)
		iris.util.promote_aux_coord_to_dim_coord(cube,'latitude')
		iris.util.promote_aux_coord_to_dim_coord(cube,'longitude')

		#Modify height
		height_n = len(cube[0,:,0,0].data)
		points = np.linspace(1, height_n, num=height_n, endpoint=True, retstep=False, dtype=int, axis=0)
		height_coord = iris.coords.DimCoord(points, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
		iris.cube.Cube.add_dim_coord(cube,height_coord,1)

	#Cube is unstaggered X, Y and one Z level
	elif stag == '' and cube.ndim == 3:
		#Read characteristics and attributes of cubes
		n_lat = cube.attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
		n_lon = cube.attributes['WEST-EAST_PATCH_END_UNSTAG']
		lat_coord = cube[0,:,0].coord('latitude')
		lon_coord = cube[0,0,:].coord('longitude')
		lats = lat_coord.points
		lons = lon_coord.points

		#Brief check it is reading in order
		if len(lats) != n_lat:
			print('check iris is reading WRF coordinates in order')
			sys.exit()
		if len(lons) != n_lon:
			print('check iris is reading WRF coordinates in order')
			sys.exit()

		#Modify latitude and longitude
		iris.cube.Cube.remove_coord(cube,'latitude')
		iris.cube.Cube.remove_coord(cube,'longitude')
		new_lat = iris.coords.AuxCoord.copy(lat_coord)
		new_lon = iris.coords.AuxCoord.copy(lon_coord)
		iris.cube.Cube.add_aux_coord(cube,new_lat,1)
		iris.cube.Cube.add_aux_coord(cube,new_lon,2)
		iris.util.promote_aux_coord_to_dim_coord(cube,'latitude')
		iris.util.promote_aux_coord_to_dim_coord(cube,'longitude')

	#Cube is unstaggered X, Y and many Z levels
	elif stag == '' and cube.ndim == 4:
		#Read characteristics and attributes of cubes
		n_lat = cube.attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
		n_lon = cube.attributes['WEST-EAST_PATCH_END_UNSTAG']
		lat_coord = cube[0,0,:,0].coord('latitude')
		lon_coord = cube[0,0,0,:].coord('longitude')
		lats = lat_coord.points
		lons = lon_coord.points

		#Brief check it is reading in order
		if len(lats) != n_lat:
			print('check iris is reading WRF coordinates in order')
			sys.exit()
		if len(lons) != n_lon:
			print('check iris is reading WRF coordinates in order')
			sys.exit()

		#Modify latitude and longitude
		iris.cube.Cube.remove_coord(cube,'latitude')
		iris.cube.Cube.remove_coord(cube,'longitude')
		new_lat = iris.coords.AuxCoord.copy(lat_coord)
		new_lon = iris.coords.AuxCoord.copy(lon_coord)
		iris.cube.Cube.add_aux_coord(cube,new_lat,2)
		iris.cube.Cube.add_aux_coord(cube,new_lon,3)
		iris.util.promote_aux_coord_to_dim_coord(cube,'latitude')
		iris.util.promote_aux_coord_to_dim_coord(cube,'longitude')

		#Modify height
		height_n = len(cube[0,:,0,0].data)
		points = np.linspace(1, height_n, num=height_n, endpoint=True, retstep=False, dtype=int, axis=0)
		height_coord = iris.coords.DimCoord(points, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
		iris.cube.Cube.add_dim_coord(cube,height_coord,1)

	#Cube is unstaggered X and Y, but not Z
	elif stag == 'Z' and cube.ndim == 4:
		#Read characteristics and attributes of cubes
		n_lat = cube.attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
		n_lon = cube.attributes['WEST-EAST_PATCH_END_UNSTAG']
		lat_coord = cube[0,0,:,0].coord('latitude')
		lon_coord = cube[0,0,0,:].coord('longitude')
		lats = lat_coord.points
		lons = lon_coord.points

		#Brief check it is reading in order
		if len(lats) != n_lat:
			print('check iris is reading WRF coordinates in order')
			sys.exit()
		if len(lons) != n_lon:
			print('check iris is reading WRF coordinates in order')
			sys.exit()

		#Modify latitude and longitude
		iris.cube.Cube.remove_coord(cube,'latitude')
		iris.cube.Cube.remove_coord(cube,'longitude')
		new_lat = iris.coords.AuxCoord.copy(lat_coord)
		new_lon = iris.coords.AuxCoord.copy(lon_coord)
		iris.cube.Cube.add_aux_coord(cube,new_lat,2)
		iris.cube.Cube.add_aux_coord(cube,new_lon,3)
		iris.util.promote_aux_coord_to_dim_coord(cube,'latitude')
		iris.util.promote_aux_coord_to_dim_coord(cube,'longitude')

		#Modify height
		height_n = len(cube[0,:,0,0].data)
		points = np.linspace(1, height_n, num=height_n, endpoint=True, retstep=False, dtype=int, axis=0)
		height_coord = iris.coords.DimCoord(points, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
		iris.cube.Cube.add_dim_coord(cube,height_coord,1)

	else:
		print('No option to read this variable')
		sys.exit()

	cube.attributes = None
	return cube

#Get geopotential height
def height_data_func(ph_data, phb_data):
	return (ph_data + phb_data)/scipy.constants.g
def height_units_func(ph_data, phb_data):
    if ph_data.units != getattr(phb_data, 'units', ph_data.units):
        raise ValueError('Units do not match')
    return 'm'
height_ifunc = iris.analysis.maths.IFunc(height_data_func, height_units_func)

#Obtain unstaggered heights (at wind speed levels) from staggered PH and PHB at one horizontal location
#Requires fix_coordinates() for ph and phb cubes
#For one horizontal point and multiple times (time,vert_levels)
def unstag_geoheights1D(PH_site,PHB_site):
	h_levels = height_ifunc(PH_site, PHB_site, new_name='Height levels')

	h_stag = np.zeros((len(h_levels.coord('time').points),len(h_levels.coord('model_level_number').points)))
	h_unstag = np.zeros((len(h_levels.coord('time').points),len(h_levels.coord('model_level_number').points)-1))

	#Loop through array of every time_step. Contains heights (m) at all levels for that time step
	for t, heights in enumerate(h_levels.slices(['model_level_number'])):
		h_stag[t,:] = heights.data - heights.data[0]							#Get level over terrain height
		for level in range(len(h_stag[0,:])-1):									#Interpolate to mid levels
			h_stag[t,level] = (h_stag[t,level] + h_stag[t,level+1])/2
		h_unstag[t,:] = h_stag[t,:-1].copy()									#Erase upper level (staggered)

	#Make time coordinate
	time_coord = iris.coords.DimCoord.copy(h_levels.coord('time'))
	#Make height coordinate
	height_n = len(h_unstag[0,:])
	h_points = np.linspace(1, height_n, num=height_n, endpoint=True, retstep=False, dtype=int, axis=0)
	height_coord = iris.coords.DimCoord(h_points, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
	#Make lat and lon (scalar) coordinates
	lat_coord = iris.coords.AuxCoord.copy(h_levels.coord('latitude'))
	lon_coord = iris.coords.AuxCoord.copy(h_levels.coord('longitude'))

	#Create cube
	cube = iris.cube.Cube(h_unstag, standard_name='height', units='m', var_name='zh', dim_coords_and_dims=[(time_coord, 0),(height_coord, 1)])
	iris.cube.Cube.add_aux_coord(cube,lat_coord)
	iris.cube.Cube.add_aux_coord(cube,lon_coord)
	return cube

#Obtain unstaggered heights (at wind speed levels) from staggered PH and PHB at multiple horizontal location
#Requires fix_coordinates() for ph and phb cubes
#For multiple horizontal points and multiple times (time,vert_levels,lat,lon)
def unstag_geoheights2D(PH,PHB):
	h_levels = height_ifunc(PH, PHB, new_name='Height levels')

	#Loop through array of every time_step. Contains heights (m) at all levels for that time step
	cube_list_t = [None] * len(h_levels.coord('time').points)
	for t, ht in enumerate(h_levels.slices(['model_level_number','latitude','longitude'])):
		cube = ht - ht[0]									#Get height over terrain
		cube_list_l = [None] * int(len(h_levels.coord('model_level_number').points)-1)
		for level in range(len(h_levels.coord('model_level_number').points)-1):			#Interpolate to mid levels
			cube_list_l[level] = (cube[level,:,:]+cube[level+1,:,:]/2)
			#Add coordinates
			height_coord = iris.coords.DimCoord(level+1, standard_name='model_level_number', long_name=None, var_name='lev', units=1, attributes={})
			iris.cube.Cube.add_aux_coord(cube_list_l[level],height_coord)
		#Arrange list of height
		cube_list_l = iris.cube.CubeList(cube_list_l)
		cube_l = cube_list_l.merge()[0]
		cube_list_t[t] = cube_l
	#Arrange list of time
	cube_list = iris.cube.CubeList(cube_list_t)
	cube = cube_list.merge()[0]
	#Add metadata
	cube.rename('height')
	cube.units = 'm'
	cube.var_name = 'zh'
	return cube

#Semilog-interpolation using lower and upper value of wt_height
#Obtains wind speed at hub height (requires unstag_geoheights() for h_levels)
#For one horizontal point and multiple times (time,vert_levels)
def vert_semilog_interp_1D(h_levels,ws_levels,ws_10m,wt_height):
	ws_h_height = np.zeros(len(h_levels.coord('time').points))
	#Loop through arrays of every time_step. Contains heights (m) and wind speeds (m/s) at all levels for that time step
	for t , (heights, w_speeds) in enumerate(zip(h_levels.slices(['model_level_number']), ws_levels.slices(['model_level_number']))):
		idx = np.searchsorted(heights.data,wt_height)	#Find upper height of wt_height
		z_low = heights[idx-1].data						#Define height of lower level (cube)
		z_up = heights[idx].data						#Define height of upper level (cube)

		#Height under lower layer: using 10 m height for low level
		if (wt_height < z_low):
			z_low = 10
			z_up = heights[idx].data
			ws_low = ws_10m.data[t]
			ws_up = w_speeds[idx].data
			#print('ERROR: Choose interpolation height higher than '+ str(round(float(heights[0].data))) + ' m')
			#sys.exit()
			if (wt_height < 10 ):
				print('ERROR: Choose interpolation height higher than 10m')
				sys.exit()
		else:
			ws_low = w_speeds[idx-1].data					#Define wind speed of lower level (cube)
			ws_up = w_speeds[idx].data						#Define wind speed of upper level (cube)
		ws_turb = (ws_up - ws_low) * np.log(wt_height/z_low) / np.log(z_up/z_low) + ws_low
		ws_h_height[t] = ws_turb
	#Make time coordinate
	time_coord = iris.coords.DimCoord.copy(h_levels.coord('time'))
	#Make height coordinate
	height_coord = iris.coords.AuxCoord(wt_height, standard_name='height', long_name=None, var_name='zh', units='m', attributes={})
	#Make lat and lon (scalar) coordinates
	lat_coord = iris.coords.AuxCoord.copy(h_levels.coord('latitude'))
	lon_coord = iris.coords.AuxCoord.copy(h_levels.coord('longitude'))

	#Create cube
	cube = iris.cube.Cube(ws_h_height, standard_name='wind_speed', units='m s-1', dim_coords_and_dims=[(time_coord, 0)])
	iris.cube.Cube.add_aux_coord(cube,lat_coord)
	iris.cube.Cube.add_aux_coord(cube,lon_coord)
	iris.cube.Cube.add_aux_coord(cube,height_coord)
	return cube

#Generic vertical
def vert_linear_interp_1D(h_levels,var_levels,wt_height):
	var_h_height = np.zeros(len(h_levels.coord('time').points))
	#Loop through arrays of every time_step. Contains heights (m) and a variable at all levels for that time step
	for t , (heights, var_values) in enumerate(zip(h_levels.slices(['model_level_number']), var_levels.slices(['model_level_number']))):
		idx = np.searchsorted(heights.data,wt_height)	#Find upper height of wt_height
		z_low = heights[idx-1].data						#Define height of lower level (cube)
		z_up = heights[idx].data						#Define height of upper level (cube)

		if (wt_height < heights[0].data ):
			print('ERROR: Choose interpolation height higher than '+ str(round(float(heights[0].data))) + ' m')
			sys.exit()

		var_low = var_values[idx-1].data					#Define variable value of lower level (cube)
		var_up = var_values[idx].data						#Define variable value of upper level (cube)
		var_hh = var_low + (wt_height-z_low)*(var_up-var_low)/(z_up-z_low)
		var_h_height[t] = var_hh

	#Make time coordinate
	time_coord = iris.coords.DimCoord.copy(h_levels.coord('time'))
	#Make height coordinate
	height_coord = iris.coords.AuxCoord(wt_height, standard_name='height', long_name=None, var_name='zh', units='m', attributes={})
	#Make lat and lon (scalar) coordinates
	lat_coord = iris.coords.AuxCoord.copy(h_levels.coord('latitude'))
	lon_coord = iris.coords.AuxCoord.copy(h_levels.coord('longitude'))

	#Create cube
	cube = iris.cube.Cube(var_h_height, standard_name=var_levels.name(), units=var_levels.units, dim_coords_and_dims=[(time_coord, 0)])
	iris.cube.Cube.add_aux_coord(cube,lat_coord)
	iris.cube.Cube.add_aux_coord(cube,lon_coord)
	iris.cube.Cube.add_aux_coord(cube,height_coord)
	return cube

#Semilog-interpolation using numpy interpolation
#Obtains wind speed at hub height (requires unstag_geoheights() for h_levels)
#For multiple horizontal points and multiple times (time,vert_levels,lat,lon)
def vertinterp_2D(h_levels,ws_levels,wt_height):
	loghh = np.log(wt_height)
	data = np.zeros((len(h_levels.coord('latitude').points),len(h_levels.coord('longitude').points)))
	cube_list_t = [None] * len(h_levels.coord('time').points)
	for t , (heights, w_speeds) in enumerate(zip(h_levels.slices(['model_level_number','latitude','longitude']), ws_levels.slices(['model_level_number','latitude','longitude']))):
		data = np.zeros((len(heights.data[0,:,0]),len(heights.data[0,0,:])))
		for lat in range(len(heights.data[0,:,0])):
			for lon in range(len(heights.data[0,0,:])):
				logh = np.log(heights.data[:,lat,lon])
				ws = w_speeds.data[:,lat,lon]
				ws_hh = np.interp(loghh, logh, ws)
				data[lat,lon] = ws_hh
				#Check we can make the interpolation
				if (wt_height < heights.data[0,lat,lon]):
					print('ERROR: Choose interpolation height higher than '+ str(round(float(heights.data[0,lat,lon]))) + ' m')
					sys.exit()
		#Make time (scalar) coordinate
		time_coord = iris.coords.AuxCoord.copy(heights.coord('time'))
		#Make height coordinate
		height_coord = iris.coords.AuxCoord(wt_height, standard_name='height', long_name=None, var_name='zh', units='m', attributes={})
		#Make lat and lon (scalar) coordinates
		lat_coord = iris.coords.AuxCoord.copy(heights.coord('latitude'))
		lon_coord = iris.coords.AuxCoord.copy(heights.coord('longitude'))
		#Create cube
		cube = iris.cube.Cube(data, standard_name='wind_speed', units='m s-1', dim_coords_and_dims=[(lat_coord, 0),(lon_coord, 1)])
		iris.cube.Cube.add_aux_coord(cube,height_coord)
		iris.cube.Cube.add_aux_coord(cube,time_coord)
		cube_list_t[t] = cube
	#Arrange list of time
	c_list = iris.cube.CubeList(cube_list_t)
	cube = c_list.merge()[0]
	return cube

def extract_hh_cube(wrfdir,date_bgn,date_end,grid,wt_loc,wt_height):
	date_bgn = datetime.datetime.strptime(date_bgn,'%d-%m-%Y').date()
	date_end = datetime.datetime.strptime(date_end,'%d-%m-%Y').date()

	#Print grid and resolution to interpolate
	sample_year = date_bgn.strftime('%Y')
	sample_month = date_bgn.strftime('%m')
	sample_day = date_bgn.strftime('%d')
	sample_file = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, sample_year, sample_month, sample_day))
	sample_cube = iris.load(sample_file,['U'])[0]
	x = int(sample_cube.attributes['DX']/1000)
	y = int(sample_cube.attributes['DY']/1000)
	print (' << Interpolating wind speed/direction for WRF grid ' + str(grid) + ' ('+ str(x) + 'km x ' + str(y) +'km) >>')

	#Interpolation routine
	date_range = (date_bgn + datetime.timedelta(n) for n in range((date_end - date_bgn).days + 1)) #Generator
	cube_list_ws = [None] * len(range((date_end - date_bgn).days + 1))
	cube_list_wd = [None] * len(range((date_end - date_bgn).days + 1))
	it = 0
	for date in date_range:
		print('\tReading files for: ' + date.strftime('%d') +'-'+ date.strftime('%m') +'-'+ date.strftime('%Y'))

		#Path of files for date
		files_ofday = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')))
		#Load files in a list of cubes (one cube per file)
		U = iris.load(files_ofday,['U'])		# U at sigma levels (west_east staggered)
		V = iris.load(files_ofday,['V'])		# V at sigma levels (south_north staggered)
		U10 = iris.load(files_ofday,['U10'])	# U at 10 m height
		V10 = iris.load(files_ofday,['V10'])	# V at 10 m height
		PH = iris.load(files_ofday,['PH'])		# perturbation geopotential (bottom_top staggered)
		PHB = iris.load(files_ofday,['PHB'])	# base-state geopotential (bottom_top staggered)

		#Get in one cube all the time steps of the day (from a list of cubes)
		U = concat_list(U)
		V = concat_list(V)
		U10 = concat_list(U10)
		V10 = concat_list(V10)
		PH = concat_list(PH)
		PHB = concat_list(PHB)

		#Rename and order coordinates in cubes
		U = fix_coordinates(U)
		V = fix_coordinates(V)
		U10 = fix_coordinates(U10)
		V10 = fix_coordinates(V10)
		PH = fix_coordinates(PH)
		PHB = fix_coordinates(PHB)

		#Horizontal interpolations
		U_site = U.interpolate(wt_loc, iris.analysis.Linear())
		V_site = V.interpolate(wt_loc, iris.analysis.Linear())
		U10_site = U10.interpolate(wt_loc, iris.analysis.Linear())
		V10_site = V10.interpolate(wt_loc, iris.analysis.Linear())
		PH_site = PH.interpolate(wt_loc, iris.analysis.Linear())
		PHB_site = PHB.interpolate(wt_loc, iris.analysis.Linear())

		#Height of unstaggered levels
		H_levels = unstag_geoheights1D(PH_site,PHB_site)

		#Log interpolation at hub height
		U_hubh = vert_semilog_interp_1D(H_levels,U_site,U10_site,wt_height)
		V_hubh = vert_semilog_interp_1D(H_levels,V_site,V10_site,wt_height)
		WS_hubh = ws_ifunc(U_hubh, V_hubh, new_name='wind_speed')
		WD_hubh = wd_ifunc(U_hubh, V_hubh, new_name='wind_from_direction')

		#Place cubes in a list
		cube_list_ws[it] = WS_hubh
		cube_list_wd[it] = WD_hubh
		it = it + 1

	#Create iris list, unifty time units and concatenate in time
	cube_list_ws = iris.cube.CubeList(cube_list_ws)
	iris.util.unify_time_units(cube_list_ws)
	cube_ws = cube_list_ws.concatenate(cube_list_ws)[0]

	cube_list_wd = iris.cube.CubeList(cube_list_wd)
	iris.util.unify_time_units(cube_list_wd)
	cube_wd = cube_list_wd.concatenate(cube_list_wd)[0]

	#Create dataframes
	df_ws = iris.pandas.as_data_frame(cube_ws)
	df_ws = df_ws.rename(columns = {0:'wind_speed ('+str(cube_ws.units)+')'})
	df_ws.index.name = 'Date'

	df_wd = iris.pandas.as_data_frame(cube_wd)
	df_wd = df_wd.rename(columns = {0:'wind_from_direction ('+str(cube_wd.units)+')'})
	df_wd.index.name = 'Date'

	df = df_ws.join(df_wd)
	print (' << ¡ready! >>\n')
	return df, cube_ws, cube_wd

#Accepts U, V and staggered variables
def extract_2D_cube(wrfdir,date_bgn,date_end,grid,wt_height,variable):
	date_bgn = datetime.datetime.strptime(date_bgn,'%d-%m-%Y').date()
	date_end = datetime.datetime.strptime(date_end,'%d-%m-%Y').date()

	#Print grid and resolution to use
	sample_year = date_bgn.strftime('%Y')
	sample_month = date_bgn.strftime('%m')
	sample_day = date_bgn.strftime('%d')
	sample_file = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, sample_year, sample_month, sample_day))
	sample_cube = iris.load(sample_file,['U'])[0]
	x = int(sample_cube.attributes['DX']/1000)
	y = int(sample_cube.attributes['DY']/1000)
	print (' << Extracting 2D field of '+str(variable)+' using WRF grid ' + str(grid) + ' ('+ str(x) + 'km x ' + str(y) +'km) >>')

	#Extraction routine
	date_range = (date_bgn + datetime.timedelta(n) for n in range((date_end - date_bgn).days + 1)) #Generator
	cube_list = [None] * len(range((date_end - date_bgn).days + 1))
	it = 0
	for date in date_range:
		print('\tReading files for: ' + date.strftime('%d') +'-'+ date.strftime('%m') +'-'+ date.strftime('%Y'))

		#Path of files for date
		files_ofday = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')))

		#Load files in a list of cubes (one cube per file)
		var = iris.load(files_ofday,[variable])		# var at sigma levels (west_east staggered)
		PH = iris.load(files_ofday,['PH'])		# perturbation geopotential (bottom_top staggered)
		PHB = iris.load(files_ofday,['PHB'])	# base-state geopotential (bottom_top staggered)

		#Get in one cube all the time steps of the day (from a list of cubes)
		var = concat_list(var)
		PH = concat_list(PH)
		PHB = concat_list(PHB)

		#Rename and order coordinates in cubes
		var = fix_coordinates(var)
		PH = fix_coordinates(PH)
		PHB = fix_coordinates(PHB)

		#Regrid
		var = var.regrid(PH,iris.analysis.Linear())

		#2D Height of unstaggered levels
		H_levels = unstag_geoheights2D(PH,PHB)

		#Log interpolation at hub height (heights,staggered_variable,height to interp)
		var_hubh = vertinterp_2D(H_levels,var,wt_height)

		#Place cubes in a list
		cube_list[it] = var_hubh
		it = it + 1

	#Create iris list, unifty time units and concatenate in time
	cube_list = iris.cube.CubeList(cube_list)
	iris.util.unify_time_units(cube_list)
	cube = cube_list.concatenate()[0]

	#Fix names
	if variable == 'U':
		cube.rename('x_wind')
		cube.units = 'm s-1'
	if variable == 'V':
		cube.rename('y_wind')
		cube.units = 'm s-1'

	df = iris.pandas.as_data_frame(cube)
	df.index.name = 'Date'

	return df, cube

def pressure_hh(wrfdir,date_bgn,date_end,grid,wt_loc,wt_height):
	date_bgn = datetime.datetime.strptime(date_bgn,'%d-%m-%Y').date()
	date_end = datetime.datetime.strptime(date_end,'%d-%m-%Y').date()

	#Print grid and resolution to interpolate
	sample_year = date_bgn.strftime('%Y')
	sample_month = date_bgn.strftime('%m')
	sample_day = date_bgn.strftime('%d')
	sample_file = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, sample_year, sample_month, sample_day))
	sample_cube = iris.load(sample_file,['U'])[0]
	x = int(sample_cube.attributes['DX']/1000)
	y = int(sample_cube.attributes['DY']/1000)
	print (' << Interpolating pressure for WRF grid ' + str(grid) + ' ('+ str(x) + 'km x ' + str(y) +'km) >>')

	#Interpolation routine
	date_range = (date_bgn + datetime.timedelta(n) for n in range((date_end - date_bgn).days + 1)) #Generator
	cube_list_fp = [None] * len(range((date_end - date_bgn).days + 1))
	it = 0
	for date in date_range:
		print('\tReading files for: ' + date.strftime('%d') +'-'+ date.strftime('%m') +'-'+ date.strftime('%Y'))

		#Path of files for date
		files_ofday = (wrfdir+'/wrfout_d{}_{}-{}-{}_*'.format("%02d" % grid, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')))
		#Load files in a list of cubes (one cube per file)
		P = iris.load(files_ofday,['P'])		# P at sigma levels
		PB = iris.load(files_ofday,['PB'])		# PB at sigma levels
		PH = iris.load(files_ofday,['PH'])		# perturbation geopotential (bottom_top staggered)
		PHB = iris.load(files_ofday,['PHB'])	# base-state geopotential (bottom_top staggered)

		#Get in one cube all the time steps of the day (from a list of cubes)
		P = concat_list(P)
		PB = concat_list(PB)
		PH = concat_list(PH)
		PHB = concat_list(PHB)

		#Rename and order coordinates in cubes
		P = fix_coordinates(P)
		PB = fix_coordinates(PB)
		PH = fix_coordinates(PH)
		PHB = fix_coordinates(PHB)

		#Horizontal interpolations
		P_site = P.interpolate(wt_loc, iris.analysis.Linear())
		PB_site = PB.interpolate(wt_loc, iris.analysis.Linear())
		PH_site = PH.interpolate(wt_loc, iris.analysis.Linear())
		PHB_site = PHB.interpolate(wt_loc, iris.analysis.Linear())

		#Full pressure on site at model levels
		FP_site = P_site + PB_site
		FP_site.rename('air_pressure')

		#Height of unstaggered levels
		H_levels = unstag_geoheights1D(PH_site,PHB_site)

		#Linear interpolation at hub height
		FP_hh = vert_linear_interp_1D(H_levels,FP_site,wt_height)

		#Place cubes in a list
		cube_list_fp[it] = FP_hh
		it = it + 1

	#Create iris list, unifty time units and concatenate in time
	cube_list_fp = iris.cube.CubeList(cube_list_fp)
	iris.util.unify_time_units(cube_list_fp)
	cube_fp = cube_list_fp.concatenate(cube_list_fp)[0]

	#Create dataframes
	df = iris.pandas.as_data_frame(cube_fp)
	df = df.rename(columns = {0:'air_pressure ('+str(cube_fp.units)+')'})
	df.index.name = 'Date'

	print (' << ¡ready! >>\n')
	return df, cube_fp

def main(args):
	variables = ['wind_speed','air_pressure']
	places = ['ojuelos','arriaga','tepexi','cdcuauhtemoc','larumorosa','merida','certe','sanfernando','puntadelgada','losmochis']
	txt = '"Extract and interpolate variables from WRF." \n-> For wind speed, info is extracted and then interpolated to a horizontal location and to a vertical location. \n-> Bilinear for horizontal interpolation (lat,lon) and logartihmic for vertical interpolation (model level numbers). \n-> Gives speed and direction \n\n-> For pressure is linear vertical interpolation and bilinear horizontal'

	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=txt,prog='gusiWRF')

	parser.add_argument('-c', '--csv', help='export to csv', action='store_true')
	parser.add_argument('-n', '--netcdf', help='export to netcdf', action='store_true')
	parser.add_argument('-fn', '--filename', help='name for output')

	req_group = parser.add_argument_group(title='required arguments')
	req_group.add_argument('-d', '--dir', help='path to wrfouts', required=True)
	req_group.add_argument('-s', '--start', help='start date (dd-mm-yyyy)', required=True)
	req_group.add_argument('-e', '--end', help='end date (dd-mm-yyyy)', required=True)
	req_group.add_argument('-g', '--grid', help='number of WRF grid', required=True)
	req_group.add_argument('-l', '--loc', help='location for horizontal interpolation', required=True, choices=places)
	req_group.add_argument('-ht', '--height', help='height for vertical interpolation', required=True)
	req_group.add_argument('-v', '--var', help='variables to extract', required=True, choices=variables)#, nargs='+')

	args = parser.parse_args()

	wrfdir = vars(args)['dir']
	date_bgn = vars(args)['start']
	date_end = vars(args)['end']
	grid = int(vars(args)['grid'])
	wt_height = int(vars(args)['height'])
	filename = vars(args)['filename']
	loc = vars(args)['loc']
	variables = vars(args)['var']

	if variables == 'wind_speed':
		df, cube_ws, cube_wd = extract_hh_cube(wrfdir,date_bgn,date_end,grid,wt_loc,wt_height)

		if vars(args)['csv']:
			print(' << Exporting to csv >>\n')
			df.to_csv(str(filename)+'.csv')

		if vars(args)['netcdf']:
			print('Exporting to netcdf >>>'+str(filename)+'.nc \n')
			iris.save([cube_ws,cube_wd], str(filename)+'.nc', zlib=True, complevel=9, shuffle=True)

		return cube_ws, cube_wd

	elif variables == 'air_pressure':
		df, cube_fp = pressure_hh(wrfdir,date_bgn,date_end,grid,wt_loc,wt_height)

		if vars(args)['csv']:
			print(' << Exporting to csv >>\n')
			df.to_csv(str(filename)+'.csv')

		if vars(args)['netcdf']:
			print('Exporting to netcdf >>>'+str(filename)+'.nc \n')
			iris.save(cube_fp, str(filename)+'.nc', zlib=True, complevel=9, shuffle=True)

		return cube_fp
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
