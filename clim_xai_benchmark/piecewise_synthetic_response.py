import numpy as np
import xarray as xr
from . import surrogate_data_generator as sdg

def gen_random_quantiles(data, n_breaking_points, feature_dim = "feature", breaking_dim = "breaking_point"):
    """Generates a specified number of random quantile values for a given dataset over a given dimension 

    Args:
        data (xarray Dataarray): Data over which quantiles will be evaluated
        n_breaking_points (int): Number of breaking points
        feature_dim (str, optional): Dimension of dataarray that corresponds to features. Defaults to "feature".
        breaking_dim (str, optional): Name of the corresponding quantile dimension. Defaults to "breaking_point".

    Returns:
        [xarray Dataarray]: quantiles  
    """
    
    n_feature = data.sizes[feature_dim]
    
    quantiles = np.random.uniform(0,1, size = [n_breaking_points, n_feature])

    quantiles = xr.DataArray(quantiles, dims = [breaking_dim, feature_dim], coords = {breaking_dim: range(n_breaking_points), feature_dim:data.coords[feature_dim]})
    return quantiles

def cal_breaking_points_from_quantiles(data, quantiles, feature_dim ="feature", sample_dim = "sample", breaking_dim ="breaking_point"):
    """Calculates the breaking points of a data array over a given sample dimension given quantile array

    Args:
        data (xarray DataArray): Data over which quantiles are evaluated 
        quantiles (xarray DataArray): Dataarray of the quantiles  
        feature_dim (str, optional): Name of the feature dimension. Defaults to "feature".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".
        breaking_dim (str, optional): Name of the breaking dimension. Defaults to "breaking_point".

    Returns:
        [xarray DataArray]: DataArray without the sample dimension but instead a breaking_point dimension
    """
    
    arr = []
    n_feature = data.sizes[feature_dim]
    n_breaking_points = quantiles.sizes[breaking_dim]

    for feature_index in range(n_feature):
        data_tmp = data.isel(feature = feature_index)
        quantile_tmp = quantiles.isel(feature=feature_index)

        quantiles_xr = data_tmp.quantile(quantile_tmp, dim=sample_dim)
        quantiles_xr = quantiles_xr.assign_coords(quantile = range(n_breaking_points))
        quantiles_xr = quantiles_xr.rename({"quantile": breaking_dim})
        arr.append(quantiles_xr)
    breaking_points = xr.concat(arr, dim=feature_dim)
    breaking_points = breaking_points.assign_coords({feature_dim:data.coords[feature_dim]})

    return breaking_points


def add_zero_breaking_point(breaking_points, feature_dim = "feature", breaking_dim = "breaking_point" ):
    """Given a set of breaking points this function adds an additional zero as a breaking point

    Args:
        breaking_points (xarray DataArray): 
        feature_dim (str, optional): Name of the feature dimension. Defaults to "feature".
        breaking_dim (str, optional): Name of the breaking point dimension. Defaults to "breaking_point".

    Returns:
        [xarray DataArray]: breaking points with an additional zero added
    """

    n_feature   = breaking_points.sizes[feature_dim]
    n_breaking_points = breaking_points.sizes[breaking_dim]

    breaking_zero = xr.DataArray(np.zeros([1, n_feature]), dims = [breaking_dim, feature_dim], coords = {breaking_dim:[n_breaking_points], feature_dim:breaking_points.coords[feature_dim]})

    breaking_points = xr.concat([breaking_zero, breaking_points], dim=breaking_dim)
    return breaking_points


def xarray_sort_reindex(data, sort_dim):
    """ Sorts an Dataarray over a given dimension and reindexes this dimension with a standard range 
    Args:
        data (xarray DataArray): Data to be sorted and reindexed 
        sort_dim (str): Name of the dimension over which to sort and reindex

    Returns:
        [xarray DataArray]: Sorted and reindexed Data
    """
    axis_num = data.get_axis_num(sort_dim)
    n_sort_dim = data.sizes[sort_dim]

    data_sorted = np.sort(data, axis=axis_num)   
    data_sorted = xr.DataArray(data_sorted, coords = data.coords).assign_coords(breaking_point = range(n_sort_dim))
    return data_sorted
 
def cal_breaking_points_y(breaking_points, slopes, breaking_dim = "breaking_point"):
    """Given a number of breaking points and slopes this function calculates the corresponding y values of a piecewise linear function

    Args:
        breaking_points (xarray DataArray): Array containing the breaking points
        slopes (xarray DataArray): Array containing the slopes
        breaking_dim (str, optional): Name of breaking_point dimension. Defaults to "breaking_point".

    Returns:
        [xarray Dataarray]: y value for braking_points 
    """
    n_breaking_points = breaking_points.sizes[breaking_dim]

    y= []
    y.append(xr.zeros_like(breaking_points).isel({breaking_dim:0}))


    for sta_index in range(n_breaking_points-1):
        end_index = sta_index+1
        logging.info(" ".join([str(sta_index),str(end_index)]))
        x_end = breaking_points.sel({breaking_dim:end_index})
        x_sta = breaking_points.sel({breaking_dim:sta_index})
        

        slope = slopes.sel({breaking_dim:sta_index}).reset_coords(drop=True)
        linear = (np.multiply(x_end-x_sta,slope) + y[sta_index]).assign_coords({breaking_dim:end_index})
        y.append(linear)

    data = xr.concat(y, dim= breaking_dim)
    
    # shift the dataarray by the y value at x = 0 to enforce that at x=0, y=0
    data = data - data.sel(breaking_point = np.abs(breaking_points).argmin(dim=breaking_dim))

    return data

def piecewise_combined(x, breaking_point_x, breaking_point_y, slope):
    """ Calculates the output over x of a piecewise linear function with given x and y breaking points as well as slopes

    Args:
        x (list of float): input values for function
        breaking_point_x (list of float): values of breaking points in x
        breaking_point_y (list of float): values of breaking points in y
        slope (list of float): values of slopes  

    Returns:
        [list of float]: output for piecewise linear function
    """
    value = np.nan

    condlist = []
    funclist = []
        
    cond = [(x < breaking_point_x[0])]
    func = [lambda k: breaking_point_y[0] + slope[0]*(k-breaking_point_x[0]),0]
    
    function = np.piecewise(x, cond, func)


    for i in range(0,len(breaking_point_x)-1):
        cond =  (  (x >= breaking_point_x[i]) * (x < breaking_point_x[i+1]))
        func =  [lambda k: breaking_point_y[i] + slope[i+1]*(k-breaking_point_x[i]),0]
        function = function+ np.piecewise(x, cond, func)


    
    
    cond = [x >= breaking_point_x[-1]]
    func =  [lambda k: breaking_point_y[-1] + slope[-1]*(k -breaking_point_x[-1]),0]
    function = function + np.piecewise(x,cond, func)

    return function



def gen_piecewise_linear_parameters(data, covariance,  n_breaking_points = 1, breaking_dim ="breaking_point", feature_dim = "feature", sample_dim= "sample"):
    """ Generates breaking points and slopes for a given dataarray and covariance matrix

    Args:
        data (xarray DataArray): Data for which piecewise parameters are generated
        covariance (xarray DataArray): Covariance matrix for the data
        n_breaking_points (int, optional): Number of breaking points in the output. One breaking point is always zero. Defaults to 1.
        breaking_dim (str, optional): Name of the dimension for the breaking points. Defaults to "breaking_point".
        feature_dim (str, optional): Name of the feature dimension. Defaults to "feature".
        sample_dim (str, optional): Name of the sample dimension. Defaults to "sample".

    Returns:
        [xarray DataSet]: Dataset containing breaking points x and y coordinates as well as slopes 
    """

    quantiles = gen_random_quantiles(data = data, n_breaking_points = 0)
    
    breaking_points_x = cal_breaking_points_from_quantiles(data= data, quantiles = quantiles, feature_dim = feature_dim, sample_dim = sample_dim)
    breaking_points_x_zero = add_zero_breaking_point(breaking_points_x)
    breaking_points_x_zero_sorted = xarray_sort_reindex(data = breaking_points_x_zero, sort_dim = breaking_dim)

    slopes = sdg.xarray_multivariate_normal_zeromean(covariance = covariance, n_sample = n_breaking_points +2, feature_dim= feature_dim, sample_dim = breaking_dim)
    slopes = slopes.assign_coords({breaking_dim:np.arange(-1,n_breaking_points+1)})
    breaking_points_y =cal_breaking_points_y(breaking_points_x_zero_sorted, slopes)


    breaking_points_merged = xr.merge([breaking_points_x_zero_sorted.rename("x"), breaking_points_y.rename("y"), slopes.rename("slopes")])
    return breaking_points_merged


def cal_output_linear_piecewise(data, breaking_points, breaking_dim ="breaking_point", sample_dim="sample", feature_dim ="feature"):
    arr = []
    n_feature = data.sizes[feature_dim]

    for feature_index in range(n_feature):
        input_data = data.isel(feature=feature_index)
        x_start = breaking_points.isel(feature=feature_index).y.dropna(dim = breaking_dim)
        y_start = breaking_points.isel(feature=feature_index).y.dropna(dim = breaking_dim)
        slope =   breaking_points.isel(feature=feature_index).slopes

        tmp_data = piecewise_combined(
        input_data.values,
        x_start.values,
        y_start.values,
        slope.values)

        tmp_data = xr.DataArray(tmp_data, dims=[sample_dim], coords = {sample_dim:input_data.coords[sample_dim]})
        tmp_data = tmp_data.assign_coords({feature_dim:feature_index})
        arr.append(tmp_data)


    result = xr.concat(arr, dim = feature_dim)
    result = result.assign_coords({feature_dim:data.coords[feature_dim]})
    return result

