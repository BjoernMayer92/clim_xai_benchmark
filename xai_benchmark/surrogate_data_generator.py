import numpy as np
import xarray as xrange


def xarray_autocovariance_matrix(data, sample_dimension = "time"):

    data_dimensions = list(data.dims)

    print(data_dimensions)
    if not sample_dimension in data_dimensions:
        raise ValueError(" ".join(sample_dimension, " not in dimensions ", str(data_dimension)))
    else:
        data_dimensions.remove(sample_dimension)

    rename_dict = {}
    for data_dimension in data_dimensions:
        rename_dict[data_dimension] = data_dimension+"_1"

    covariance = xr.cov(data, data.rename(rename_dict), dim=sample_dimension)

    return covariance


def xarray_multivariate_normal_distribution(mean, covariance, n_sample, feature_dim = "location",  sample_dim = "sample"):

    data_surrogate = np.random.multivariate_normal(mean, covariance, size = n_sample)
    
    data_surrogate = xr.DataArray(data_surrogate, dims  = [sample_dim, feature_dim], coords = {sample_dim:range(n_sample),
    feature_dim: mean.coords[feature_dim]})

    return data_surrogate


def xarray_multivariate_normal_zeromean(covariance, n_sample, feature_dim = "feature", sample_dim = "sample"):
    
    n_feature = covariance.sizes[feature_dim]
    mean = np.zeros(n_feature)
    mean = xr.DataArray(mean, dims = [feature_dim], coords = {feature_dim: covariance.coords[feature_dim]})

    data = xarray_multivariate_normal_distribution(mean, covariance, n_sample, feature_dim = feature_dim,  sample_dim = sample_dim)

    return data

