import numpy as np
import xarray as xr
import os
from astropy.convolution import Gaussian2DKernel, convolve
from dask.diagnostics import ProgressBar
from datetime import datetime
from xarrayutils import aggregate


def merge_aviso(ddir_dt,
                fid_dt='dt_global_allsat_msla_uv_*.nc',
                ddir_nrt=None,
                fid_nrt='nrt_global_allsat_msla_uv_*.nc',
                engine='scipy'):

    """read aviso files into xarray dataset
    This function merges delayed-time and near-real time products if optional
    near-real time parameters are given.

    PARAMETERS
    ----------
    ddir_dt : path
        data directory for delayed time product
    fid_dt : str
        string pattern identifying delayed time products
        (default:'dt_global_allsat_msla_uv_*.nc')
    ddir_dt : path
        data directory for near-real time product
        (default: None)
    fid_dt : str
        string pattern identifying near-real time products
        (default:nrt_global_allsat_msla_uv_*.nc')

    RETURNS
    -------
    ds : xarray.Dataset
        combined Aviso dataset
    start_date : datetime
        date of first aviso data
    transition_date : datetime
        date when data switches from delayed-time and near-real time
    """
    ds_dt = xr.open_mfdataset(ddir_dt+'/'+fid_dt,
                              engine=engine).sortby('time')
    if ddir_nrt is not None:
        transition_date = ds_dt.time.isel(time=-1)
        ds_nrt = xr.open_mfdataset(ddir_nrt+'/'+fid_nrt,
                                   engine=engine).sortby('time')
        ds = xr.concat((ds_dt,
                        ds_nrt.isel(time=ds_nrt.time > transition_date)),
                       dim='time')
    else:
        ds = ds_dt
        transition_date = None

    # Test if time is continous
    if np.any(ds.time.diff('time').data != ds.time.diff('time')[0].data):
        raise RuntimeError('Time steps are not homogeneous. Likely missing \
        files between the dt and nrt products')

    start_date = ds.time[0].data
    ds = ds.chunk({'time': 1})
    ds.attrs['start_date'] = start_date
    ds.attrs['transition_date'] = transition_date

    return ds


def high_pass_filter(np_ar, stddev):
    gaussian_kernel = Gaussian2DKernel(stddev=stddev)
    if (np_ar.ndim > 2) and (np_ar.shape[0] > 1):
        out = np.zeros_like(np_ar)
        for k in xrange(np_ar.shape[0]):
            out[k] = convolve(np_ar[k], gaussian_kernel, boundary='wrap')
        return np_ar - out
    elif (np_ar.ndim > 2):
        return np_ar - convolve(np_ar.squeeze(),
                                gaussian_kernel,
                                boundary='wrap')[np.newaxis, :, :]
    else:
        return np_ar - convolve(np_ar, gaussian_kernel, boundary='wrap')


def filter_aviso(ds, stddev, time_subsample=1):

    u = ds['u'][::time_subsample]
    v = ds['v'][::time_subsample]

    ufilt = u.data.map_blocks(high_pass_filter,
                              dtype=np.float64,
                              stddev=stddev)
    vfilt = v.data.map_blocks(high_pass_filter,
                              dtype=np.float64,
                              stddev=stddev)

    filtered_ds = xr.Dataset({'u': (u.dims, ufilt, u.attrs),
                              'v': (u.dims, vfilt, v.attrs)},
                             coords=u.coords)

    # TODO: Adjust these attrs in case some other product is used...(e.g
    # a combo of dt and nrt)
    filtered_ds.attrs['title'] = 'Spatially Filtered Global Ocean Surface \
        Geostrophic Velocities'
    filtered_ds.attrs['institution'] = 'Lamont Doherty Earth Observatory'
    filtered_ds.attrs['source'] = 'Processed Satellite Observations'
    filtered_ds.attrs['comment'] = ("Derived from AVISO DT merged Global \
    Ocean" "Gridded Geostrophic Velocities SSALTO/Duacs L4 product")
    filtered_ds.attrs['history'] = datetime.now().strftime('%F') + " created"
    return filtered_ds


def write_yearly_files(ds, odir, fname, verbose=False, engine='netcdf4'):
    if not os.path.exists(odir):
        os.mkdir(odir)

    '''writes out yearly .nc files from xarray dataset to odir'''
    years, datasets = zip(*ds.groupby('time.year'))
    paths = [os.path.join(odir, fname+'_%s.nc') % y
             for y in years]
    if verbose:
        print('Writing dataset to '+odir)
    with ProgressBar():
        xr.save_mfdataset(datasets, paths, engine=engine)


def calculate_eke(ds, remove_seas=True):
    ''' Calculate EKE from aviso '''
    # remove seasonal cycle from
    if remove_seas:
        with ProgressBar():
            print('Calculating Seasonal cycle for EKE')
            seas_clim = ds.groupby('time.month').mean().compute()
        ds = ds.groupby('time.month')-seas_clim
    eke = 0.5*(ds.u**2 + ds.v**2)
    eke = eke.resample('MS', 'time')
    return eke
