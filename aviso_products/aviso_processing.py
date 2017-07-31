import numpy as np
import xarray as xr
import os
from astropy.convolution import Gaussian2DKernel, convolve
from dask.diagnostics import ProgressBar
from datetime import datetime


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


def filter_aviso(fname, stddev, time_subsample=1):
    ds = xr.open_mfdataset(fname, engine='scipy')
    ds = ds.chunk({'time': 1})

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


def write_yearly_files(ds, odir, fname, verbose=False):
    if not os.path.exists(odir):
        os.mkdir(odir)

    '''writes out yearly .nc files from xarray dataset to odir'''
    years, datasets = zip(*ds.groupby('time.year'))
    paths = [os.path.join(odir, fname+'_%s.nc') % y
             for y in years]
    if verbose:
        print('Writing dataset to '+odir)
    with ProgressBar():
        xr.save_mfdataset(datasets, paths, engine='netcdf4')