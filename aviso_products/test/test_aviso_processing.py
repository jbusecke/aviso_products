from __future__ import print_function
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from aviso_products.aviso_processing import merge_aviso


@pytest.fixture(scope='session')
def vel_dir(tmpdir_factory):
    # Create test arrays
    u_dt = np.array([
                [[np.nan, 1.0],
                 [1.0, 1.0]],
                [[np.nan, 2.0],
                 [2.0, 2.0]],
                [[np.nan, 3.0],
                 [3.0, 3.0]]
                ])

    v_dt = -np.array([
                    [[np.nan, 1.0],
                     [1.0, 1.0]],
                    [[np.nan, 2.0],
                     [2.0, 2.0]],
                    [[np.nan, 3.0],
                     [3.0, 3.0]]
                    ])

    u_nrt = np.array([
                    [[np.nan, 10.0],
                     [10.0, 10.0]],
                    [[np.nan, 20.0],
                     [20.0, 20.0]]
                    ])

    v_nrt = -np.array([
                    [[np.nan, 10.0],
                     [10.0, 10.0]],
                    [[np.nan, 20.0],
                     [20.0, 20.0]]
                    ])
    vel_dir = tmpdir_factory.mktemp('vel_test')

    lon = range(0, 2)
    lat = range(200, 202)

    for tt in range(u_dt.shape[0]):
        file = str(vel_dir.join('dt_'+str(tt)+'.nc'))
        u = u_dt[tt, :, :]
        v = v_dt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (u_dt.shape[0]-tt), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_dt.shape[0]):
        file = str(vel_dir.join('missing_dt_'+str(tt)+'.nc'))
        u = u_dt[tt, :, :]
        v = v_dt[tt, :, :]
        # introduce a 'gap'
        if tt != 0:
            tt = tt + 1
        time = pd.date_range('2000-01-%02i' % (tt+1), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_nrt.shape[0]):
        file = str(vel_dir.join('nrt_'+str(tt)+'.nc'))
        u = u_nrt[tt, :, :]
        v = v_nrt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (tt+3), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_nrt.shape[0]):
        file = str(vel_dir.join('missing_nrt_'+str(tt)+'.nc'))
        u = u_nrt[tt, :, :]
        v = v_nrt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (tt+5), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)
    return vel_dir


def test_merge_aviso(vel_dir):
    ddir = str(vel_dir)

    ds_dt, sd_dt, td_dt = merge_aviso(ddir,
                                      fid_dt='dt*.nc',
                                      ddir_nrt=None)
    ds_dt_check = xr.open_mfdataset(ddir+'/dt*.nc').sortby('time')
    xr.testing.assert_allclose(ds_dt, ds_dt_check)
    assert sd_dt == np.datetime64('2000-01-01')
    assert td_dt is None
    print(ds_dt.chunks)
    assert all([x == 1 for x in list(ds_dt.chunks['time'])])

    ds_nrt, sd_nrt, td_nrt = merge_aviso(ddir,
                                         fid_dt='dt*.nc',
                                         ddir_nrt=ddir,
                                         fid_nrt='nrt_*.nc')
    check_time = slice('2000-01-04', None)
    ds_nrt_check = xr.merge([xr.open_mfdataset(ddir+'/dt*.nc'),
                            xr.open_mfdataset(ddir+'/nrt*.nc').
                            sel(time=check_time)]).sortby('time')

    xr.testing.assert_allclose(ds_nrt, ds_nrt_check)
    assert sd_nrt == np.datetime64('2000-01-01')
    assert td_nrt == np.datetime64('2000-01-03')

    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='dt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='missing_nrt_*.nc')
        assert 'Time steps are not homogeneous' in excinfo.value.message

    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='missing_dt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='nrt_*.nc')
