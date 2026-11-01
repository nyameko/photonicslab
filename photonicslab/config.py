# -*- coding: utf-8 -*-
#!/usr/bin/env python3

MAX_NT = 1000
MAX_NX = 2000
MAX_NY = 2000
MAX_NZ = 1000
MAX_DPI = 600

# module decorator
import time
def time_it(func):
    """Timing decorator."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(func.__name__ + " took " + str((end-start)) + " seconds.")
        return result
    return wrapper

import h5py
def get_hdf5_data(hdf5filename, hdf5grp):
    with h5py.File(str(hdf5filename), 'a', libver='latest') as hf:
        grp = hf.require_group(str(hdf5grp))
        nt = grp.attrs.get('nt')
        nx = grp.attrs.get('nx')
        ny = grp.attrs.get('ny')
        nz = grp.attrs.get('nz')
        Lx = grp.attrs.get('Lx')
        Ly = grp.attrs.get('Ly')
        Lz = grp.attrs.get('Lz')
        return [nt, nx, ny, nz, Lx, Ly, Lz]

__all__ = ['MAX_NT', 'MAX_NX', 'MAX_NY', 'MAX_NZ', 'MAX_DPI', 'time_it']
