#!/usr/bin/env python3

import os
import psutil
import segyio

import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from segysak import segy

from dasf.utils.types import is_dask_array


def compute_chunk_size(shape, byte_size, kernel=None, preview=None):
    """
    Description
    -----------
    Compute ideal block size for Dask Array given specific information about
    the computer being used, the input data, kernel size, and whether or not
    this operation is is 'preview' mode.
    Parameters
    ----------
    shape : tuple (len 3), shape of seismic data
    byte_size : int, byte size of seismic data dtype
    Keywork Arguments
    -----------------
    kernel : tuple (len 3), operator size
    preview : str, enables or disables preview mode and specifies direction
        Acceptable inputs are (None, 'inline', 'xline', 'z')
        Optimizes chunk size in different orientations to facilitate rapid
        screening of algorithm output
    Returns
    -------
    chunk_size : tuple (len 3), optimal chunk size
    """

    # Evaluate kernel
    if kernel is None:
        kernel = (1, 1, 1)
        ki, kj, kk = kernel
    else:
        ki, kj, kk = kernel

    # Identify acceptable chunk sizes
    i_s = np.arange(ki, shape[0])
    j_s = np.arange(kj, shape[1])
    k_s = np.arange(kk, shape[2])

    modi = shape[0] % i_s
    modj = shape[1] % j_s
    modk = shape[2] % k_s

    kki = i_s[(modi >= ki) | (modi == 0)]
    kkj = j_s[(modj >= kj) | (modj == 0)]
    kkk = k_s[(modk >= kk) | (modk == 0)]

    # Compute Machine Specific information
    mem = psutil.virtual_memory().available
    cpus = psutil.cpu_count()
    byte_size = byte_size
    M = ((mem / (cpus * byte_size)) / (ki * kj * kk)) * 0.75

    # Compute chunk size if preview mode is disabled
    if preview is None:
        # M *= 0.3
        Mij = kki * kkj.reshape(-1, 1) * shape[2]
        Mij[Mij > M] = -1
        Mij = Mij.diagonal()

        chunks = [kki[Mij.argmax()], kkj[Mij.argmax()], shape[2]]

    # Compute chunk size if preview mode is enabled
    else:
        kki = kki.min()
        kkj = kkj.min()
        kkk = kkk.min()

        if preview == 'inline':
            if (kki * shape[1] * shape[2]) < M:
                chunks = [kki, shape[1], shape[2]]

            else:
                j_s = np.arange(kkj, shape[1])
                modj = shape[1] % j_s
                kkj = j_s[(modj >= kj) | (modj == 0)]
                Mj = j_s * kki * shape[2]
                Mj = Mj[Mj < M]
                chunks = [kki, Mj.argmax(), shape[2]]

        elif preview == 'xline':
            if (kkj * shape[0] * shape[2]) < M:
                chunks = [shape[0], kkj, shape[2]]

            else:
                i_s = np.arange(kki, shape[0])
                modi = shape[0] % i_s
                kki = i_s[(modi >= ki) | (modi == 0)]
                Mi = i_s * kkj * shape[2]
                Mi = Mi[Mi < M]
                chunks = [Mi.argmax(), kkj, shape[2]]

        else:
            if (kkk * shape[0] * shape[1]) < M:
                chunks = [shape[0], shape[2], kk]
            else:
                j_s = np.arange(kkj, shape[1])
                modj = shape[1] % j_s
                kkj = j_s[(modj >= kj) | (modj == 0)]
                Mj = j_s * kkk * shape[0]
                Mj = Mj[Mj < M]
                chunks = [shape[0], Mj.argmax(), kkk]

    return tuple(chunks)


def create_array(darray,
                 kernel=None,
                 hw=None,
                 boundary='reflect',
                 preview=None):
    """
    Description
    -----------
    Convert input to Dask Array with ideal chunk size as necessary. Perform
    necessary ghosting as needed for opertations utilizing windowed
    functions.
    Parameters
    ----------
    darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
        Arrays
    Keywork Arguments
    -----------------
    kernel : tuple (len 3), operator size
    hw : tuple (len 3), height and width sizes
    boundary : str, indicates data reflection between data chunks
        For further reference see Dask Overlaping options.
    preview : str, enables or disables preview mode and specifies direction
        Acceptable inputs are (None, 'inline', 'xline', 'z')
        Optimizes chunk size in different orientations to facilitate rapid
        screening of algorithm output
    Returns
    -------
    darray : Dask Array
    chunk_init : tuple (len 3), chunk size before ghosting. Used in select
        cases
    """

    # Compute chunk size and convert if not a Dask Array
    if not is_dask_array(darray):
        chunk_size = compute_chunk_size(darray.shape,
                                        darray.dtype.itemsize,
                                        kernel=kernel,
                                        preview=preview)
        darray = da.from_array(darray, chunks=chunk_size)

    # Ghost Dask Array if operation specifies a kernel
    if kernel is not None:
        if hw is None:
            hw = tuple(np.array(kernel) // 2)
        darray = da.overlap.overlap(darray, depth=hw, boundary=boundary)

    chunks_init = darray.chunks

    return (darray, chunks_init)


def trim_dask_array(in_data, kernel, hw=None, boundary='reflect'):
    """
    Description
    -----------
    Trim resuling Dask Array given a specified kernel size
    Parameters
    ----------
    in_data : Dask Array
    kernel : tuple (len 3), operator size
    Returns
    -------
    out : Dask Array
    """

    # Compute half windows and assign to dict
    if hw is None:
        hw = tuple(np.array(kernel) // 2)
    axes = {0: hw[0], 1: hw[1], 2: hw[2]}

    return da.overlap.trim_internal(in_data, axes=axes, boundary=boundary)


def extract_patches(in_data, kernel, xp):
    """
    Description
    -----------
    Reshape in_data into a collection of patches defined by kernel
    Parameters
    ----------
    in_data : Dask Array, data to convert
    kernel : tuple (len 3), operator size
    Returns
    -------
    out : Numpy Array, has shape (in_data.shape[0], in_data.shape[1],
                                  in_data.shape[2], kernel[0], kernel[1],
                                  kernel[2])
    """

    # This is a workaround for cases where dask chunks are empty
    # Numpy handles if quietly, CuPy does not.
    if in_data.shape == (0, 0, 0):
        return []

    padding = np.array(kernel) // 2
    patches = xp.pad(in_data, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), mode='symmetric')
    strides = patches.strides + patches.strides
    shape = tuple(list(patches.shape) + list(kernel))

    patches = xp.lib.stride_tricks.as_strided(patches,
                                              shape=shape,
                                              strides=strides)
    shape = in_data.shape
    patches = patches[:shape[0], :shape[1], :shape[2]]

    return patches


def local_events(in_data, comparator, is_cupy=False):
    """
    Description
    -----------
    Find local peaks or troughs depending on comparator used
    Parameters
    ----------
    in_data : Dask Array, data to convert
    comparator : function, defines truth between neighboring elements
    is_cupy : handles data directly from GPU
    Returns
    -------
    out : Numpy Array
    """

    if is_cupy:
        idx = cp.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1)
        plus = in_data.take(idx + 1, axis=-1)
        minus = in_data.take(idx - 1, axis=-1)
        plus[:,:,-1] = trace[:,:,-1]
        minus[:,:,0] = trace[:,:,0]
        result = cp.ones(in_data.shape, dtype=bool)
    else:
        idx = np.arange(0, in_data.shape[-1])
        trace = in_data.take(idx, axis=-1, mode='clip')
        plus = in_data.take(idx + 1, axis=-1, mode='clip')
        minus = in_data.take(idx - 1, axis=-1, mode='clip')

        result = np.ones(in_data.shape, dtype=bool)

    result &= comparator(trace, plus)
    result &= comparator(trace, minus)

    return result


def convert_to_seisnc(segyin, iline=189, xline=193, cdpx=181, cdpy=185):
    directory = os.path.dirname(segyin)

    os.makedirs(directory, exist_ok=True)

    seisnc_file = os.path.splitext(os.path.basename(segyin))[0] + ".seisnc"

    seisnc_path = os.path.join(directory, seisnc_file)

    if os.path.exists(seisnc_path):
        return seisnc_path

    segy.segy_converter(
        segyin, seisnc_path, iline=iline, xline=xline, cdpx=cdpx, cdpy=cdpy
    )

    return seisnc_path


# XXX: Function map_segy needs to open locally the file due to problems of
# serialization when dask transports the segyio object through workers.
def map_segy(x, tmp, contiguous, xp, mode='r', iline=189, xline=193,
             strict=True, ignore_geometry=False, endian='big',
             block_info=None):
    segyfile = segyio.open(tmp, mode=mode, iline=iline, xline=xline,
                           ignore_geometry=ignore_geometry, strict=strict,
                           endian=endian)

    if contiguous:
        loc = block_info[None]['array-location'][0]
        return segyfile.trace.raw[loc[0]:loc[1]]
    else:
        dim_x, dim_y, dim_z = block_info[None]['shape']
        loc_x, loc_y, loc_z = block_info[None]['array-location']
        subcube_x = []
        for i in range(loc_x[0], loc_x[1]):
            subcube_y = []
            for j in range(loc_y[0], loc_y[1]):
                block = segyfile.trace.raw[dim_y * i + j][loc_z[0]:loc_z[1]]
                subcube_y.append(block)
            subcube_x.append(subcube_y)

        return xp.asarray(subcube_x).astype(xp.float64)


def inf_to_max_value(array, xp):
    if array.dtype == xp.float64 or array.dtype == xp.float32:
        return np.finfo(array.dtype).max
    elif array.dtype == xp.int64 or array.dtype == xp.int32:
        return np.iinfo(array.dtype).max


def inf_to_min_value(array, xp):
    if array.dtype == xp.float64 or array.dtype == xp.float32:
        return np.finfo(array.dtype).min
    elif array.dtype == xp.int64 or array.dtype == xp.int32:
        return np.iinfo(array.dtype).min


def set_time_chunk_overlap(dask_array):
    if dask_array.shape[-1] != dask_array.chunksize[-1]:
        print("WARNING: splitting the time axis in chunks can cause significant performance degradation.")

        time_edge = int(dask_array.chunksize[-1] * 0.1)
        if time_edge < 5:
            time_edge = dask_array.chunksize[-1] * 0.5

        return (1, 1, int(time_edge))
    return None


def dask_cupy_angle_wrapper(data):
    return data.map_blocks(cp.angle, dtype=data.dtype,
                           meta=cp.array((), dtype=data.dtype))


def matching_dtypes(src_dtype, target_dtype, default):
    dtypes = {
        "float32": {
            "int": "int32",
            "complex": "complex64",
        },
        "float64": {
            "int": "int64",
            "complex": "complex128",
        }
    }

    return dtypes.get(str(src_dtype), {}).get(target_dtype, default)
