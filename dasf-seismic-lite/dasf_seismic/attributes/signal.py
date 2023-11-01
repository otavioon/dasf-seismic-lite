#!/usr/bin/env python3

import numpy as np

import dask.array as da

from scipy import ndimage as ndi
from scipy import signal

try:
    import cupy as cp
    if cp.cuda.runtime.runtimeGetVersion() >= 12000:
        import cupyx.scipy.signal as cusignal
    else:
        import cusignal

    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass

from dasf.transforms import Transform

from dasf_seismic.utils.utils import create_array
from dasf_seismic.utils.utils import trim_dask_array
from dasf_seismic.utils.utils import extract_patches
from dasf_seismic.utils.utils import set_time_chunk_overlap


class FirstDerivative(Transform):
    def __init__(self, axis=-1, preview=None):
        super().__init__()

        self._axis = axis
        self._preview = preview

    def __lazy_transform(self, X, xndi, xp):
        kernel = (3, 3, 3)
        axes = [ax for ax in range(X.ndim) if ax != self._axis]
        X, chunks_init = create_array(X, kernel, preview=self._preview)

        result0 = X.map_blocks(xndi.correlate1d,
                               weights=xp.array([-0.5, 0.0, 0.5]),
                               axis=self._axis, dtype=X.dtype,
                               meta=xp.array((), dtype=X.dtype))
        result1 = result0.map_blocks(xndi.correlate1d,
                                     weights=xp.array([0.178947, 0.642105,
                                                       0.178947]),
                                     axis=axes[0], dtype=X.dtype,
                                     meta=xp.array((), dtype=X.dtype))
        result2 = result1.map_blocks(xndi.correlate1d,
                                     weights=xp.array([0.178947, 0.642105,
                                                       0.178947]),
                                     axis=axes[1], dtype=X.dtype,
                                     meta=xp.array((), dtype=X.dtype))

        return trim_dask_array(result2, kernel)

    def __transform(self, X, xndi, xp):
        axes = [ax for ax in range(X.ndim) if ax != self._axis]

        result0 = xndi.correlate1d(X, weights=xp.array([-0.5, 0.0, 0.5]),
                                   axis=self._axis)

        result1 = xndi.correlate1d(result0,
                                   weights=xp.array([0.178947, 0.642105,
                                                     0.178947]),
                                   axis=axes[0])

        result2 = xndi.correlate1d(result1,
                                   weights=xp.array([0.178947, 0.642105,
                                                     0.178947]),
                                   axis=axes[1])

        return result2

    def _lazy_transform_gpu(self, X):
        return self.__lazy_transform(X, cundi, cp)

    def _lazy_transform_cpu(self, X):
        return self.__lazy_transform(X, ndi, np)

    def _transform_gpu(self, X):
        return self.__transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self.__transform(X, ndi, np)


class SecondDerivative(Transform):
    def __init__(self, axis=-1, preview=None):
        super().__init__()

        self._axis = axis
        self._preview = preview

    def __lazy_transform(self, X, xndi, xp):
        kernel = (5, 5, 5)
        axes = [ax for ax in range(X.ndim) if ax != self._axis]
        X, chunks_init = create_array(X, kernel, preview=self._preview)

        result0 = X.map_blocks(xndi.correlate1d,
                               weights=xp.array([0.232905, 0.002668,
                                                 -0.471147, 0.002668,
                                                 0.232905]),
                               axis=self._axis, dtype=X.dtype,
                               meta=xp.array((), dtype=X.dtype))
        result1 = result0.map_blocks(xndi.correlate1d,
                                     weights=xp.array([0.030320, 0.249724,
                                                       0.439911, 0.249724,
                                                       0.030320]),
                                     axis=axes[0], dtype=X.dtype,
                                     meta=xp.array((), dtype=X.dtype))
        result2 = result1.map_blocks(xndi.correlate1d,
                                     weights=xp.array([0.030320, 0.249724,
                                                       0.439911, 0.249724,
                                                       0.030320]),
                                     axis=axes[1], dtype=X.dtype,
                                     meta=xp.array((), dtype=X.dtype))

        return trim_dask_array(result2, kernel)

    def __transform(self, X, xndi, xp):
        axes = [ax for ax in range(X.ndim) if ax != self._axis]

        result0 = xndi.correlate1d(X,
                                   weights=xp.array([0.232905, 0.002668,
                                                     -0.471147, 0.002668,
                                                     0.232905]),
                                   axis=self._axis)

        result1 = xndi.correlate1d(result0,
                                   weights=xp.array([0.030320, 0.249724,
                                                     0.439911, 0.249724,
                                                     0.030320]),
                                   axis=axes[0])

        result2 = xndi.correlate1d(result1,
                                   weights=xp.array([0.030320, 0.249724,
                                                     0.439911, 0.249724,
                                                     0.030320]),
                                   axis=axes[1])

        return result2

    def _lazy_transform_gpu(self, X):
        return self.__lazy_transform(X, cundi, cp)

    def _lazy_transform_cpu(self, X):
        return self.__lazy_transform(X, ndi, np)

    def _transform_gpu(self, X):
        return self.__transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self.__transform(X, ndi, np)


class HistogramEqualization(Transform):
    def __interpolate(self, chunk, cdf, bins, xp):
        out = xp.interp(chunk.ravel(), bins, cdf).astype(chunk.dtype)

        return out.reshape(chunk.shape)

    def _lazy_transform_gpu(self, X):
        da_max = X.max()
        da_min = X.min()

        hist, bins = da.histogram(X,
                                  bins=np.linspace(da_min, da_max, 256,
                                                   dtype=X.dtype))

        cdf = hist.cumsum(axis=-1)
        cdf = cdf / cdf[-1]
        bins = (bins[:-1] + bins[1:]) / 2

        return X.map_blocks(self.__interpolate, cdf=cdf, bins=bins, xp=cp,
                            dtype=X.dtype, meta=cp.array((), dtype=X.dtype))

    def _lazy_transform_cpu(self, X):
        da_max = X.max()
        da_min = X.min()

        hist, bins = da.histogram(X,
                                  bins=np.linspace(da_min, da_max, 256,
                                                   dtype=X.dtype))

        cdf = hist.cumsum(axis=-1)
        cdf = cdf / cdf[-1]
        bins = (bins[:-1] + bins[1:]) / 2

        return X.map_blocks(self.__interpolate, cdf=cdf, bins=bins, xp=np,
                            dtype=X.dtype)

    def _transform_gpu(self, X):
        X_max = X.max()
        X_min = X.min()

        hist, bins = cp.histogram(X,
                                  bins=cp.linspace(X_min, X_max, 256,
                                                   dtype=X.dtype))

        cdf = hist.cumsum(axis=-1)
        cdf = cdf / cdf[-1]
        bins = (bins[:-1] + bins[1:]) / 2

        return self.__interpolate(X, cdf=cdf, bins=bins, xp=cp)

    def _transform_cpu(self, X):
        X_max = X.max()
        X_min = X.min()

        hist, bins = np.histogram(X,
                                  bins=np.linspace(X_min, X_max, 256,
                                                   dtype=X.dtype))

        cdf = hist.cumsum(axis=-1)
        cdf = cdf / cdf[-1]
        bins = (bins[:-1] + bins[1:]) / 2

        return self.__interpolate(X, cdf=cdf, bins=bins, xp=np)


class TimeGain(Transform):
    def __init__(self, gain_val=1.5, preview=None):
        super().__init__()

        self._gain_val = gain_val
        self._preview = preview

    def _lazy_transform_gpu(self, X):
        z_ind = da.ones_like(X, chunks=X.chunks).cumsum(axis=-1)

        gain = (1 + z_ind) ** self._gain_val

        return X * gain

    def _lazy_transform_cpu(self, X):
        z_ind = da.ones_like(X, chunks=X.chunks).cumsum(axis=-1)

        gain = (1 + z_ind) ** self._gain_val

        return X * gain

    def _transform_gpu(self, X):
        z_ind = cp.ones_like(X, dtype=X.dtype).cumsum(axis=-1)

        gain = (1 + z_ind) ** self._gain_val

        return X * gain

    def _transform_cpu(self, X):
        z_ind = np.ones_like(X).cumsum(axis=-1)

        gain = (1 + z_ind) ** self._gain_val

        return X * gain


class RescaleAmplitudeRange(Transform):
    def __init__(self, min_val, max_val, preview=None):
        super().__init__()

        self._min_val = min_val
        self._max_val = max_val
        self._preview = preview

    def _lazy_transform(self, X):
        return da.clip(X, self._min_val, self._max_val)

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X)

    def _transform_gpu(self, X):
        return cp.clip(X, self._min_val, self._max_val)

    def _transform_cpu(self, X):
        return np.clip(X, self._min_val, self._max_val)


class RMS(Transform):
    def __init__(self, kernel=(1, 1, 9), preview=None):
        super().__init__()

        if len(kernel) != 3:
            raise ValueError("Kernel should be a 3-D tuple (x, y, z)")

        self._kernel = kernel
        self._preview = preview

    def _rms(self, chunk, kernel):
        x = extract_patches(chunk, kernel, np)
        out = np.sqrt(np.mean(x ** 2, axis=(-3, -2, -1)))

        shape = (np.array(chunk.shape) - np.array(out.shape)) // 2
        pad = ((shape[0], shape[0]),
               (shape[1], shape[1]),
               (shape[2], shape[2]))

        return np.pad(out, pad, mode='constant', constant_values=0)

    def _rms_cu(self, chunk, kernel):
        rms_raw = cp.RawKernel('''extern "C" __global__
            void rms_kernel(const double* x, int filter_size, double* y) {
                double ss = 0;
                for (int i = 0; i < filter_size; ++i) { ss += x[i]*x[i]; }
                y[0] = sqrt(ss/filter_size);
            }''', 'rms_kernel')

        return cundi.generic_filter(input=chunk, function=rms_raw, size=kernel)

    def _lazy_transform(self, X, function, xp):
        X, chunks_init = create_array(X, self._kernel,
                                      preview=self._preview)
        result = X.map_blocks(function, kernel=self._kernel,
                              dtype=X.dtype, meta=xp.array((), dtype=X.dtype))

        result = trim_dask_array(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, function, xp):
        result = function(X, kernel=self._kernel)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, self._rms_cu, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, self._rms, np)

    def _transform_gpu(self, X):
        return self._transform(X, self._rms_cu, cp)

    def _transform_cpu(self, X):
        return self._transform(X, self._rms, np)


class RMS2(Transform):
    def __init__(self, kernel=(1, 1, 9), preview=None):
        super().__init__()

        if len(kernel) != 3:
            raise ValueError("Kernel should be a 3-D tuple (x, y, z)")

        self._kernel = kernel
        self._preview = preview

    def __operation(self, chunk, kernel, xp):
        x = extract_patches(chunk, kernel, xp)
        out = xp.sqrt(xp.mean(x ** 2, axis=(-3, -2, -1)))

        shape = (np.array(chunk.shape) - np.array(out.shape)) // 2
        pad = ((shape[0], shape[0]),
               (shape[1], shape[1]),
               (shape[2], shape[2]))

        return xp.pad(out, pad, mode='constant', constant_values=0)

    def _lazy_transform(self, X, xp):
        X, chunks_init = create_array(X, self._kernel,
                                      preview=self._preview)
        result = X.map_blocks(self.__operation, kernel=self._kernel, xp=xp,
                              dtype=X.dtype, meta=xp.array((), dtype=X.dtype))

        result = trim_dask_array(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xp):
        result = self.__operation(X, kernel=self._kernel, xp=xp)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, np)

    def _transform_gpu(self, X):
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        return self._transform(X, np)


class TraceAGC(Transform):
    def __init__(self, kernel=(1, 1, 9), preview=None):
        super().__init__()

        if len(kernel) != 3:
            raise ValueError("Kernel should be a 3-D tuple (x, y, z)")

        self._kernel = kernel
        self._preview = preview

        self.__rms = RMS(kernel=kernel, preview=preview)

    def _lazy_transform_gpu(self, X):
        X, chunks_init = create_array(X, self._kernel,
                                      preview=self._preview)
        rms = self.__rms._lazy_transform_gpu(X)
        rms_max = rms.max()

        result = X * (1.5 - (rms / rms_max))
        result = trim_dask_array(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        X, chunks_init = create_array(X, self._kernel,
                                      preview=self._preview)
        rms = self.__rms._lazy_transform_cpu(X)
        rms_max = rms.max()

        result = X * (1.5 - (rms / rms_max))
        result = trim_dask_array(result, self._kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        rms = self.__rms._transform_gpu(X)
        rms_max = rms.max()

        result = X * (1.5 - (rms / rms_max))
        result[cp.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        rms = self.__rms._transform_cpu(X)
        rms_max = rms.max()

        result = X * (1.5 - (rms / rms_max))
        result[np.isnan(result)] = 0

        return result


class GradientMagnitude(Transform):
    def __init__(self, sigmas=(1, 1, 1), preview=None):
        super().__init__()

        if len(sigmas) != 3:
            raise ValueError("Sigmas should be a 3-D tuple (x, y, z)")

        self._sigmas = sigmas
        self._preview = preview

    def _lazy_transform(self, X, xndi):
        kernel = tuple(2 * (4 * np.array(self._sigmas) + 0.5).astype(int) + 1)
        X, chunks_init = create_array(X, kernel, preview=self._preview)

        result = X.map_blocks(xndi.gaussian_gradient_magnitude,
                              sigma=self._sigmas, dtype=X.dtype)

        result = trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xndi, xp):
        result = xndi.gaussian_gradient_magnitude(X, sigma=self._sigmas)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cundi)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, ndi)

    def _transform_gpu(self, X):
        return self._transform(X, cundi, cp)

    def _transform_cpu(self, X):
        return self._transform(X, ndi, np)


class ReflectionIntensity(Transform):
    def __init__(self, kernel=(1, 1, 9), preview=None):
        super().__init__()

        self._kernel = kernel
        self._preview = preview

    def __operation(self, chunk, kernel, xp):
        if not hasattr(xp, 'trapz'):
            # XXX: Some CuPy versions don't have trapz() method
            raise NotImplementedError("Method trapz() is not implemented yet")

        x = extract_patches(chunk, (1, 1, kernel[-1]), xp)
        return xp.trapz(x).reshape(x.shape[:3])

    def _lazy_transform(self, X, xp):
        X, chunks_init = create_array(X, self._kernel, preview=self._preview)

        result = X.map_blocks(self.__operation, kernel=self._kernel, xp=xp,
                              dtype=X.dtype, chunks=chunks_init, meta=xp.array((), dtype=X.dtype))

        result = trim_dask_array(result, self._kernel)

        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xp):
        result = self.__operation(X, kernel=self._kernel, xp=xp)

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, np)

    def _transform_gpu(self, X):
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        return self._transform(X, np)


class PhaseRotation(Transform):
    def __init__(self, rotation, preview=None):
        super().__init__()

        self._rotation = rotation
        self._preview = preview

    def _lazy_transform(self, X, xsignal, xp):
        kernel = set_time_chunk_overlap(X)

        if kernel:
            X, chunks_init = create_array(X, kernel, preview=self._preview)

        phi = xp.deg2rad(self._rotation, dtype=X.dtype)

        analytical_trace = X.map_blocks(xsignal.hilbert, dtype=X.dtype,
                                        meta=xp.array((), dtype=X.dtype))

        result = (analytical_trace.real * da.cos(phi) -
                  analytical_trace.imag * da.sin(phi))

        if kernel:
            result = trim_dask_array(result, kernel)

        result[da.isnan(result)] = 0

        return result

    def _transform(self, X, xsignal, xp):
        phi = xp.deg2rad(self._rotation, dtype=X.dtype)

        analytical_trace = xsignal.hilbert(X)

        result = (analytical_trace.real * xp.cos(phi) -
                  analytical_trace.imag * xp.sin(phi))

        result[xp.isnan(result)] = 0

        return result

    def _lazy_transform_gpu(self, X):
        return self._lazy_transform(X, cusignal, cp)

    def _lazy_transform_cpu(self, X):
        return self._lazy_transform(X, signal, np)

    def _transform_gpu(self, X):
        return self._transform(X, cusignal, cp)

    def _transform_cpu(self, X):
        return self._transform(X, signal, np)
