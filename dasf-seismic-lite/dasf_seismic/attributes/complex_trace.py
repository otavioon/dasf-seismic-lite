#!/usr/bin/env python3

import numpy as np
from scipy import signal

from pkg_resources import parse_version

import dask.array as da

try:
    import cupy as cp
    import cusignal
except ImportError:
    pass

from dasf.transforms import Transform

from dasf_seismic.utils.utils import create_array
from dasf_seismic.utils.utils import trim_dask_array
from dasf_seismic.utils.utils import local_events
from dasf_seismic.utils.utils import set_time_chunk_overlap
from dasf_seismic.utils.utils import dask_cupy_angle_wrapper
from dasf_seismic.attributes.signal import FirstDerivative


class Hilbert(Transform):
    def __init__(self, preview=None):
        super().__init__()

        self._preview = preview

    def __real_signal_hilbert(self, X, xsignal):
        # Avoiding return complex128
        return xsignal.hilbert(X)

    def _lazy_transform(self, X, xsignal, xp):
        return X.map_blocks(self.__real_signal_hilbert, xsignal=xsignal,
                            dtype=X.dtype, meta=xp.array((), dtype=X.dtype))

    def _lazy_transform_cpu(self, X):
        kernel = set_time_chunk_overlap(X)

        if kernel:
            X, chunks_init = create_array(X, kernel,
                                          preview=self._preview)

        analytical_trace = self._lazy_transform(X, xsignal=signal, xp=np)

        if kernel:
            return trim_dask_array(analytical_trace, kernel)
        return analytical_trace

    def _lazy_transform_gpu(self, X):
        if X.shape[-1] != X.chunksize[-1]:
            time_edge = int(X.chunksize[-1] * 0.1)
            if time_edge < 5:
                time_edge = X.chunksize[-1] * 0.5

            kernel = (1, 1, int(time_edge))
            X, chunks_init = create_array(X, kernel,
                                          preview=self._preview)

        analytical_trace = self._lazy_transform(X, xsignal=cusignal, xp=cp)

        if X.shape[-1] != X.chunksize[-1]:
            return trim_dask_array(analytical_trace, kernel)
        return analytical_trace

    def _transform_cpu(self, X):
        return self.__real_signal_hilbert(X, signal)

    def _transform_gpu(self, X):
        return self.__real_signal_hilbert(X, cusignal)


class Envelope(Hilbert):
    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.absolute(analytical_trace)

    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        return da.absolute(analytical_trace)

    def _transform_cpu(self, X):
        return np.absolute(super()._transform_cpu(X))

    def _transform_gpu(self, X):
        return cp.absolute(super()._transform_gpu(X))


class InstantaneousPhase(Hilbert):
    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.rad2deg(da.angle(analytical_trace))

    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        if parse_version(cp.__version__) < parse_version("12.0.0"):
            return da.rad2deg(dask_cupy_angle_wrapper(analytical_trace))
        return da.rad2deg(da.angle(analytical_trace))

    def _transform_cpu(self, X):
        analytical_trace = super()._transform_cpu(X)

        return np.rad2deg(np.angle(analytical_trace))

    def _transform_gpu(self, X):
        analytical_trace = super()._transform_gpu(X)

        return cp.rad2deg(cp.angle(analytical_trace))


class CosineInstantaneousPhase(Hilbert):
    def _lazy_transform_gpu(self, X):
        analytical_trace = super()._lazy_transform_gpu(X)

        if parse_version(cp.__version__) < parse_version("12.0.0"):
            return da.cos(dask_cupy_angle_wrapper(analytical_trace))
        return da.cos(da.angle(analytical_trace))

    def _lazy_transform_cpu(self, X):
        analytical_trace = super()._lazy_transform_cpu(X)

        return da.cos(da.angle(analytical_trace))

    def _transform_gpu(self, X):
        return cp.cos(cp.angle(super()._transform_gpu(X)))

    def _transform_cpu(self, X):
        return np.cos(np.angle(super()._transform_cpu(X)))


class RelativeAmplitudeChange(Transform):
    def __init__(self, preview=None):
        super().__init__()

        self._preview = preview

        self.__envelope = Envelope(preview=preview)
        self.__first_derivative = FirstDerivative(preview=preview, axis=-1)

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        env_prime = self.__first_derivative._lazy_transform_gpu(X)

        result = env_prime / env

        return da.clip(result, -1, 1)

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        env_prime = self.__first_derivative._lazy_transform_cpu(X)

        result = env_prime / env

        return da.clip(result, -1, 1)

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        env_prime = self.__first_derivative._transform_gpu(X)

        result = env_prime / env

        return cp.clip(result, -1, 1)

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        env_prime = self.__first_derivative._transform_cpu(X)

        result = env_prime / env

        return np.clip(result, -1, 1)


class AmplitudeAcceleration(RelativeAmplitudeChange):
    def __init__(self, preview=None):
        super().__init__(preview=preview)

        self.__first_derivative = FirstDerivative(preview=preview, axis=-1)

    def _lazy_transform_gpu(self, X):
        rac = super()._lazy_transform_gpu(X)

        return self.__first_derivative._lazy_transform_gpu(rac)

    def _lazy_transform_cpu(self, X):
        rac = super()._lazy_transform_cpu(X)

        return self.__first_derivative._lazy_transform_cpu(rac)

    def _transform_gpu(self, X):
        rac = super()._transform_gpu(X)

        return self.__first_derivative._transform_gpu(rac)

    def _transform_cpu(self, X):
        rac = super()._transform_cpu(X)

        return self.__first_derivative._transform_cpu(rac)


class InstantaneousFrequency(Transform):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__()

        self._sample_rate = sample_rate
        self._preview = preview

        self.__inst_phase = InstantaneousPhase(preview=preview)
        self.__first_derivative = FirstDerivative(preview=preview, axis=-1)

    def _lazy_transform_gpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._lazy_transform_gpu(X)
        phase = da.deg2rad(phase)
        phase = phase.map_blocks(cp.unwrap, dtype=X.dtype)

        phase_prime = self.__first_derivative._lazy_transform_gpu(phase)

        return da.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _lazy_transform_cpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._lazy_transform_cpu(X)
        phase = da.deg2rad(phase)
        phase = phase.map_blocks(np.unwrap, dtype=X.dtype)

        phase_prime = self.__first_derivative._lazy_transform_cpu(phase)

        return da.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _transform_gpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._transform_gpu(X)
        phase = cp.deg2rad(phase)
        phase = cp.unwrap(phase)

        phase_prime = self.__first_derivative._transform_gpu(phase)

        return cp.absolute((phase_prime / (2.0 * np.pi) * fs))

    def _transform_cpu(self, X):
        fs = 1000 / self._sample_rate

        phase = self.__inst_phase._transform_cpu(X)
        phase = np.deg2rad(phase)
        phase = np.unwrap(phase)

        phase_prime = self.__first_derivative._transform_cpu(phase)

        return np.absolute((phase_prime / (2.0 * np.pi) * fs))


class InstantaneousBandwidth(RelativeAmplitudeChange):
    def _lazy_transform_gpu(self, X):
        rac = super()._lazy_transform_gpu(X)

        return da.absolute(rac) / (2.0 * np.pi)

    def _lazy_transform_cpu(self, X):
        rac = super()._lazy_transform_cpu(X)

        return da.absolute(rac) / (2.0 * np.pi)

    def _transform_gpu(self, X):
        rac = super()._transform_gpu(X)

        return cp.absolute(rac) / (2.0 * np.pi)

    def _transform_cpu(self, X):
        rac = super()._transform_cpu(X)

        return np.absolute(rac) / (2.0 * np.pi)


class DominantFrequency(Transform):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__()

        self._sample_rate = sample_rate
        self._preview = preview

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate,
                                                  preview=preview)
        self.__inst_band = InstantaneousBandwidth(preview=preview)

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        inst_band = self.__inst_band._lazy_transform_gpu(X)
        return da.hypot(inst_freq, inst_band)

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        inst_band = self.__inst_band._lazy_transform_cpu(X)
        return da.hypot(inst_freq, inst_band)

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        inst_band = self.__inst_band._transform_gpu(X)
        return cp.hypot(inst_freq, inst_band)

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        inst_band = self.__inst_band._transform_cpu(X)
        return np.hypot(inst_freq, inst_band)


class FrequencyChange(Transform):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__()

        self._sample_rate = sample_rate
        self._preview = preview

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate,
                                                  preview=preview)
        self.__first_derivative = FirstDerivative(preview=preview, axis=-1)

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        return self.__first_derivative._lazy_transform_gpu(inst_freq)

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        return self.__first_derivative._lazy_transform_cpu(inst_freq)

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        return self.__first_derivative._transform_gpu(inst_freq)

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        return self.__first_derivative._transform_cpu(inst_freq)


class Sweetness(Envelope):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__(preview=preview)

        self._sample_rate = sample_rate

        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate,
                                                  preview=preview)

    def __sweetness_limit(self, X):
        X[X < 5] = 5
        return X

    def _lazy_transform_gpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        inst_freq = inst_freq.map_blocks(self.__sweetness_limit, dtype=X.dtype)
        env = super()._lazy_transform_gpu(X)

        return env / inst_freq

    def _lazy_transform_cpu(self, X):
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        inst_freq = inst_freq.map_blocks(self.__sweetness_limit, dtype=X.dtype)
        env = super()._lazy_transform_cpu(X)

        return env / inst_freq

    def _transform_gpu(self, X):
        inst_freq = self.__inst_freq._transform_gpu(X)
        inst_freq = self.__sweetness_limit(inst_freq)
        env = super()._transform_gpu(X)

        return env / inst_freq

    def _transform_cpu(self, X):
        inst_freq = self.__inst_freq._transform_cpu(X)
        inst_freq = self.__sweetness_limit(inst_freq)
        env = super()._transform_cpu(X)

        return env / inst_freq


class QualityFactor(InstantaneousFrequency):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__(sample_rate=sample_rate, preview=preview)

        self.__rac = RelativeAmplitudeChange(preview=preview)

    def _lazy_transform_gpu(self, X):
        inst_freq = super()._lazy_transform_gpu(X)
        rac = self.__rac._lazy_transform_gpu(X)

        return (np.pi * inst_freq) / rac

    def _lazy_transform_cpu(self, X):
        inst_freq = super()._lazy_transform_cpu(X)
        rac = self.__rac._lazy_transform_cpu(X)

        return (np.pi * inst_freq) / rac

    def _transform_gpu(self, X):
        inst_freq = super()._transform_gpu(X)
        rac = self.__rac._transform_gpu(X)

        return (np.pi * inst_freq) / rac

    def _transform_cpu(self, X):
        inst_freq = super()._transform_cpu(X)
        rac = self.__rac._transform_cpu(X)

        return (np.pi * inst_freq) / rac


class ResponsePhase(Transform):
    def __init__(self, preview=None):
        super().__init__()

        self._preview = preview

        self.__envelope = Envelope(preview=preview)
        self.__inst_phase = InstantaneousPhase(preview=preview)

    def __operation(self, chunk1, chunk2, chunk3, xp):
        out = xp.zeros(chunk1.shape)
        for i, j in np.ndindex(out.shape[:-1]):
            ints = xp.unique(chunk3[i, j, :])
            for ii in ints:
                idx = xp.where(chunk3[i, j, :] == ii)[0]
                peak = idx[chunk1[i, j, idx].argmax()]
                out[i, j, idx] = chunk2[i, j, peak]

        return out

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        phase = self.__inst_phase._lazy_transform_gpu(X)

        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(self.__operation, env, phase, troughs, cp,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        phase = self.__inst_phase._lazy_transform_cpu(X)

        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(self.__operation, env, phase, troughs, np,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        phase = self.__inst_phase._transform_gpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, phase, troughs, cp)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        phase = self.__inst_phase._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, phase, troughs, np)

        result[da.isnan(result)] = 0

        return result


class ResponseFrequency(Transform):
    def __init__(self, sample_rate=4, preview=None):
        super().__init__()

        self.__envelope = Envelope(preview=preview)
        self.__inst_freq = InstantaneousFrequency(sample_rate=sample_rate,
                                                  preview=preview)

    def __operation(self, chunk1, chunk2, chunk3, xp):
        out = xp.zeros(chunk1.shape)
        for i, j in np.ndindex(out.shape[:-1]):
            ints = xp.unique(chunk3[i, j, :])
            for ii in ints:
                idx = xp.where(chunk3[i, j, :] == ii)[0]
                peak = idx[chunk1[i, j, idx].argmax()]
                out[i, j, idx] = chunk2[i, j, peak]

        return out

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        inst_freq = self.__inst_freq._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(self.__operation, env, inst_freq, troughs, cp,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        inst_freq = self.__inst_freq._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)
        result = da.map_blocks(self.__operation, env, inst_freq, troughs, np,
                               dtype=X.dtype)
        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        inst_freq = self.__inst_freq._transform_gpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, inst_freq, troughs, cp)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        inst_freq = self.__inst_freq._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, inst_freq, troughs, np)

        result[da.isnan(result)] = 0

        return result


class ResponseAmplitude(Transform):
    def __init__(self, preview=None):
        super().__init__()

        self._preview = preview

        self.__envelope = Envelope(preview=preview)

    def __operation(self, chunk1, chunk2, chunk3, xp):
        out = xp.zeros(chunk1.shape)
        for i, j in np.ndindex(out.shape[:-1]):
            ints = xp.unique(chunk3[i, j, :])
            for ii in ints:
                idx = xp.where(chunk3[i, j, :] == ii)[0]
                peak = idx[chunk1[i, j, idx].argmax()]
                out[i, j, idx] = chunk2[i, j, peak]

        return out

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(self.__operation, env, X, troughs, cp,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(self.__operation, env, X, troughs, np,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        troughs = local_events(env, comparator=cp.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, X, troughs, cp)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, X, troughs, np)

        result[da.isnan(result)] = 0

        return result


class ApparentPolarity(Transform):
    def __init__(self, preview=None):
        super().__init__()

        self._preview = preview

        self.__envelope = Envelope(preview=preview)

    def __operation(self, chunk1, chunk2, chunk3, xp):
        out = xp.zeros(chunk1.shape)
        for i, j in np.ndindex(out.shape[:-1]):
            ints = xp.unique(chunk3[i, j, :])

            for ii in ints:
                idx = xp.where(chunk3[i, j, :] == ii)[0]
                peak = idx[chunk1[i, j, idx].argmax()]
                out[i, j, idx] = chunk1[i, j, peak] * \
                    xp.sign(chunk2[i, j, peak])

        return out

    def _lazy_transform_gpu(self, X):
        env = self.__envelope._lazy_transform_gpu(X)
        troughs = env.map_blocks(local_events, comparator=cp.less,
                                 is_cupy=True, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(self.__operation, env, X, troughs, cp,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _lazy_transform_cpu(self, X):
        env = self.__envelope._lazy_transform_cpu(X)
        troughs = env.map_blocks(local_events, comparator=np.less,
                                 is_cupy=False, dtype=X.dtype)

        troughs = troughs.cumsum(axis=-1)

        X = X.rechunk(env.chunks)

        result = da.map_blocks(self.__operation, env, X, troughs, np,
                               dtype=X.dtype)

        result[da.isnan(result)] = 0

        return result

    def _transform_gpu(self, X):
        env = self.__envelope._transform_gpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=True)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, X, troughs, cp)

        result[da.isnan(result)] = 0

        return result

    def _transform_cpu(self, X):
        env = self.__envelope._transform_cpu(X)
        troughs = local_events(env, comparator=np.less, is_cupy=False)

        troughs = troughs.cumsum(axis=-1)
        result = self.__operation(env, X, troughs, np)

        result[da.isnan(result)] = 0

        return result
