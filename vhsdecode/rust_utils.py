import numpy as np
import scipy.signal as sps

try:
    from vhsd_rust import sosfiltfilt, sosfiltfilt_f32

    _HAS_VHSD_RUST = True
except ModuleNotFoundError:
    sosfiltfilt = None
    sosfiltfilt_f32 = None
    _HAS_VHSD_RUST = False

try:
    from vhsd_rust import upconvert_filter_bursts as _upconvert_filter_bursts
    from vhsd_rust import upconvert_filter_bursts_f32 as _upconvert_filter_bursts_f32
except ImportError:
    # extension built from an older source without these functions
    _upconvert_filter_bursts = None
    _upconvert_filter_bursts_f32 = None


def sos_filter_as_array_and_order(filter):
    """Convert the sos filter to a array derive the filter order for use inside
    rust code with sci_rs
    We do this here rather than in rust for now for easier interop."""
    filter_view = filter.ravel()
    assert (
        len(filter_view) % 6 == 0
    ), "filter length is not divideable by 6, there is a bug somewhere!"
    return int(len(filter_view) / 6), filter_view


def sosfiltfilt_rust(sos, input):
    assert input.dtype != np.complex128
    if input.dtype == np.complex128:
        input = abs(input)

    if not _HAS_VHSD_RUST:
        return sps.sosfiltfilt(sos, input)

    order, filter = sos_filter_as_array_and_order(sos)

    if input.dtype == np.float64:
        return sosfiltfilt(order, filter, input)
    # if input.dtype == np.float32:
    #    return sosfiltfilt_f32(order, filter, input)
    return sosfiltfilt_f32(order, filter, input.astype(np.float32))


def upconvert_filter_bursts_rust(sos, chroma, heterodynes, burst_starts, burst_ends, padding):
    """Multiply chroma[start:end] with heterodyne[start:end] and filter it with
    sosfiltfilt_rust for each (heterodyne, start, end), dropping padding samples from
    both ends of each filtered burst, all in one call that releases the GIL.

    Returns (bursts, burst_lens): row i of bursts holds burst_lens[i] valid samples
    followed by zeros. Returns None if the inputs are not supported by the Rust
    implementation (the caller then has to do it in python).
    """
    if _upconvert_filter_bursts is None or chroma.dtype != np.float32 or not heterodynes:
        return None

    # same precision as sosfiltfilt_rust(sos, heterodyne * chroma)
    heterodyne_dtype = heterodynes[0].dtype
    if heterodyne_dtype == np.float32:
        upconvert_filter_bursts = _upconvert_filter_bursts_f32
    elif heterodyne_dtype == np.float64:
        upconvert_filter_bursts = _upconvert_filter_bursts
    else:
        return None

    if (
        any(h.dtype != heterodyne_dtype or h.ndim != 1 for h in heterodynes)
        or min(burst_ends) < 0
        or max(burst_ends) > min(len(chroma), min(len(h) for h in heterodynes))
    ):
        return None

    order, filter = sos_filter_as_array_and_order(sos)
    return upconvert_filter_bursts(
        order, filter, chroma, heterodynes, burst_starts, burst_ends, padding
    )
