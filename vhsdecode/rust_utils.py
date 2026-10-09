import numpy as np
import scipy.signal as sps

try:
    from vhsd_rust import sosfiltfilt, sosfiltfilt_f32
    from vhsd_rust import rf_filter_hilbert as _rf_filter_hilbert
    from vhsd_rust import rf_envelope as _rf_envelope

    _HAS_VHSD_RUST = True
except ModuleNotFoundError:
    sosfiltfilt = None
    sosfiltfilt_f32 = None
    _HAS_VHSD_RUST = False


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
    return sosfiltfilt_f32(order, filter, np.ascontiguousarray(input, dtype=np.float32))


def _is_contiguous(array, dtype):
    return array.dtype == dtype and array.ndim == 1 and array.flags.c_contiguous


def rf_filter_hilbert(spectrum, filters, hilbert, out):
    """Apply each filter in filters to spectrum in place, then write spectrum * hilbert
    to out, in one pass when the arrays allow it.
    Same result as `for f in filters: spectrum *= f` followed by
    `np.multiply(spectrum, hilbert, out=out)`."""
    if (
        _HAS_VHSD_RUST
        and _is_contiguous(spectrum, np.complex128)
        and _is_contiguous(out, np.complex128)
        and _is_contiguous(hilbert, np.float64)
        and all(_is_contiguous(f, np.float64) for f in filters)
        and all(len(a) == len(spectrum) for a in (out, hilbert, *filters))
    ):
        _rf_filter_hilbert(spectrum, list(filters), hilbert, out)
        return out

    for f in filters:
        spectrum *= f
    return np.multiply(spectrum, hilbert, out=out)


def rf_envelope(sos, analytic, shift):
    """RF envelope, the same as
    sosfiltfilt_rust(sos, np.roll(np.abs(analytic.real.astype(np.float32)), shift)).
    Returns the envelope and whether any of its samples is zero."""
    if _HAS_VHSD_RUST and _is_contiguous(analytic, np.complex128) and len(analytic):
        order, filter = sos_filter_as_array_and_order(sos)
        return _rf_envelope(order, filter, analytic, shift)

    raw_env = np.roll(np.abs(analytic.real.astype(np.float32)), shift)
    env = sosfiltfilt_rust(sos, raw_env)
    return env, bool(np.any(env == 0))
