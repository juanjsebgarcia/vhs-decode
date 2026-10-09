mod bspline;
mod bursts;
mod filters;
mod levels;
mod ported;

use numpy::ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyList;

use bspline::bspline_evaluate;
use bursts::upconvert_filter_bursts_impl;
use filters::{
    f32_sos_from_scipy_dyn_slice, from_scipy_dyn_slice, sos_filtfilt, sos_filtfilt_f32,
};
use levels::fallback_vsync_loc_means_impl;
use ported::unwrap_angles_impl;

// https://mazzo.li/posts/vectorized-atan2.html
#[inline(always)]
fn atan_approx(x: f32) -> f32 {
    const A1: f32 = 0.99997726f32;
    const A3: f32 = -0.33262347f32;
    const A5: f32 = 0.19354346f32;
    const A7: f32 = -0.11643287f32;
    const A9: f32 = 0.05265332f32;
    const A11: f32 = -0.01172120f32;

    let x_sq = x * x;
    x * (A1 + x_sq * (A3 + x_sq * (A5 + x_sq * (A7 + x_sq * (A9 + x_sq * A11)))))
}

#[inline(always)]
fn atan2f_fast(y: f32, x: f32) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    use std::f32::consts::PI;

    let x = x + f32::MIN_POSITIVE.copysign(x);
    let swap = x.abs() < y.abs();
    let atan_input = if swap { x } else { y } / if swap { y } else { x };
    let mut res = atan_approx(atan_input);
    let tmp = if atan_input >= 0.0f32 {
        FRAC_PI_2
    } else {
        -FRAC_PI_2
    };
    res = if swap { tmp - res } else { res };
    match (x >= 0f32, y >= 0f32) {
        (true, true) => res,
        (false, true) => PI + res,
        (false, false) => -PI + res,
        (true, false) => res,
    }
}

#[inline(always)]
fn hilbert_two(a: Complex64, b: Complex64, freq: f32) -> f32 {
    use std::f32::consts::TAU;

    let a = atan2f_fast(a.im as f32, a.re as f32);
    let b = atan2f_fast(b.im as f32, b.re as f32);
    let diff = b - a;
    let diff = diff - (diff / TAU).floor() * TAU;
    diff * freq / TAU
}

#[inline(always)]
fn hilbert_more(a: &[Complex64; 8], b: &[Complex64; 8], out: &mut [f64; 8], freq: f32) {
    for i in 0..8 {
        out[i] = hilbert_two(a[i], b[i], freq) as f64;
    }
}

#[inline(never)]
fn hilbert_all(input_slice: &[Complex64], output_slice: &mut [f64], freq: f32) {
    let len = input_slice.len();
    assert_ne!(len, 0);

    let big_chunks = (len - 1) / 8;
    for i in 0..big_chunks {
        let prevs_slice = &input_slice[i * 8..(i + 1) * 8];
        let currs_slice = &input_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let outs_slice = &mut output_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let prevs = <&[Complex64; 8]>::try_from(prevs_slice).unwrap();
        let currs = <&[Complex64; 8]>::try_from(currs_slice).unwrap();
        let outs = <&mut [f64; 8]>::try_from(outs_slice).unwrap();
        hilbert_more(prevs, currs, outs, freq);
    }
    for i in big_chunks * 8..len - 1 {
        output_slice[i + 1] = hilbert_two(input_slice[i], input_slice[i + 1], freq) as f64;
    }
}

fn unwrap_hilbert_impl(input_array: ArrayView1<'_, Complex64>, freq: f64) -> Array1<f64> {
    let mut output_array = Array1::<f64>::zeros(input_array.len());

    let input_slice = input_array.as_slice().unwrap();
    let output_slice = output_array.as_slice_mut().unwrap();

    hilbert_all(input_slice, output_slice, freq as f32);

    output_array
}

fn complex_angle(input_array: ArrayView1<'_, Complex64>) -> Array1<f64> {
    // Replicate the np.angle function for 1-dimensional input
    // it's possible we could use SIMD in some way to speed this up but that might be complicated.
    let mut output_array = Array1::<f64>::zeros(input_array.len());
    Zip::from(&mut output_array)
        .and(&input_array)
        .for_each(|i, &j| {
            *i = j.im.atan2(j.re);
        });
    output_array
}

fn diff_forward_in_place_impl(mut input_array: ArrayViewMut1<'_, f64>) {
    // output the difference of the output in place, putting 0 in the first element.
    // equivialent to calling np.ediff1d(input_array, to_begin=0)
    for i in (2..input_array.len()).rev() {
        input_array[i] -= input_array[i - 1];
    }
    input_array[1] -= input_array[0];
    input_array[0] = 0.0;
}

#[pyfunction]
fn complex_angle_py<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, Complex64>,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| complex_angle(input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn unwrap_angles<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| unwrap_angles_impl(input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn diff_forward_in_place<'py>(py: Python<'py>, mut input_array: PyReadwriteArray1<'py, f64>) {
    let input_array = input_array.as_array_mut();
    py.detach(|| diff_forward_in_place_impl(input_array));
}

#[pyfunction]
fn unwrap_hilbert<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, Complex64>,
    freq: f64,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| unwrap_hilbert_impl(input_array, freq));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn fallback_vsync_loc_means<'py>(
    py: Python<'py>,
    demod_05: PyReadonlyArray1<'py, f64>,
    pulses: Bound<'py, PyList>,
    sample_freq_mhz: f64,
    min_len: f64,
    max_len: f64,
) -> (Bound<'py, PyList>, Bound<'py, PyList>) {
    let demod_05 = demod_05.as_array();
    //let output_list = py.allow_threads(|| fallback_vsync_loc_means_impl(demod_05, freq));
    //output_array.into_pyarray_bound(py)
    fallback_vsync_loc_means_impl(py, demod_05, pulses, sample_freq_mhz, min_len, max_len)
}

#[pyfunction]
fn sosfiltfilt<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    input_array: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let sos_filter = sos_filter.as_array();
    let input_array = input_array.as_array();
    let output_array = py.detach(|| sos_filtfilt(order, sos_filter, input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn sosfiltfilt_f32<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    input_array: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let sos_filter = sos_filter.as_array();
    let input_array = input_array.as_array();
    let output_array = py.detach(|| sos_filtfilt_f32(order, sos_filter, input_array));
    output_array.into_pyarray(py)
}

/// Checks the burst bounds of upconvert_filter_bursts(_f32).
fn check_burst_bounds(
    chroma_len: usize,
    heterodyne_lens: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> PyResult<()> {
    if heterodyne_lens.len() != starts.len() || ends.len() != starts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "heterodynes, starts and ends must have the same length",
        ));
    }
    for (i, &heterodyne_len) in heterodyne_lens.iter().enumerate() {
        let end = ends[i].max(starts[i]);
        if end > chroma_len || end > heterodyne_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "burst bounds exceed the chroma or heterodyne length",
            ));
        }
    }
    Ok(())
}

/// Up-convert (multiply with the heterodyne) and filter the color burst area of
/// several lines in one call, see bursts::upconvert_filter_bursts_impl.
/// Returns the filtered bursts (one zero padded row per line) and the number of
/// valid samples in each row.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn upconvert_filter_bursts<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    chroma: PyReadonlyArray1<'py, f32>,
    heterodynes: Vec<PyReadonlyArray1<'py, f64>>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    padding: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let lens: Vec<usize> = heterodynes.iter().map(|h| h.len()).collect();
    check_burst_bounds(chroma.len(), &lens, &starts, &ends)?;
    let sos = from_scipy_dyn_slice(order as usize, sos_filter.as_slice()?);
    let chroma = chroma.as_array();
    let heterodynes: Vec<_> = heterodynes.iter().map(|h| h.as_array()).collect();
    let (bursts, lens) = py.detach(|| {
        upconvert_filter_bursts_impl(&sos, chroma, &heterodynes, &starts, &ends, padding)
    });
    Ok((bursts.into_pyarray(py), lens.into_pyarray(py)))
}

/// Single precision version of upconvert_filter_bursts (f32 heterodynes, the product
/// is filtered in f32 like sosfiltfilt_f32 does).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn upconvert_filter_bursts_f32<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    chroma: PyReadonlyArray1<'py, f32>,
    heterodynes: Vec<PyReadonlyArray1<'py, f32>>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    padding: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>)> {
    let lens: Vec<usize> = heterodynes.iter().map(|h| h.len()).collect();
    check_burst_bounds(chroma.len(), &lens, &starts, &ends)?;
    let sos = f32_sos_from_scipy_dyn_slice(order as usize, sos_filter.as_slice()?);
    let chroma = chroma.as_array();
    let heterodynes: Vec<_> = heterodynes.iter().map(|h| h.as_array()).collect();
    let (bursts, lens) = py.detach(|| {
        upconvert_filter_bursts_impl(&sos, chroma, &heterodynes, &starts, &ends, padding)
    });
    Ok((bursts.into_pyarray(py), lens.into_pyarray(py)))
}

/// Evaluate the nu-th derivative of the B-spline with knots t, coefficients c and degree k
/// at x, like scipy.interpolate.BSpline(t, c, k, extrapolate)(x, nu) for 1-D c.
/// fused selects whether the multiply-adds are fused, to match how scipy was compiled.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bspline_eval<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    k: usize,
    x: PyReadonlyArray1<'py, f64>,
    nu: usize,
    extrapolate: bool,
    fused: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = t.as_slice()?;
    let c = c.as_slice()?;
    let x = x.as_slice()?;
    if t.len() < 2 * k + 2 || c.len() < t.len() - k - 1 || nu > k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "unsupported spline: need len(t) >= 2k + 2, len(c) >= len(t) - k - 1, nu <= k",
        ));
    }
    let mut out = vec![0.0f64; x.len()];
    py.detach(|| bspline_evaluate(t, c, k, x, nu, extrapolate, fused, &mut out));
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn check_debug<'py>(
    _py: Python<'py>,
) -> PyResult<bool> {
    if cfg!(debug_assertions) {
        Ok(true)
    } else {
        Ok(false)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn vhsd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(complex_angle_py, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_angles, m)?)?;
    m.add_function(wrap_pyfunction!(diff_forward_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_hilbert, m)?)?;
    m.add_function(wrap_pyfunction!(fallback_vsync_loc_means, m)?)?;
    m.add_function(wrap_pyfunction!(sosfiltfilt, m)?)?;
    m.add_function(wrap_pyfunction!(sosfiltfilt_f32, m)?)?;
    m.add_function(wrap_pyfunction!(upconvert_filter_bursts, m)?)?;
    m.add_function(wrap_pyfunction!(upconvert_filter_bursts_f32, m)?)?;
    m.add_function(wrap_pyfunction!(bspline_eval, m)?)?;
    m.add_function(wrap_pyfunction!(check_debug, m)?)?;
    Ok(())
}
