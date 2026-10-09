//! Element-wise steps of VHSRFDecode.demodblock fused into single passes.
//!
//! Every function here does the same floating point operations, in the same order,
//! as the numpy code it replaces, so the results are bit-identical. Rust does not
//! contract a * b + c into a fused multiply-add unless asked to, so the products
//! are rounded like numpy's.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::Complex64;

use crate::filters::sos_filtfilt_f32;

/// numpy's complex multiply, `x * complex(f, 0.0)`, which is what multiplying a
/// complex128 array by a float64 array does (the float is cast to complex first).
/// The products with the zero imaginary part are kept so that the signs of zero
/// results match numpy. They are exact, so a fused or unfused multiply-add gives
/// the same result here.
#[inline(always)]
fn mul_by_real(x: Complex64, f: f64) -> Complex64 {
    let f_im = 0.0f64;
    Complex64::new(x.re * f - x.im * f_im, x.re * f_im + x.im * f)
}

/// spectrum *= filters[0]; spectrum *= filters[1]; ...; analytic = spectrum * hilbert
///
/// All filters are real (float64), like the RF band-pass, the notch and the
/// one-sided hilbert weights.
pub fn rf_filter_hilbert_impl(
    spectrum: &mut [Complex64],
    filters: &[ArrayView1<'_, f64>],
    hilbert: &[f64],
    analytic: &mut [Complex64],
) {
    let filters: Vec<&[f64]> = filters.iter().map(|f| f.as_slice().unwrap()).collect();
    match filters.as_slice() {
        [] => {
            for ((x, a), &h) in spectrum.iter().zip(analytic.iter_mut()).zip(hilbert) {
                *a = mul_by_real(*x, h);
            }
        }
        [f0] => {
            for (((x, a), &h), &g0) in spectrum
                .iter_mut()
                .zip(analytic.iter_mut())
                .zip(hilbert)
                .zip(*f0)
            {
                *x = mul_by_real(*x, g0);
                *a = mul_by_real(*x, h);
            }
        }
        [f0, f1] => {
            for ((((x, a), &h), &g0), &g1) in spectrum
                .iter_mut()
                .zip(analytic.iter_mut())
                .zip(hilbert)
                .zip(*f0)
                .zip(*f1)
            {
                *x = mul_by_real(mul_by_real(*x, g0), g1);
                *a = mul_by_real(*x, h);
            }
        }
        _ => {
            for (i, (x, a)) in spectrum.iter_mut().zip(analytic.iter_mut()).enumerate() {
                for f in filters.iter() {
                    *x = mul_by_real(*x, f[i]);
                }
                *a = mul_by_real(*x, hilbert[i]);
            }
        }
    }
}

/// The RF envelope: sosfiltfilt (in single precision) of
/// np.roll(np.abs(analytic.real.astype(np.float32)), shift).
/// Also returns whether any envelope sample is exactly zero.
pub fn rf_envelope_impl(
    sos_order: u32,
    sos_filter: ArrayView1<'_, f64>,
    analytic: &[Complex64],
    shift: usize,
) -> (Array1<f32>, bool) {
    let len = analytic.len();
    let shift = shift % len.max(1);
    let mut raw_env = Array1::<f32>::zeros(len);
    {
        let raw_env = raw_env.as_slice_mut().unwrap();
        let (head, tail) = raw_env.split_at_mut(shift);
        // np.roll(x, shift): out[shift:] = x[:-shift], out[:shift] = x[-shift:]
        for (o, a) in tail.iter_mut().zip(&analytic[..len - shift]) {
            *o = (a.re as f32).abs();
        }
        for (o, a) in head.iter_mut().zip(&analytic[len - shift..]) {
            *o = (a.re as f32).abs();
        }
    }
    let env = sos_filtfilt_f32(sos_order, sos_filter, raw_env.view());
    let has_zero = env.iter().any(|&v| v == 0.0);
    (env, has_zero)
}
