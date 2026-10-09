//! Forward-backward filtering with second order sections, bit-identical to
//! sci_rs::signal::filter::sosfiltfilt_dyn (and so to what vhsd_rust.sosfiltfilt
//! returned before), but with one buffer instead of about six and with the filter
//! state kept in registers (for 1 to 10 sections).
//!
//! The arithmetic is the same as sci_rs, operation by operation and in the same order:
//! the odd extension `2 * x[0] - x[i]`, the steady state from sci_rs's own
//! sosfilt_zi_dyn scaled by the first sample, and the direct form II transposed
//! section update. Running the sections for one sample before the next sample, as
//! sci_rs does, keeps every intermediate value the same. Rust does not contract
//! multiplies and adds into FMA, so the rounding is the same as well.

use sci_rs::na::RealField;
use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::{sosfilt_zi_dyn, sosfiltfilt_dyn};
use std::cmp::min;
use std::iter::Sum;
use std::ops::SubAssign;

/// One direct form II transposed section step, as in sci_rs sosfilt_dyn.
#[inline(always)]
fn section_step<F: RealField + Copy>(b: &[F; 3], a: &[F; 3], z: &mut [F; 2], x: F) -> F {
    let x_new = b[0] * x + z[0];
    z[0] = b[1] * x - a[1] * x_new + z[1];
    z[1] = b[2] * x - a[2] * x_new;
    x_new
}

/// All N sections for one sample.
#[inline(always)]
fn sections_step<F: RealField + Copy, const N: usize>(
    b: &[[F; 3]; N],
    a: &[[F; 3]; N],
    z: &mut [[F; 2]; N],
    x: F,
) -> F {
    let mut x = x;
    for k in 0..N {
        x = section_step(&b[k], &a[k], &mut z[k], x);
    }
    x
}

/// Filter `data` in place with N sections (N known at compile time so the state can
/// stay in registers). `forward` selects the direction.
#[inline(never)]
fn run_sections<F: RealField + Copy, const N: usize>(
    sos: &[Sos<F>],
    zi: &[[F; 2]],
    data: &mut [F],
    forward: bool,
) {
    let mut b = [[F::zero(); 3]; N];
    let mut a = [[F::zero(); 3]; N];
    let mut z = [[F::zero(); 2]; N];
    for k in 0..N {
        b[k] = sos[k].b;
        a[k] = sos[k].a;
        z[k] = zi[k];
    }
    if forward {
        for v in data.iter_mut() {
            *v = sections_step(&b, &a, &mut z, *v);
        }
    } else {
        for v in data.iter_mut().rev() {
            *v = sections_step(&b, &a, &mut z, *v);
        }
    }
}

fn run<F: RealField + Copy>(sos: &[Sos<F>], zi: &[[F; 2]], data: &mut [F], forward: bool) {
    match sos.len() {
        1 => run_sections::<F, 1>(sos, zi, data, forward),
        2 => run_sections::<F, 2>(sos, zi, data, forward),
        3 => run_sections::<F, 3>(sos, zi, data, forward),
        4 => run_sections::<F, 4>(sos, zi, data, forward),
        5 => run_sections::<F, 5>(sos, zi, data, forward),
        6 => run_sections::<F, 6>(sos, zi, data, forward),
        7 => run_sections::<F, 7>(sos, zi, data, forward),
        8 => run_sections::<F, 8>(sos, zi, data, forward),
        9 => run_sections::<F, 9>(sos, zi, data, forward),
        10 => run_sections::<F, 10>(sos, zi, data, forward),
        _ => unreachable!(),
    }
}

/// Same result as sci_rs::signal::filter::sosfiltfilt_dyn(input.iter(), sos)
/// (which is used directly for more than 10 sections).
/// Panics like sci_rs if input is not longer than the padding (3 * ntaps).
pub fn sosfiltfilt_exact<F>(input: &[F], sos: &[Sos<F>]) -> Vec<F>
where
    F: RealField + Copy + PartialEq + Sum + SubAssign,
{
    let n = sos.len();
    if n == 0 || n > 10 {
        // Not used by the decoder; keep sci_rs's own implementation for these.
        return sosfiltfilt_dyn(input.iter(), sos);
    }
    let bzeros = sos.iter().filter(|s| s.b[2] == F::zero()).count();
    let azeros = sos.iter().filter(|s| s.a[2] == F::zero()).count();
    let ntaps = 2 * n + 1 - min(bzeros, azeros);
    let edge = ntaps * 3;
    let len = input.len();
    assert!(len > edge);

    // Odd extension, as sci_rs::signal::filter::odd_ext_dyn.
    let two = F::one() + F::one();
    let first = input[0];
    let last = input[len - 1];
    let mut data = Vec::with_capacity(len + 2 * edge);
    data.extend((0..edge).map(|i| two * first - input[edge - i]));
    data.extend_from_slice(input);
    data.extend((0..edge).map(|i| two * last - input[len - 2 - i]));

    let mut init_sos = sos.to_vec();
    sosfilt_zi_dyn::<_, _, Sos<F>>(init_sos.iter_mut());

    let x0 = data[0];
    let zi: Vec<[F; 2]> = init_sos.iter().map(|s| [s.zi0 * x0, s.zi1 * x0]).collect();
    run(sos, &zi, &mut data, true);

    let y0 = data[data.len() - 1];
    let zi: Vec<[F; 2]> = init_sos.iter().map(|s| [s.zi0 * y0, s.zi1 * y0]).collect();
    run(sos, &zi, &mut data, false);

    data.truncate(edge + len);
    data.drain(..edge);
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<F>(sos: &[Sos<F>], input: &[F])
    where
        F: RealField + Copy + PartialEq + Sum + SubAssign + std::fmt::Debug,
    {
        let expected = sosfiltfilt_dyn(input.iter(), sos);
        let got = sosfiltfilt_exact(input, sos);
        assert_eq!(expected.len(), got.len());
        for (e, g) in expected.iter().zip(got.iter()) {
            // Bitwise: also distinguishes -0.0 from 0.0.
            assert!(e == g && e.is_sign_negative() == g.is_sign_negative(), "{e:?} != {g:?}");
        }
    }

    #[test]
    fn matches_sci_rs() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for n in 0..13 {
            let sos: Vec<Sos<f64>> = (0..n)
                .map(|k| {
                    let a1 = 0.5 * rnd();
                    let a2 = if k == 0 { 0.0 } else { 0.3 * rnd() };
                    Sos::new([rnd(), rnd(), if k == 0 { 0.0 } else { rnd() }], [1.0, a1, a2])
                })
                .collect();
            let sos32: Vec<Sos<f32>> = sos
                .iter()
                .map(|s| {
                    Sos::new(
                        [s.b[0] as f32, s.b[1] as f32, s.b[2] as f32],
                        [s.a[0] as f32, s.a[1] as f32, s.a[2] as f32],
                    )
                })
                .collect();
            for len in [3 * (2 * n + 1) + 1, 100, 1000, 32768] {
                let x: Vec<f64> = (0..len).map(|_| rnd() * 100.0).collect();
                let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
                check(&sos, &x);
                check(&sos32, &x32);
            }
        }
    }
}
