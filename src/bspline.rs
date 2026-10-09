//! Evaluation of 1-D B-splines (scipy.interpolate.BSpline.__call__) that can run without the GIL.
//!
//! This follows scipy's implementation (_evaluate_spline / _deBoor_D / _find_interval in
//! scipy/interpolate/src/__fitpack.cc) operation by operation so the results are identical.
//! The only thing that depends on how scipy was compiled is whether the compiler fused the
//! multiply-adds in it (it does on arm64, e.g. macOS wheels). Both variants are provided,
//! the caller picks the one that matches the installed scipy (see vhsdecode/rust_utils.py).

#[inline(always)]
fn mul_add<const FUSED: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Index l such that t[l] <= x < t[l + 1], starting the search at prev_l.
/// Returns None for nan, or x out of the base interval when not extrapolating.
#[inline(always)]
fn find_interval(t: &[f64], k: usize, x: f64, prev_l: usize, extrapolate: bool) -> Option<usize> {
    let n = t.len() - k - 1;
    let tb = t[k];
    let te = t[n];

    if x.is_nan() {
        return None;
    }
    if (x < tb || x > te) && !extrapolate {
        return None;
    }

    let mut l = if k < prev_l && prev_l < n { prev_l } else { k };
    while x < t[l] && l != k {
        l -= 1;
    }
    l += 1;
    while x >= t[l] && l != n {
        l += 1;
    }
    Some(l - 1)
}

/// The k + 1 non-zero values of the m-th derivative of the B-splines at x, for
/// t[ell] <= x < t[ell + 1], written to h (length k + 1). hh is scratch space (length k + 1).
#[inline(always)]
fn de_boor_d<const FUSED: bool>(
    t: &[f64],
    x: f64,
    k: usize,
    ell: usize,
    m: usize,
    h: &mut [f64],
    hh: &mut [f64],
) {
    h[0] = 1.0;
    // k - m "standard" de Boor iterations
    for j in 1..=k.saturating_sub(m) {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n;
            let xb = t[ind];
            let xa = t[ind - j];
            if xb == xa {
                h[n] = 0.0;
                continue;
            }
            let w = hh[n - 1] / (xb - xa);
            h[n - 1] = mul_add::<FUSED>(w, xb - x, h[n - 1]);
            h[n] = w * (x - xa);
        }
    }

    // m "derivative" recursions
    for j in (k + 1).saturating_sub(m).max(1)..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n;
            let xb = t[ind];
            let xa = t[ind - j];
            if xb == xa {
                // (sic) scipy zeroes h[m] here
                h[m] = 0.0;
                continue;
            }
            let w = (j as f64) * hh[n - 1] / (xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
}

fn evaluate_impl<const FUSED: bool>(
    t: &[f64],
    c: &[f64],
    k: usize,
    xs: &[f64],
    nu: usize,
    extrapolate: bool,
    out: &mut [f64],
) {
    let mut h = vec![0.0; k + 1];
    let mut hh = vec![0.0; k + 1];
    let mut interval = k;
    for (x, o) in xs.iter().zip(out.iter_mut()) {
        let x = *x;
        match find_interval(t, k, x, interval, extrapolate) {
            None => {
                *o = f64::NAN;
                // scipy keeps searching from the result of the failed search (-1),
                // which restarts the next search at k.
                interval = k;
            }
            Some(l) => {
                interval = l;
                de_boor_d::<FUSED>(t, x, k, l, nu, &mut h, &mut hh);
                let mut sum = 0.0;
                for a in 0..=k {
                    sum = mul_add::<FUSED>(c[l + a - k], h[a], sum);
                }
                *o = sum;
            }
        }
    }
}

/// Evaluate the nu-th derivative of the B-spline (t, c, k) at xs into out.
/// Requires len(t) >= 2k + 2, len(c) >= len(t) - k - 1, nu <= k.
pub fn bspline_evaluate(
    t: &[f64],
    c: &[f64],
    k: usize,
    xs: &[f64],
    nu: usize,
    extrapolate: bool,
    fused: bool,
    out: &mut [f64],
) {
    if fused {
        evaluate_impl::<true>(t, c, k, xs, nu, extrapolate, out)
    } else {
        evaluate_impl::<false>(t, c, k, xs, nu, extrapolate, out)
    }
}
