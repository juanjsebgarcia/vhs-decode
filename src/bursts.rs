use std::iter::Sum;
use std::ops::SubAssign;

use numpy::ndarray::{Array2, ArrayView1};
use sci_rs::na::{DVector, RealField};
use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::{pad, sosfilt_dyn, sosfilt_zi_dyn, Pad};

/// sci_rs::signal::filter::sosfiltfilt_dyn with the steady state filter state
/// (sosfilt_zi_dyn, which solves a small linear system per section) computed once for
/// all the lines that are filtered instead of on every call. Same arithmetic otherwise,
/// so the results are identical.
struct SosFiltFilt<T: RealField + Copy> {
    init_sos: Vec<Sos<T>>,
    ntaps: usize,
}

impl<T> SosFiltFilt<T>
where
    T: RealField + Copy + Sum + SubAssign,
{
    fn new(sos: &[Sos<T>]) -> Self {
        let n = sos.len();
        let ntaps = 2 * n + 1;
        let bzeros = sos.iter().filter(|s| s.b[2] == T::zero()).count();
        let azeros = sos.iter().filter(|s| s.a[2] == T::zero()).count();
        let ntaps = ntaps - bzeros.min(azeros);

        let mut init_sos = sos.to_vec();
        sosfilt_zi_dyn::<_, _, Sos<T>>(init_sos.iter_mut());
        SosFiltFilt { init_sos, ntaps }
    }

    fn filtfilt(&self, y: Vec<T>) -> Vec<T> {
        let y_len = y.len();
        let x = DVector::<T>::from_vec(y);
        let (edge, ext) = pad(Pad::Odd, None, x, 0, self.ntaps);

        let x0 = *ext.index(0);
        let mut sos_x = self.init_sos.clone();
        for s in sos_x.iter_mut() {
            s.zi0 *= x0;
            s.zi1 *= x0;
        }
        let y = sosfilt_dyn(ext.iter(), &mut sos_x);

        let y0 = *y.last().unwrap();
        let mut sos_y = self.init_sos.clone();
        for s in sos_y.iter_mut() {
            s.zi0 *= y0;
            s.zi1 *= y0;
        }
        let mut z = sosfilt_dyn(y.iter().rev(), &mut sos_y)
            .into_iter()
            .skip(edge)
            .take(y_len)
            .collect::<Vec<_>>();
        z.reverse();
        z
    }
}

/// Up-converts and band-pass filters the color burst area of several lines.
///
/// For each line i, the samples [starts[i], ends[i]) of chroma are multiplied with the
/// same samples of heterodynes[i], the product is filtered forwards and backwards with
/// sos (the same sosfiltfilt as vhsd_rust.sosfiltfilt), and padding samples are dropped
/// from both ends of the result. Row i of the returned array holds lens[i] valid samples
/// followed by zeros.
///
/// Produces the same values as doing this per line with numpy and sosfiltfilt_rust:
/// the product is computed in the heterodyne's precision T (the f32 chroma samples are
/// widened first when T is f64, as numpy does), and filtered in that precision.
pub fn upconvert_filter_bursts_impl<T>(
    sos: &[Sos<T>],
    chroma: ArrayView1<'_, f32>,
    heterodynes: &[ArrayView1<'_, T>],
    starts: &[usize],
    ends: &[usize],
    padding: usize,
) -> (Array2<T>, Vec<i64>)
where
    T: RealField + Copy + From<f32> + Sum + SubAssign,
{
    let line_count = starts.len();
    let mut filtered_lines = Vec::with_capacity(line_count);
    let filter = SosFiltFilt::new(sos);

    for i in 0..line_count {
        let (start, end) = (starts[i], ends[i].max(starts[i]));
        let heterodyne = heterodynes[i].slice(numpy::ndarray::s![start..end]);
        let chroma_part = chroma.slice(numpy::ndarray::s![start..end]);

        let product: Vec<T> = heterodyne
            .iter()
            .zip(chroma_part.iter())
            .map(|(&h, &c)| h * T::from(c))
            .collect();

        let filtered = filter.filtfilt(product);
        // same as filtered[padding:-padding] in python (empty if padding is 0)
        let kept = if padding == 0 || filtered.len() <= 2 * padding {
            Vec::new()
        } else {
            filtered[padding..filtered.len() - padding].to_vec()
        };
        filtered_lines.push(kept);
    }

    let max_len = filtered_lines.iter().map(|f| f.len()).max().unwrap_or(0);
    let mut bursts = Array2::<T>::zeros((line_count, max_len));
    let mut lens = Vec::with_capacity(line_count);
    for (i, filtered) in filtered_lines.iter().enumerate() {
        for (j, &v) in filtered.iter().enumerate() {
            bursts[[i, j]] = v;
        }
        lens.push(filtered.len() as i64);
    }

    (bursts, lens)
}
