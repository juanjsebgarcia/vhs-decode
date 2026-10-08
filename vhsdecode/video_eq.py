import numpy as np
import scipy.signal as sps
import vhsdecode.utils as utils
from vhsdecode.linear_filter import FiltersClass


class VideoEQ:
    """Sharpness control based on format paremers and sharpness setting"""

    def __init__(self, decoder_params, sharpness_level, freq_hz):
        # sharpness filter / video EQ
        iir_eq_loband = utils.firdes_highpass(
            freq_hz,
            decoder_params["video_eq"]["loband"]["corner"],
            decoder_params["video_eq"]["loband"]["transition"],
            decoder_params["video_eq"]["loband"]["order_limit"],
        )

        self._video_eq_filter = {
            0: FiltersClass(iir_eq_loband[0], iir_eq_loband[1], freq_hz),
            # 1: FiltersClass(iir_eq_hiband[0], iir_eq_hiband[1], freq_hz),
        }

        # Initial filter state for the forward-only edge filter below, scaled per
        # block. Not carried between calls: demodblock runs concurrently on the
        # DemodCache worker threads in no fixed order, so carried state made
        # the output depend on thread scheduling.
        self._eq_b, self._eq_a = iir_eq_loband[0], iir_eq_loband[1]
        self._eq_zi = sps.lfilter_zi(self._eq_b, self._eq_a)

        self._gain = decoder_params["video_eq"]["loband"]["order_limit"]
        self._sharpness_level = sharpness_level

    def filter_video(self, demod):
        """It enhances the upper band of the video signal"""
        overlap = 10  # how many samples the edge distortion produces
        ha = self._video_eq_filter[0].filtfilt(demod)
        hb, _ = sps.lfilter(
            self._eq_b, self._eq_a, demod[:overlap], zi=self._eq_zi * demod[0]
        )
        hc = np.concatenate(
            (hb[:overlap], ha[overlap:])
        )  # edge distortion compensation, needs check
        hf = np.multiply(self._gain, hc)

        gain = self._sharpness_level
        result = np.multiply(np.add(np.roll(np.multiply(gain, hf), 0), demod), 1)

        return result
