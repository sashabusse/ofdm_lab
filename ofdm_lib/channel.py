import numpy as np


class ChannelConfig:
    def __init__(self, snr, freq_shift, time_delay, multipath):
        self.snr = snr
        self.freq_shift = freq_shift
        self.time_delay = time_delay
        self.multipath = multipath


def agwn_cmplx(Ps, snr, sz):
    Pn = Ps / (10 ** (snr / 10))
    return np.random.normal(0, np.sqrt(Pn/2), sz) + 1j * np.random.normal(0, np.sqrt(Pn/2), sz)


def freq_shift(signal, df):
    k = np.arange(len(signal))
    signal *= np.exp(1j*(2*np.pi)*df*k)
    return signal


def time_delay(signal, time_delay, Ps, snr):
    return signal[time_delay:]


def multi_path(signal, delay_profile):
    result = np.zeros(signal.shape, dtype=signal.dtype)
    for delay, amp in delay_profile:
        result += amp*np.pad(signal[:len(signal)-delay], (delay, 0), mode='constant', constant_values=(0, 0))

    return result


