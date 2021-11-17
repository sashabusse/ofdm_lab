import numpy as np


def calc_ccdf(iq_pts, ofdm_cfg, interp_mul):
    ofdm_sym_spectrum = iq_pts.reshape((-1, ofdm_cfg.n_carrier))
    ofdm_sym_spectrum = np.pad(
        ofdm_sym_spectrum,
        ((0, 0), (1, (ofdm_cfg.n_fft * interp_mul) - ofdm_cfg.n_carrier - 1)),
        mode='constant', constant_values=((0, 0), (0, 0))
    )
    ofdm_sym = np.fft.ifft(ofdm_sym_spectrum, axis=1)

    papr = np.max(np.abs(ofdm_sym)**2, axis=1)/np.mean(np.abs(ofdm_sym)**2, axis=1)
    papr = 10*np.log10(papr)
    papr = np.sort(papr)
    return papr, 1-((np.arange(len(papr)) + 0.5)/len(papr))

