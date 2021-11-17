import numpy as np
from ofdm_lib.map_demap import constellation_max_amp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


class OfdmConfig:
    def __init__(self, n_fft, n_carrier, t_guard_div, frame_sz, pilot_percent, pilot_amp_r, constellation):
        self.n_fft = n_fft
        self.n_carrier = n_carrier
        self.t_guard_div = t_guard_div
        self.frame_sz = frame_sz

        self.t_guard = int(self.n_fft/self.t_guard_div)
        self.t_sym = self.n_fft + self.t_guard

        # could change
        self.t_frame = self.t_sym * frame_sz

        self.pilot_percent = pilot_percent
        pilot_dist = (n_carrier-1)/(int(self.pilot_percent * (n_carrier-1)) - 1)

        self.pilot_ind = np.rint(np.arange(1, n_carrier + 1, pilot_dist)).astype(int)
        self.pilot_cnt = len(self.pilot_ind)

        self.data_ind = np.setdiff1d(np.arange(1, self.n_carrier), self.pilot_ind)
        self.data_cnt = len(self.data_ind)

        self.constellation = constellation
        self.pilot_amp_r = pilot_amp_r
        self.pilot_amp = self.pilot_amp_r * constellation_max_amp(self.constellation)


def multiplex_to_ofdm(iq_pts_in, ofdm_cfg):
    ofdm_sym_data = iq_pts_in.reshape((-1, ofdm_cfg.data_cnt))
    ofdm_sym_spectrum = np.zeros((ofdm_sym_data.shape[0], ofdm_cfg.n_fft), dtype=complex)

    ofdm_sym_spectrum[:, ofdm_cfg.data_ind] = ofdm_sym_data
    ofdm_sym_spectrum[:, ofdm_cfg.pilot_ind] = ofdm_cfg.pilot_amp * np.exp(1j*np.pi*(np.arange(ofdm_cfg.pilot_cnt) - 0.5))

    ofdm_sym = np.fft.ifft(ofdm_sym_spectrum, axis=1)

    # insert guard
    ofdm_sym = np.pad(ofdm_sym, ((0, 0), (ofdm_cfg.t_guard, 0)), mode="wrap")

    # convert to 1d stream
    iq_pts_out = ofdm_sym.reshape((-1,))
    return iq_pts_out


def demultiplex_from_ofdm(iq_pts_in, ofdm_cfg, interp_kind='linear'):
    # take out of stream ofdm symbols
    iq_pts_in = np.hstack((
        iq_pts_in[0:((len(iq_pts_in) + ofdm_cfg.t_guard)//ofdm_cfg.t_sym)*ofdm_cfg.t_sym - ofdm_cfg.t_guard],
        np.zeros(ofdm_cfg.t_guard)
    ))
    ofdm_sym = iq_pts_in.reshape((-1, ofdm_cfg.t_sym))
    ofdm_sym = ofdm_sym[:, :ofdm_cfg.n_fft]

    ofdm_sym_spectrum = np.fft.fft(ofdm_sym, axis=1)

    # working with pilots
    # accurate synchronization (no averaging over different ofdm symbol)
    for sym_ind in range(ofdm_sym_spectrum.shape[0]):
        shifts = []
        for i in range(1, ofdm_cfg.pilot_cnt):
            if i % 2 == 1:
                dph_tx = np.pi
            else:
                dph_tx = -np.pi

            dph_rx = \
                np.angle(ofdm_sym_spectrum[sym_ind][ofdm_cfg.pilot_ind[i]]) - \
                np.angle(ofdm_sym_spectrum[sym_ind][ofdm_cfg.pilot_ind[i-1]])

            diff = dph_tx - dph_rx
            if diff < 0:
                diff += 2*np.pi

            diff = diff % (2*np.pi)

            shifts.append(diff/(ofdm_cfg.pilot_ind[i] - ofdm_cfg.pilot_ind[i-1]))

        shift = np.mean(shifts)
        ofdm_sym_spectrum[sym_ind] *= np.exp(1j*shift*np.arange(ofdm_sym_spectrum.shape[1]))
        ofdm_sym_spectrum[sym_ind] *= np.exp(-1j*(np.angle(ofdm_sym_spectrum[sym_ind][1]) + np.pi/2))

        # channel characteristic interpolation
        if not (interp_kind is None):
            pilot_amplitudes = ofdm_sym_spectrum[sym_ind, ofdm_cfg.pilot_ind]
            interp_data = np.abs(pilot_amplitudes)/ofdm_cfg.pilot_amp
            interp = interp1d(ofdm_cfg.pilot_ind,
                              #pilot_amplitudes/(ofdm_cfg.pilot_amp * np.exp(1j*np.pi*(np.arange(ofdm_cfg.pilot_cnt) - 0.5))),
                              interp_data,
                              kind=interp_kind)

            ofdm_sym_spectrum[sym_ind:, 1:ofdm_cfg.n_carrier + 1] /= interp(np.arange(1, ofdm_cfg.n_carrier + 1))

    iq_pts_out = ofdm_sym_spectrum[:, ofdm_cfg.data_ind].reshape((-1, ))
    return iq_pts_out, ofdm_sym_spectrum

