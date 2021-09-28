import numpy as np
from BitArray import BitArray


class OfdmConfig:
    def __init__(self, n_fft, n_carrier, t_guard_div, frame_sz):
        self.n_fft = n_fft
        self.n_carrier = n_carrier
        self.t_guard_div = t_guard_div
        self.frame_sz = frame_sz

        self.t_guard = int(self.n_fft/self.t_guard_div)
        self.t_sym = self.n_fft + self.t_guard

        # could change
        self.t_frame = self.t_sym * frame_sz


def multiplex_to_ofdm(iq_pts_in, ofdm_cfg):
    ofdm_sym_spectrum = iq_pts_in.reshape((-1, ofdm_cfg.n_carrier))
    ofdm_sym_spectrum = np.pad(
        ofdm_sym_spectrum,
        ((0, 0), (1, ofdm_cfg.n_fft - ofdm_cfg.n_carrier - 1)),
        mode='constant', constant_values=((0, 0), (0, 0))
    )
    ofdm_sym = np.fft.ifft(ofdm_sym_spectrum, axis=1)

    # insert guard
    ofdm_sym = np.pad(ofdm_sym, ((0, 0), (ofdm_cfg.t_guard, 0)), mode="wrap")

    # convert to 1d stream
    iq_pts_out = ofdm_sym.reshape((-1,))
    return iq_pts_out


def demultiplex_from_ofdm(iq_pts_in, ofdm_cfg):
    ofdm_sym = iq_pts_in.reshape((-1, ofdm_cfg.t_sym))
    # remove guard
    ofdm_sym = ofdm_sym[:, ofdm_cfg.t_guard:]

    ofdm_sym_spectrum = np.fft.fft(ofdm_sym, axis=1)

    iq_pts_out = ofdm_sym_spectrum[:, 1:1+ofdm_cfg.n_carrier].reshape((-1))
    return iq_pts_out


def randomize(inp_stream, init_state):
    state = init_state.copy()
    out_stream = BitArray(inp_stream.size())
    for i in range(inp_stream.size()):
        out_stream.write_bit(i, inp_stream.get_bit(i) ^ state[0])
        state = np.roll(state, 1)
        state[0] ^= state[-1]

    return out_stream

