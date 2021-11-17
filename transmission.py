import random as rnd
import numpy as np
from ofdm_lib.BitArray import read_file
from ofdm_lib.map_demap import mapping, demapping, get_constellation_order
from ofdm_lib.ofdm_multiplex_demultiplex import OfdmConfig, multiplex_to_ofdm, demultiplex_from_ofdm
from ofdm_lib.randomizer import randomize
from ofdm_lib.Utility import calc_ber
from ofdm_lib.channel import ChannelConfig, agwn_cmplx, freq_shift, time_delay, multi_path
from ofdm_lib.auto_corr_analysis import auto_corr_analysis


def signal_tx_operation(input_bit_buffer, ofdm_cfg, use_randomize, rnd_init_state):
    # << randomize if needed >> --------------------------------------------------
    if use_randomize:
        tx_rand_bit_buffer = randomize(input_bit_buffer, rnd_init_state)
    else:
        tx_rand_bit_buffer = input_bit_buffer
    # --------------------------------------------------------------------------

    # << modulate with constellation - OFDM >> ---------------------------------
    tx_iq_points = mapping(tx_rand_bit_buffer, ofdm_cfg.constellation)
    tx_ofdm_iq_points = multiplex_to_ofdm(tx_iq_points, ofdm_cfg)
    # --------------------------------------------------------------------------

    return tx_ofdm_iq_points


def signal_channel_propagation(iq, chan_cfg):
    channel_inp = iq

    Ps = channel_inp.dot(channel_inp.conj()).real/len(channel_inp)
    channel_noisy = multi_path(channel_inp, chan_cfg.multipath) + agwn_cmplx(Ps, chan_cfg.snr, len(channel_inp))

    channel_fr_sh = freq_shift(channel_noisy, chan_cfg.freq_shift)
    channel_delayed = time_delay(channel_fr_sh, chan_cfg.time_delay, Ps, chan_cfg.snr)

    return channel_delayed


def signal_rx_rough_correction(iq, ofdm_cfg):
    corr, tg_pos, freq_offset = auto_corr_analysis(iq, ofdm_cfg, ofdm_cfg.t_guard//2, tg_pos_lvl=0.7, fr_shift_lvl=0.8)

    # frequency shift correction
    iq_corrected = iq * np.exp(-1j*(2*np.pi)*freq_offset*np.arange(len(iq)))

    # time correction
    iq_corrected = iq_corrected[tg_pos:]

    return iq_corrected, corr, tg_pos, freq_offset


def signal_rx_operation(iq, ofdm_cfg, use_randomize, rnd_init_state, interp_kind="linear"):
    data_iq, ofdm_spec = demultiplex_from_ofdm(iq, ofdm_cfg, interp_kind=interp_kind)
    rx_rand_bit_buffer = demapping(data_iq, ofdm_cfg.constellation)
    # --------------------------------------------------------------------------

    # << derandomization >> ----------------------------------------------------
    if use_randomize:
        output_bit_buffer = randomize(rx_rand_bit_buffer, rnd_init_state)
    else:
        output_bit_buffer = rx_rand_bit_buffer
    # --------------------------------------------------------------------------
    return output_bit_buffer, ofdm_spec

