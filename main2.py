
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ofdm_lib.BitArray import read_file
from ofdm_lib.map_demap import mapping, demapping, get_constellation_order
from ofdm_lib.ofdm_multiplex_demultiplex import OfdmConfig, multiplex_to_ofdm, demultiplex_from_ofdm
from ofdm_lib.randomizer import randomize
from ofdm_lib.Utility import calc_ber
from ofdm_lib.channel import ChannelConfig, agwn_cmplx, freq_shift, time_delay, multi_path

from transmission import signal_tx_operation, signal_channel_propagation, signal_rx_operation, signal_rx_rough_correction


# << configuration >> ----------------------------------------------------
#constellation = "BPSK"
#constellation = "QPSK"
constellation = "16-QAM"

ofdm_cfg = OfdmConfig(
    n_fft=1024,
    n_carrier=400,
    t_guard_div=8,
    frame_sz=40,
    pilot_percent=0.1,
    pilot_amp_r=(4/3),
    constellation=constellation
)

use_randomize = True
rnd_init_state = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

chan_cfg = ChannelConfig(
    snr=12,
    freq_shift=0.2/ofdm_cfg.n_fft,  # from (-30 ... 30)/n_fft
    time_delay=0,
    multipath=[(0, 1), (4, 0.6), (10, 0.3)]
)

# << read data to transmit >> --------------------------------------------
QAM_cells = ofdm_cfg.data_cnt * ofdm_cfg.frame_sz
buffer_sz_bits = QAM_cells * get_constellation_order(constellation)
data_file_name = 'data/War_and_Peace.doc'
input_bit_buffer = read_file(data_file_name, buffer_sz_bits)
# --------------------------------------------------------------------------

tx_out = signal_tx_operation(input_bit_buffer, ofdm_cfg, use_randomize, rnd_init_state)
ch_out = signal_channel_propagation(tx_out, chan_cfg)
correction_out, corr, tg_pos, fo = signal_rx_rough_correction(ch_out, ofdm_cfg)
print("tg_pos={}".format(tg_pos))
print('should be {:.4f}'.format(2*np.pi*(128-tg_pos)/1024))

iq, ofdm_spec = demultiplex_from_ofdm(correction_out, ofdm_cfg, interp_kind='linear')

#out_bits, ofdm_spec = signal_rx_operation(correction_out, ofdm_cfg, use_randomize, rnd_init_state, interp_kind='linear')
print()
print(calc_ber(input_bit_buffer, out_bits))






