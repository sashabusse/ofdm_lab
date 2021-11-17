import random as rnd
import numpy as np
from matplotlib import pyplot as plt
from ofdm_lib.BitArray import read_file
from ofdm_lib.map_demap import mapping, demapping, get_constellation_order
from ofdm_lib.ofdm_multiplex_demultiplex import OfdmConfig, multiplex_to_ofdm, demultiplex_from_ofdm
from ofdm_lib.randomizer import randomize
from ofdm_lib.Utility import calc_ber
from ofdm_lib.channel import ChannelConfig, agwn_cmplx, freq_shift, time_delay, multi_path
from transmission import signal_transmission
rnd.seed(1)

# << configuration >> ----------------------------------------------------
#constellation = "BPSK"
#constellation = "QPSK"
#constellation = "16-QAM"

ofdm_cfg = OfdmConfig(
    n_fft=1024,
    n_carrier=400,
    t_guard_div=8,
    frame_sz=100,
    pilot_percent=0.2
)

use_randomize = True
rnd_init_state = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

chan_cfg = ChannelConfig(
    snr=30,
    freq_shift=0/ofdm_cfg.n_fft,  # from (-30 ... 30)/n_fft
    time_delay=0,
    multipath=[[0, 1]]
)
# channel config

# --------------------------------------------------------------------------

fig, ax = plt.subplots(1, 1)

for constellation in ["BPSK", "QPSK", "16-QAM"]:
    # << read data to transmit >> --------------------------------------------
    QAM_cells = ofdm_cfg.n_carrier * ofdm_cfg.frame_sz
    buffer_sz_bits = QAM_cells * get_constellation_order(constellation)
    data_file_name = 'data/War_and_Peace.doc'
    input_bit_buffer = read_file(data_file_name, buffer_sz_bits)
    # --------------------------------------------------------------------------

    snr_list = np.arange(0, 30, 1)
    ber = []
    for snr in snr_list:
        chan_cfg.snr = snr
        cur_ber, signals = signal_transmission(
            input_bit_buffer, constellation, ofdm_cfg, chan_cfg,
            use_randomize, rnd_init_state)
        ber.append(cur_ber)

    ax.set_yscale('log')
    ax.plot(snr_list, np.array(ber))
    ax.set_xlabel("SNR dB")
    ax.set_ylabel("BER")
    ax.set_ylim((1e-4, 1))
    ax.grid(True)

ax.legend(["BPSK", "QPSK", "16-QAM"])
plt.show()

