import random as rnd
from lab1 import read_file, mapping, demapping, error_check, get_constellation_order
from lab2_ofdm import OfdmConfig, multiplex_to_ofdm, demultiplex_from_ofdm, randomize
from Utility import print_bit_array, print_iq_points
import numpy as np
import sys
rnd.seed(1)


constellation = "BPSK"
#constellation = "QPSK"
#constellation = "16-QAM"

ofdm_cfg = OfdmConfig(
    n_fft=1024,
    n_carrier=400,
    t_guard_div=8,
    frame_sz=20
)

QAM_cells = ofdm_cfg.n_carrier * ofdm_cfg.frame_sz
buffer_sz_bits = QAM_cells * get_constellation_order(constellation)

use_randomize = True
rnd_init_state = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]


log_to_file = False
log_max_cells = int(16)
log_file_name = "log/" + constellation + ".txt"
if log_to_file:
    sys.stdout = open(log_file_name, 'w')


data_file_name = 'data/War_and_Peace.doc'
input_bit_buffer = read_file(data_file_name, buffer_sz_bits)

print("input bit stream:")
print_bit_array(input_bit_buffer, min(QAM_cells, log_max_cells), get_constellation_order(constellation))
print()

# randomize if needed
if use_randomize:
    tx_rand_bit_buffer = randomize(input_bit_buffer, rnd_init_state)
else:
    tx_rand_bit_buffer = input_bit_buffer

tx_iq_points = mapping(tx_rand_bit_buffer, constellation)

print("TX IQ points:")
print_iq_points(tx_iq_points[:min(QAM_cells, log_max_cells)])
print()

tx_ofdm_iq_points = multiplex_to_ofdm(tx_iq_points, ofdm_cfg)

# канал Lab 4-5
# noiseData = Noise (Tx_OFDM_Signal, SNR); %lab 4 | добавление абгш
# freq_shifted_data = frequency_shift(noiseData, Freq_shift, N_fft,T_guard); % lab 4 | частотный сдвиг
# multi_data = multi_path(freq_shifted_data,channel); % lab 4 | многолучевой прием
# time_shifted_data = delay(multi_data,Time_delay); % lab 4

# приемник
rx_ofdm_iq_points = tx_ofdm_iq_points
rx_IQ_points = demultiplex_from_ofdm(rx_ofdm_iq_points, ofdm_cfg)

print("RX IQ points")
print_iq_points(rx_IQ_points[:min(QAM_cells, log_max_cells)])
print()

rx_rand_bit_buffer = demapping(rx_IQ_points, constellation)
output_bit_buffer = randomize(rx_rand_bit_buffer, rnd_init_state)

print("output bit buffer:")
print_bit_array(output_bit_buffer, min(QAM_cells, log_max_cells), get_constellation_order(constellation))
print()

# deranddata_delay = derandomizator (finishBits_delay, register)
probability = error_check(input_bit_buffer, output_bit_buffer)

print("bit error probability:", probability)
