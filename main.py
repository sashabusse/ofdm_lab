import random as rnd
from lab1 import read_file, mapping, demapping, error_check, get_constellation_order
from Utility import print_bit_array, print_iq_points
import sys
rnd.seed(1)


#constellation = "BPSK"
#constellation = "QPSK"
constellation = "16-QAM"

log_to_file = True
log_max_cells = 16
log_file_name = "log/" + constellation + ".txt"
if log_to_file:
    sys.stdout = open(log_file_name, 'w')


data_file_name = 'data/War_and_Peace.doc'
QAM_cells = 10

# Посчитать, сколько бит посчитать, в зависимости от количества точек созвазедия
buffer_sz_bits = QAM_cells * get_constellation_order(constellation)

# N_carrier = 400; #lab 2
# N_fft = 1024; #lab 2
# T_guard = 0; #lab 2

input_bit_buffer = read_file(data_file_name, buffer_sz_bits)

print("input bit stream:")
print_bit_array(input_bit_buffer, min(QAM_cells, log_max_cells), get_constellation_order(constellation))
print()

# randomizer (bits, register) # lab 2
tx_IQ_points = mapping(input_bit_buffer, constellation)

print("TX IQ points:")
print_iq_points(tx_IQ_points[:min(QAM_cells, log_max_cells)])
print()

# OFDM_symbols = OFDM_Mod(IQ_points, N_fft, N_carrier); # lab 2
# Tx_OFDM_Signal = signal_generator (OFDM_symbols, T_guard); # lab 2

# канал Lab 4-5
# noiseData = Noise (Tx_OFDM_Signal, SNR); %lab 4 | добавление абгш
# freq_shifted_data = frequense_shift(noiseData, Freq_shift, N_fft,T_guard); % lab 4 | частотный сдвиг
# multi_data = multipath(freq_shifted_data,channel); % lab 4 | многолучевой прием
# time_shifted_data = delay(multi_data,Time_delay); % lab 4



# приемник

# Rx_OFDM_Signal = OFDM_Signal_Demod(Tx_OFDM_Signal, T_guard);_
rx_IQ_points = tx_IQ_points

print("RX IQ points")
print_iq_points(rx_IQ_points[:min(QAM_cells, log_max_cells)])
print()

output_bit_buffer = demapping(rx_IQ_points, constellation)

print("output bit buffer:")
print_bit_array(output_bit_buffer, min(QAM_cells, log_max_cells), get_constellation_order(constellation))
print()

# deranddata_delay = derandomizator (finishBits_delay, register)
probability = error_check(input_bit_buffer, output_bit_buffer)

print("bit error probability:", probability)
