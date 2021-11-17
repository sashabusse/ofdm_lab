import numpy as np


def calc_ber(input_bit_buffer, output_bit_buffer):
    lost = input_bit_buffer.size() - output_bit_buffer.size()
    return np.sum((input_bit_buffer.data[lost:] ^ output_bit_buffer.data))/output_bit_buffer.size()


def print_bit_array(bit_array, symbol_cnt, symbol_sz_bits):
    for bit_ind in range(symbol_cnt * symbol_sz_bits):
        # print delimiter before symbol
        if bit_ind % symbol_sz_bits == 0:
            print(" ", end="")

        print(bit_array.get_bit(bit_ind), end="")
    print()


def print_iq_points(iq_points):
    re = np.real(iq_points)
    im = np.imag(iq_points)
    for i in range(len(iq_points)):
        print("({0:0.2f} + {1:0.2f}j)".format(re[i], im[i]), end=" ")
    print()


