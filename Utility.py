import numpy as np
import matplotlib.pyplot as plt


def gray_code(length_bits):
    res = np.zeros(2**length_bits, int)
    for bit_ind in range(0, length_bits):
        for i in range(0, 2**length_bits):
            mirror = int(i/(2**(bit_ind + 1))) % 2
            val = int(i/(2**bit_ind)) % 2
            res[i] |= (val ^ mirror) << bit_ind

    return res


def save_constellation_plot(symbol_2_iq_map, constellation_order, title="", fname="constellation.jpg"):
    plt.figure().clear()
    plt.title(title)
    re = np.real(symbol_2_iq_map)
    im = np.imag(symbol_2_iq_map)
    plt.scatter(re, im, s=60, c='blue')
    for i in range(len(re)):
        label = ''
        for bit_ind in range(constellation_order):
            label += str((i >> (constellation_order - 1 - bit_ind)) & 1)
        plt.text(re[i] + 0.1, im[i] + 0.1, label)

    plt.xlim(left=min(re) - 1, right=max(re) + 1)
    plt.ylim(bottom=min(re) - 1, top=max(re) + 1)
    plt.grid(True)
    plt.savefig(fname)


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


