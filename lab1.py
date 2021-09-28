import numpy as np
from Utility import gray_code, save_constellation_plot
from BitArray import BitArray


# read buffer_sz_bits bits from file
def read_file(file_name, buffer_sz_bits):
    bit_arr = BitArray(buffer_sz_bits)
    bytes_to_read = int(buffer_sz_bits/8)
    if buffer_sz_bits % 8:
        bytes_to_read += 1

    fd = open(file_name, "rb")
    data = fd.read(bytes_to_read)
    fd.close()

    bit_arr.init_from_bytes(data)
    return bit_arr


def get_constellation_order(constellation):
    if constellation == "BPSK":
        return 1
    elif constellation == "QPSK":
        return 2
    elif constellation == "16-QAM":
        return 4
    else:
        assert False, "such constellation not supported"


# returns array of complex that maps bit patterns (symbols) to IQ
def get_symbol_2_iq_map(constellation):
    # hardcode values for BPSK and QPSK
    if constellation == "BPSK":
        return np.array([
            -1,
            1
        ])
    if constellation == "QPSK":
        return np.array([
            -1-1j,
            -1+1j,
            1-1j,
            1+1j
        ])

    # general algorithm to generate 2 dimensional gray codes and map
    order = get_constellation_order(constellation)

    g_code = gray_code(int(order/2))
    symbol_2_iq_map = np.zeros((order**2, ), complex)

    for i in range(0, order):
        for j in range(0, order):
            symbol_2_iq_map[(g_code[i] << int(order/2)) + g_code[j]] = \
                (i - order/2 + 0.5) * 2 + (j - order/2 + 0.5) * (-2j)
    return symbol_2_iq_map


def constellation_norm(constellation_map):
    return np.sqrt( np.average(np.abs(constellation_map)**2) )


def mapping(input_bit_buffer, constellation):
    # get chosen constellation map
    symbol_2_iq_map = get_symbol_2_iq_map(constellation)

    order = get_constellation_order(constellation)
    save_constellation_plot(symbol_2_iq_map, order,
                            "mapping constellation (not normalized)",
                            "log/" + constellation + ".jpg")

    # normalize constellation
    norm = constellation_norm(symbol_2_iq_map)
    symbol_2_iq_map = symbol_2_iq_map/norm
    save_constellation_plot(symbol_2_iq_map, order,
                            "mapping constellation (normalized / norm = {0:0.2f})".format(norm),
                            "log/" + constellation + "_normalized.jpg")

    # generate iq points based on chosen constellation map
    iq_points = []
    for i in range(int(input_bit_buffer.size() / order)):
        symbol = input_bit_buffer.get_symbol(i, order)
        iq_points.append(symbol_2_iq_map[symbol])

    return np.array(iq_points)


def demapping(rx_iq_points, constellation):
    # get chosen constellation map
    symbol_2_iq_map = get_symbol_2_iq_map(constellation)
    # normalize constellation
    norm = constellation_norm(symbol_2_iq_map)
    symbol_2_iq_map = symbol_2_iq_map/norm

    order = get_constellation_order(constellation)
    output_bit_buffer = BitArray(len(rx_iq_points) * order)

    # demodulate iq to bits
    for i in range(0, len(rx_iq_points)):
        symbol = np.argmin( np.abs(symbol_2_iq_map - rx_iq_points[i]) )
        output_bit_buffer.write_symbol(i, symbol, order)

    return output_bit_buffer


def error_check(input_bit_buffer, output_bit_buffer):
    assert input_bit_buffer.size() == output_bit_buffer.size(), "len of input and output are different"
    err = 0
    for i in range(input_bit_buffer.size()):
        err += input_bit_buffer.get_bit(i) ^ output_bit_buffer.get_bit(i)
    return err/input_bit_buffer.size()







