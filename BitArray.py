import numpy as np


# container for bits with useful features
class BitArray:
    def __init__(self, sz_bits):
        self.data = np.zeros((sz_bits,), int)

    # loads BitArray with data from bytes MSB first by default
    def init_from_bytes(self, byte_array):
        assert len(byte_array) * 8 >= len(self.data), "byte array is too short"
        for i in range(len(self.data)):
            byte_ind = int(i/8)
            bit_ind = i % 8
            self.data[i] = (byte_array[byte_ind] >> (7 - bit_ind)) & 1

    def size(self):
        return len(self.data)

    def write_bit(self, ind, val):
        self.data[ind] = val

    # stores symbol_sz_bits bits in a row in apropriate position
    def write_symbol(self, ind, val, symbol_sz_bits):
        offset = symbol_sz_bits*ind
        for i in range(symbol_sz_bits):
            self.data[offset + i] = (val >> (symbol_sz_bits - 1 - i)) & 1

    def get_bit(self, ind):
        return self.data[ind]

    # arrange symbol_sz_bits bits as int
    def get_symbol(self, ind, symbol_sz_bits):
        result = int(0)
        offset = symbol_sz_bits*ind
        for bit_ind in range(symbol_sz_bits):
            result = (result << 1) | self.data[bit_ind + offset]

        return result

