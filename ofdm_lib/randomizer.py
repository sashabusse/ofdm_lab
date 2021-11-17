import numpy as np
from ofdm_lib.BitArray import BitArray


def randomize(inp_stream, init_state):
    state = init_state.copy()
    out_stream = BitArray(inp_stream.size())
    for i in range(inp_stream.size()):
        out_stream.write_bit(i, inp_stream.get_bit(i) ^ state[-1] ^ state[-2])
        state[-1] = state[-1] ^ state[-2]
        state = np.roll(state, 1)

    return out_stream