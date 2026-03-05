import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model // 2)[np.newaxis, :]
    div = np.power(10000, 2 * i / d_model)
    PE = np.zeros((seq_length, d_model))
    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div)
    return PE
    pass