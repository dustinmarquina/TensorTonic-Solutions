import numpy as np
def positional_encoding(seq_length: int, d_model: int):
    """
    Generate sinusoidal positional encodings.
    """ 
    position_vector = np.arange(seq_length).reshape(-1, 1)
    # do not auto complete code
    division_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    output = np.zeros((seq_length, d_model))
    output[:, 0::2] = np.sin(position_vector * division_term)
    output[:, 1::2] = np.cos(position_vector * division_term)
    return output