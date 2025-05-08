import commpy
import numpy as np
import math


def pass_channel(modulated_bits, snr, channel_type):

    input_shape = modulated_bits.shape
    modulated_bits = modulated_bits.reshape(-1)

    if channel_type == 'awgn':
        bits_with_noise = commpy.awgn(modulated_bits, snr)

    elif channel_type == 'rayleigh':
        N = len(modulated_bits)
        h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        modulated_bits_times_channel_coefficients = modulated_bits * h
        modulated_bits_times_channel_coefficients_with_noise = commpy.awgn(modulated_bits_times_channel_coefficients, snr)
        bits_with_noise = modulated_bits_times_channel_coefficients_with_noise / h

    elif channel_type == 'rician':
        K = 100
        N = len(modulated_bits)
        m = np.random.randn(N, 1)
        t = np.random.randn(N, 1)
        Complex_Mat = 1j * m[1, :]
        Complex_Mat += t[:, 1]
        h = math.sqrt(K/K+1) + math.sqrt(1/K+1) * Complex_Mat / math.sqrt(2)
        s = modulated_bits * h
        r = commpy.awgn(s, snr)
        bits_with_noise = r / h

    else:
        raise ValueError("Unsupported channel type.")
    
    bits_with_noise = bits_with_noise.reshape(input_shape)
    
    return bits_with_noise