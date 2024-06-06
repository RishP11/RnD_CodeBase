# Required dependencies 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools

# Created by RISHABH POMAJE for the paper #Learning to communicate over fading channels#
# Alias used is "csl" while importing
"""
This script defines various functions defined which are to be used as black boxes to simulate a chain of actual digital 
communication system (wired/ wireless) and analyse its various aspects such as bit error rate, symbol error rate,
studying impacts of various modulation and coding schemes. The most used data structure is np.ndarray since there 
is a lot of efficient calculations available from NumPy. 
"""

def qfunc(x) :
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

#
# Functions to be used at the transmitter
#

def hamming_encoder(input_stream, gnrtr_matrix) :
    """
    Function that first generates a codebook depending on the generator matrix 
    (provided as a numpy multi-dimensional ndarray). 
    Using that codebook, it performs the channel coding using Hamming codes.
    
    Args:
    - input_stream (array-like): The input bit stream to be encoded.
    - gnrtr_matrix (ndarray): The generator matrix used for encoding.
    
    Returns:
    - ndarray: The channel coded bit stream generated as per the Generator Matrix specified.
    
    Note:
    - The length of the input bit stream should be compatible with the generator matrix.
    - The generator matrix should be properly constructed for the desired Hamming code.
    """
    input_size = len(input_stream)
    # k = length of the input packet ; n = length of the output packet
    k, n = np.shape(gnrtr_matrix)
    # Generating the codebook :
    code_book = np.array([], dtype=int)
    possible_inputs = [list(i) for i in itertools.product([0, 1], repeat=k)]
    for i in range(2 ** k) :
        code_word = np.array(np.matmul(possible_inputs[i], gnrtr_matrix) % 2)
        code_book = np.append(code_book, code_word) 
    code_book = code_book.reshape(2 ** k, n)
    # Encoding the stream using the codebook :
    index = 0
    channel_coded_stream = []
    for i in range(input_size // k) :
        temp = input_stream[index:index+k]
        asgn_id = np.sum(temp * 2 ** np.arange(k - 1, -1, -1))
        channel_coded_stream.append(code_book[asgn_id])
        index = index + k
        
    channel_coded_stream = np.array(channel_coded_stream)

    return channel_coded_stream.flatten()

def pulse_pos_modulation(input_stream, energy_per_symbol) :
    """
    Function to perform Pulse Position Modulation with amplitude energy_per_symbol in one symbol duration of 
    the two slots available per bit depending on the bit.
    
    Args:
    - input_stream (array-like): The input bit stream to be modulated.
    - energy_per_symbol (float): The energy per symbol for Pulse Position Modulation.
    
    Returns:
    - ndarray: An array representing the modulated signal, where each index corresponds to the discrete 
    symbol time unit corresponding to the two time slots per bit. Packaged as [].
    
    Note:
    - Each bit in the input stream is mapped to two symbol durations, with the energy_per_symbol divided 
    between the two time slots depending on the bit value (0 or 1).
    """
    output_stream = np.zeros(len(input_stream) * 2)
    coord = np.sqrt(energy_per_symbol)
    index = 0
    for i in range(len(input_stream)) :
        output_stream[index] = coord * (1 - input_stream[i])
        output_stream[index+1] = coord * input_stream[i]
        index = index + 2

    return output_stream

def BPSK_mapper(input_stream, energy_per_symbol) :
    """
    Function to perform constellation mapping on the input stream as per Binary Phase Shift Keying (BPSK) 
    with the given energy per symbol.
    
    Args:
    - input_stream (array-like): The input bit stream to be mapped.
    - energy_per_symbol (float): The energy per symbol for BPSK modulation.
    
    Returns:
    - ndarray: An array consisting of possibly complex symbols, each with magnitude sqrt(energy_per_symbol), 
    where positive values represent bit 0 and negative values represent bit 1.
    """
    output_stream = np.where(input_stream == 0, np.sqrt(energy_per_symbol), -1 * np.sqrt(energy_per_symbol))

    return output_stream

#
# Functions to be used at the receiver
#
def BPSK_demapper(rx_stream) :
    """
    Function to perform Binary Phase Shift Keying (BPSK) constellation demapping.

    Args:
    - rx_stream (ndarray): The received signal.

    Returns:
    - ndarray: The bit stream demodulated using the Minimum Distance method.

    Note:
    - This function assumes the received signal is possibly complex, but the demapping is performed using 
    the real part as the sufficient statistic.
    - Demapping is done using the Minimum Distance method, where a bit is decoded based on whether the 
    real part of the received signal is greater than 0.
    """
    rx_bits = np.where(np.real(rx_stream) > 0 , 0, 1)
    return rx_bits

def square_law_detector(rx_signal):
    """
    Function to decide the bit directly depending on the received signal by using the square law.
    (Non-coherent detection)

    Args:
    - rx_signal (ndarray): The received signal, represented as two lists, one for the time 0 slot and 
    one for the time 1 slot.

    Returns:
    - ndarray: The bit stream as an ndarray, obtained by performing square law detection.

    Note:
    - This function assumes non-coherent detection, where the bit decision is made directly based on 
    the received signal without needing phase synchronization.
    """
    estimated_message = np.ones(len(rx_signal) // 2, dtype=int)

    # Perform vectorized comparison
    rx_signal_reshaped = rx_signal.reshape(-1, 2)
    estimated_message = np.where(np.abs(rx_signal_reshaped[:, 0]) > np.abs(rx_signal_reshaped[:, 1]), 0, 1)

    return estimated_message


def hamming_decoder(rx_stream, parity_chk_matrix):
    """
    Function to perform hard decision syndrome decoding for Hamming codes, given a bit stream and parity check matrix 
    as input.
    
    Args:
    - rx_stream (ndarray): The received bit stream to be decoded.
    - parity_chk_matrix (ndarray): The parity check matrix for the Hamming code.

    Returns:
    - ndarray: The 't-error' corrected bit stream.
    
    Note:
    - The matrices provided as inputs to this function must be ndarray with dtype = int.
    - The current implementation is matrix based and hence inefficient.
    """
    # Get the size of the input stream and dimensions of the parity check matrix
    input_size = len(rx_stream)
    k_1, n = parity_chk_matrix.shape
    k = n - k_1
    decoded_stream = np.zeros((input_size // n, k), dtype=int)
    # Generating the syndrome book :
    possible_single_errors = np.eye(n, dtype=int) 
    syndrome_book = parity_chk_matrix.T
    # Performing the syndrome decoding :
    num_blocks = input_size // n 
    for idx in range(num_blocks):
        rx_packet = rx_stream[idx * n: (idx + 1) * n]  # Extract packet for processing
        syndrome = np.matmul(rx_packet, parity_chk_matrix.T) % 2
        if np.sum(syndrome) != 0 :
        # Correcting 1-bit errors :
            for j in range(n):
                if np.array_equal(syndrome, syndrome_book[j]):
                    rx_packet = np.bitwise_xor(rx_packet, possible_single_errors[j])
        decoded_stream[idx] = rx_packet[:k]  # Append the corrected packet
        
    return decoded_stream.flatten()

def ML_Detection(rx_signal, G, E_sym, n, k) :
    """ 
    Function to perform Maximum Likelihood Decoding for Hamming Code.
    
    Args:
    - rx_signal (ndarray): The received signal.
    - G (ndarray): The generator matrix for the Hamming code.
    - E_sym (float): The energy per symbol.
    - n (int): Length of the codeword.
    - k (int): Length of the message.
    
    Returns:
    - ndarray: The Maximum Likelihood Decoding estimates.
    """
    # Matrix of all possible codewords :
    code_book = np.array([], dtype=int)
    possible_inputs = [list(i) for i in itertools.product([0, 1], repeat=k)]
    for i in range(2 ** k) :
        code_word = np.array(np.matmul(possible_inputs[i], G) % 2)
        code_book = np.append(code_book, code_word)
    C = BPSK_mapper(code_book, E_sym)  
    code_book = code_book.reshape(2 ** k, n)
    C = C.reshape(2 ** k, n)
    
    # Finding the MLD Estimates :
    n_blocks = len(rx_signal) // n
    estimates = []
    for i in range(n_blocks):
        start_idx = i * n
        end_idx = (i + 1) * n
        received_segment = rx_signal[start_idx:end_idx]
        distances = [np.linalg.norm(C[j] - received_segment) for j in range(2**k)]
        nearest_codeword_idx = np.argmin(distances)
        nearest_codeword = code_book[nearest_codeword_idx]  
        estimates.append(np.array(nearest_codeword[0:k])) # Extracting the first 4 bits
    
    return np.array(estimates).flatten()

def calcBLER(original_message, estimated_message, block_length) :
    """ 
    Function to calculate the block error rate.
    
    Args:
    - original_message (ndarray): The original bit stream.
    - estimated_message (ndarray): The estimated bit stream.
    - block_length (int): The length of each block.
    
    Returns:
    - list: A list containing the number of block errors and the block error rate.
    """
    n_errors = 0
    n_blocks = len(original_message) // block_length
    idx = 0
    for _ in range(n_blocks) :
        if not np.all(original_message[idx:idx+block_length] == estimated_message[idx:idx+block_length]) :
            n_errors += 1 
        idx += block_length  
    return [n_errors, n_errors / n_blocks]      