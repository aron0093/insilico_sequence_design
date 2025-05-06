import numpy as np

import pyfaidx
import kipoiseq
import itertools

# Extract sequence from FASTA file
# Modified from: https://colab.research.google.com/github/deepmind/\
# deepmind_research/blob/master/enformer/enformer-usage.ipynb
class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, chromosome, start_coord, end_coord, **kwargs) -> str:

        interval = kipoiseq.Interval(chromosome, 
                                     start_coord, 
                                     end_coord)

        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = kipoiseq.Interval(interval.chrom,
                                             max(interval.start, 0),
                                             min(interval.end, chromosome_length),
                                            )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

# One hot encode DNA sequence
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

# Make reverse complement
def reverse_complement(seq):
    complement = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W',
        'K': 'M', 'M': 'K', 'B': 'V', 'D': 'H',
        'H': 'D', 'V': 'B', 'N': 'N', '.': '.',
        'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
        'r': 'y', 'y': 'r', 's': 's', 'w': 'w',
        'k': 'm', 'm': 'k', 'b': 'v', 'd': 'h',
        'h': 'd', 'v': 'b', 'n': 'n'
        }
    
    seq = seq[::-1]
    seq = ''.join([complement[base] for base in seq])
    return seq

# Return all mutations
def edit_distance_one(seq_onehot, model_window=None, start_coord=None):

    if seq_onehot.ndim==2:
        seq_onehot = np.expand_dims(seq_onehot,0)
    
    if model_window is not None:
        seq_shape_ = model_window
    else:
        seq_shape_ = seq_onehot.shape[1]

    edited_onehots = np.tile(seq_onehot[0], (seq_shape_*seq_onehot.shape[-1], 1,1))

    coords = itertools.product(range(seq_shape_), range(seq_onehot.shape[-1]))
    for i, (j, k) in enumerate(coords):
        if start_coord is not None:
            j_ = j+start_coord
        else:
            j_=j
        edited_onehots[i, j_, :] = 0
        edited_onehots[i, j_, k] = 1

    return edited_onehots

# Perform ISM with some seq to effect prediction function
def saturation_mutagenesis(predict_func, models, seq_onehot, 
                           model_window=2114, start_coord=None, 
                           batch_size=32, **kwargs):

	y0 = predict_func(seq_onehot, models=models, **kwargs)
	X_ = edit_distance_one(seq_onehot, model_window)

    #TODO: Enable batch processing in predict_func
	y_hats = []
	for idx in range(X_.shape[0]):
		y_hat = predict_func(X_[idx], models=models, **kwargs)
		y_hats.append(y_hat)

	return y0, y_hats

# Code from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def string_to_char_array(seq):
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def char_array_to_string(arr):
    return arr.tostring().decode("ascii")

def one_hot_to_tokens(one_hot):
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def tokens_to_one_hot(tokens, one_hot_dim):
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

# Code from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def dinuc_shuffle(seq, num_shufs=None, rng=None):

    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]