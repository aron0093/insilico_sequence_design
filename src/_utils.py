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

# Return all mutations
def edit_distance_one(seq_onehot, model_window=None):

    if seq_onehot.ndim==2:
        seq_onehot = np.expand_dims(seq_onehot,0)
    
    if model_window is not None:
        seq_shape_ = model_window
    else:
        seq_shape_ = seq_onehot.shape[1]

    edited_onehots = np.tile(seq_onehot[0], (seq_shape_*seq_onehot.shape[-1], 1,1))

    coords = itertools.product(range(seq_shape_), range(seq_onehot.shape[-1]))
    for i, (j, k) in enumerate(coords):
        edited_onehots[i, j, :] = 0
        edited_onehots[i, j, k] = 1

    return edited_onehots

# Perform ISM with some seq to effect prediction function
def saturation_mutagenesis(predict_func, model, seq_onehot, 
                           model_window=2114, batch_size=32, 
						   **kwargs):

	y0 = predict_func(seq_onehot, model)
	X_ = edit_distance_one(seq_onehot, model_window)
	y_hat = predict_func(X_, model=model)

	return y0, y_hat
