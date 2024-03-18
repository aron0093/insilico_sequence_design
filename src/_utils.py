import numpy as np

import pyfaidx
import kipoiseq

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
        
