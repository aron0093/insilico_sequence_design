import os

import kipoiseq
from _utils import FastaStringExtractor, one_hot_encode

import logging

# Extract reference sequence
def extract_refseq(chromosome, insert_coord, fasta=None,
                   model_window=None, window_padding=0):

    '''
    Extract reference sequence centered around insert site.

    ARGS
        chromosome: {'chr{i}' type(i) is int}
            Chromosome where insertion site is found.
        insert_coord: int
            DNA sequence coordinate of the inserion site. 
        fasta: str or FastaStringExtractor
            Genomic reference in FASTA format.
        model_window: int (default: None)
            Input sequence length of the predictor model.
        window_padding: int (default: 0)
            Increase extracted sequence length (model_window) on either side.

    RETURNS
        DNA sequence string.
    
    '''

    # Check/load params
    mid_point = int(model_window/2)
    
    if type(fasta) is str:
        if os.path.exists(fasta):
            fasta_extractor = FastaStringExtractor(fasta)
        else:
            raise ValueError('Fasta file does not exist.')
    elif type(fasta) is FastaStringExtractor:
        fasta_extractor = fasta
    
    # Extract reference sequence
    start_coord = insert_coord-mid_point-window_padding
    end_coord = insert_coord+mid_point+window_padding
    reference_sequence = fasta_extractor.extract(chromosome, 
                                                 start_coord, 
                                                 end_coord)

    return reference_sequence

# Insert edits into reference sequences
def edit_refseq(insert_seq, ref_seq, insert_offset=0, len_overwrite=None):

    '''
    Insert reference sequence at midpoint of sequence.

    ARGS
        insert_seq: str with alphabet {'A','T,'C','G'}
            Use "" for no insert sequence.
        ref_seq: int
            Reference sequence to be edited. 
        len_overwrite: int (default: len(insert_seq))
            Number of bases to overwrite with insert_seq.    
        model_window: int (default: None)
            Input sequence length of the predictor model.
        insert_offset: int (default: 0)
            Offcenter insert site from center.

    RETURNS
        Edited DNA sequence string.
    
    '''

    # Make edited sequence:
    # Insert sequence at mid_point of reference sequence
    if len_overwrite is None:
        len_overwrite=len(insert_seq)

    if insert_offset > int(len(ref_seq)/2) or insert_offset <= -int(len(ref_seq)/2):
        raise ValueError('insert_offset places insert site outside of sequence.')
    elif abs(insert_offset) > int(len(ref_seq)/2)/3:
        logging.warning('insert_offset places insert site close to sequence edge.')

    edited_sequence = ref_seq[:int(len(ref_seq)/2)+insert_offset] + \
                      insert_seq + \
                      ref_seq[int(len(ref_seq)/2)+len_overwrite+insert_offset:]

    return edited_sequence

# Function to score a pair of edited/reference sequences
def score_seq(predict_func, edited_seq, ref_seq, **kwargs):

    '''
    Compute edit effect using a predictive model.

    ARGS
        predict_func: function
            Prediction function that predicts effect for a sequence string.
        edited_seq: str with alphabet {'A','T,'C','G'}
            DNA sequence with in-silico edit.
        ref_seq: str with alphabet {'A','T,'C','G'}
            DNA sequence from reference before edit.
        **kwargs:
            Model specific input for predict_func.

    RETURNS
        Ratio of effect change w.r.t. reference.
    
    '''

    # One hot encode sequence
    edited_sequence_onehot = one_hot_encode(edited_seq)
    reference_sequence_onehot = one_hot_encode(ref_seq)

    # Predict effect
    edited_sequence_effect = predict_func(edited_sequence_onehot, **kwargs)
    reference_sequence_effect = predict_func(reference_sequence_onehot, **kwargs)
    
    # Calculate change
    effect_change = edited_sequence_effect/(reference_sequence_effect+1e-12)

    return effect_change