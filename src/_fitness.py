import numpy as np
from _mutagenesis import make_edit
from _sequence import score_seq

import logging

# Function to evaluate fitness of a sequence given
# predictions from a trained sequence model
def evaluate_fitness(edit_bundle, predict_func, objective='max', **kwargs):

    '''
    Evaluate the fitness of a sequence w.r.t. reference

    ARGS
        edit_bundle: list
            [ref_seq, insert_seq, insert_offset, overwritten_wt_bp]
            ref_seq: 
                Reference sequence to be edited. 
            insert_seq:
                Use "" for no insert sequence.
            insert_offset: int
                insert seq offset from insert site.
            overwritten_wt_bp: int
                Number of WT bases to overwrite (delete).
        predict_func: func
            Scoring function to predict activity of a sequence.
        objective: {'max', 'min', float}
            Fitness objective w.r.t. scoring function.
        **kwargs:
            Model (predict_func) specific parameters. 
    RETURNS
        fitness, score
    
    '''

    ref_seq, insert_seq, insert_offset, overwritten_wt_bp = edit_bundle
    edited_seq = make_edit(edit_bundle)

    score = score_seq(predict_func, edited_seq, ref_seq, **kwargs)

    fitness = 0.

    if isinstance(objective, int) or isinstance(objective, float):
        fitness = target_score / np.abs(score - target_score)
    elif objective == 'max' :
        fitness = score
    elif objective == 'min' :
        fitness = -score

    return fitness, score

# Function to accept/reject edit (MCMC criteria)
def evaluate_acceptance(fitness_curr, fitness_prev, temperature=1., clip_prob=0.9):

    '''
    MCMC acceptance criteria for sequence edits.

    '''

    accept_prob = np.exp(np.minimum((fitness_curr - fitness_prev) / (temperature + 1e-12), np.log(clip_prob)))
    accept_rand = np.random.rand()

    accepts = (fitness_curr > fitness_prev) | (accept_rand >= (1. - accept_prob))

    return accepts

