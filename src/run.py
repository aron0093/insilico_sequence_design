import os
import argparse

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from _utils import FastaStringExtractor
from _sequence import extract_refseq
from _annealing import run_simulated_annealing

from plotting import plot_fitness, plot_temp_scaling
from matplotlib import pyplot as plt

from models.chromBPNet import load_trained_model, predict_accessibility

# Function for running simulated annealing to design a number of sequence edits based on model predictions
def main(*model_paths,
         fasta_file,
         chromosome,
         insert_coord,
         insert_sequence=None,
         objective='max',
         clip_prob=0.9,
         n_iters=1000,
         temperature_range=(0.001,0.0001),
         max_edited_bp=10, 
         max_overwritten_wt_bp=3, 
         insert_offset_range=(-2, 2),
         action_probs={'SUB' : 0.7, 
                       'INS' : 0.15, 
                       'DEL' : 0.05, 
                       'SWITCH_POS' : 0.05, 
                       'INS_WT' : 0.025, 
                       'DEL_WT' : 0.025},
         output_path=None):


    # Extract reference sequence
    window_padding=100
    fse = FastaStringExtractor(fasta_file)
    ref_seq = extract_refseq(chromosome, insert_coord, 
                             fasta=fse, model_window=2114,
                             window_padding=window_padding)

    # Make randomn insert sequence
    if insert_sequence is None:
        insert_sequence = ''.join([np.random.choice(['A', 'C', 'G', 'T']) for j in \
                                   range(np.random.randint(5, max_edited_bp+1))])

    init_bundle = [ref_seq, insert_sequence, 0, 0]

    # Setup annealing
    temperature_ = [temperature_range[0]]
    edit_history_ = [init_bundle]
    fitness_ = [1.]
    score_ = [1.]

    models = [load_trained_model(model_path) for model_path in model_paths]

    # Initialise annealing
    annealing = run_simulated_annealing(predict_accessibility,
                                        init_bundle, action_probs,
                                        init_fitness=fitness_[0])

    # Run annealing
    for i in tqdm(range(n_iters), desc='Mutating sequence'):

        annealing.temp_annealing(i, n_iters=n_iters,
                                 temperature_range=temperature_range,
                                 exp_scale=1./0.7)
        annealing.iterate(clip_prob=clip_prob, 
                          objective=objective, 
                          max_edited_bp=max_edited_bp, 
                          max_overwritten_wt_bp=max_overwritten_wt_bp, 
                          insert_offset_range=insert_offset_range, 
                          models=models)

        # Save steps
        temperature_.append(annealing.temperature)
        edit_history_.append(annealing.edit_history)
        fitness_.append(annealing.fitness)
        score_.append(annealing.score)

        edit_record = pd.DataFrame(edit_history_, columns=['reference_sequence', 
                                                           'insert_sequence', 
                                                           'insert_offset', 
                                                           'overwritten_wildtype_basepairs'])
        edit_record.index.name = 'iteration'

        edit_record['reference_sequence'] = edit_record['reference_sequence'].apply(lambda x: x[window_padding: -window_padding])
        edit_record['fitness'] = fitness_
        edit_record['score'] = score_
        edit_record['temperature'] = temperature_
        edit_record['objective'] = objective

        # Save output at each iteration
        if output_path is not None:
            # Save output to file
            edit_record.to_csv(output_path, sep='\t')

    return edit_record

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_paths', nargs='+', type=str)
    parser.add_argument('--fasta_file', type=str)
    parser.add_argument('--chromosome', type=str)
    parser.add_argument('--insert_coord', type=int)
    parser.add_argument('--insert_sequence', default=None, type=str)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('-o','--output_path', default=None, type=str)

    args = parser.parse_args()

    # Make sure output path exists
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    edit_record = main(*args.model_paths, fasta_file=args.fasta_file, chromosome=args.chromosome, 
                       insert_coord=args.insert_coord, insert_sequence=args.insert_sequence, n_iters=args.num_iters,
                       output_path=args.output_path)

    # Save output to file
    if args.output_path is not None:
        edit_record.to_csv(args.output_path, sep='\t')

        # Save plots
        fig, axs = plt.subplots(ncols=2, figsize=(15,7))
        plot_fitness(edit_record, ax=axs[0])
        plot_temp_scaling(edit_record, ax=axs[1])

        plt.savefig(args.output_path.split('.')[0]+'.png')




