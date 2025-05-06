import os
import argparse

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from _utils import FastaStringExtractor, reverse_complement
from _sequence import extract_refseq
from _annealing import run_simulated_annealing

from plotting import plot_fitness, plot_temp_scaling
from matplotlib import pyplot as plt

# Function for running simulated annealing to design a number of sequence edits based on model predictions
def main(*model_paths,
         fasta_file,
         chromosome,
         insert_coord,
         model_type='chrombpnet',
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
         output_path=None,
         take_reverse_complement=False,
         add_mask=None,
         sample_idx=None):

    # Setup model specific params and prediction func
    add_args = {}
    if model_type=='chrombpnet':

        from models.chrombpnet_ import load_trained_model as load_chrombpnet_model
        from models.chrombpnet_ import predict_accessibility

        models = [load_chrombpnet_model(model_path) for model_path in model_paths]
        predict_func = predict_accessibility
        model_window = 2114

    elif model_type=='procapnet':

        from models.procapnet_ import load_trained_model as load_procapnet_model
        from models.procapnet_ import predict_transcription

        models = [load_procapnet_model(model_path) for model_path in model_paths]        
        predict_func = predict_transcription
        model_window = 2114

    elif model_type=='decima':

        from models.decima_ import load_trained_model as load_decima_model
        from models.decima_ import predict_expression

        models = [load_decima_model(model_path) for model_path in model_paths]
        predict_func = predict_expression
        model_window = 524288
        add_args = {'add_mask':add_mask, 'sample_idx':sample_idx}
            
    # Extract reference sequence
    window_padding=100
    fse = FastaStringExtractor(fasta_file)
    ref_seq = extract_refseq(chromosome, insert_coord, 
                             fasta=fse, model_window=model_window,
                             window_padding=window_padding)
    if take_reverse_complement:
        ref_seq = reverse_complement(ref_seq)


    # Make randomn insert sequence if none is provided
    if insert_sequence is None:
        insert_sequence = ''.join([np.random.choice(['A', 'C', 'G', 'T']) for j in \
                                   range(np.random.randint(5, max_edited_bp+1))])

    insert_offset_midrange = int((insert_offset_range[0] + insert_offset_range[1])/2)
    init_bundle = [ref_seq, insert_sequence, insert_offset_midrange, 0]

    # Setup annealing
    temperature_ = [temperature_range[0]]
    edit_history_ = [init_bundle]
    fitness_ = [1.]
    score_ = [1.]

    # Initialise annealing
    annealing = run_simulated_annealing(predict_func,
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
                          models=models, **add_args)

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
    parser.add_argument('--model_type', choices=['chrombpnet', 'procapnet', 'decima'], default='chrombpnet')
    parser.add_argument('--fasta_file', type=str)
    parser.add_argument('--chromosome', type=str)    
    parser.add_argument('--insert_coord', type=int)
    parser.add_argument('--insert_sequence', default=None, type=str)
    parser.add_argument('--insert_offset_range', nargs='+', type=int)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('-o','--output_path', default=None, type=str)

    # Decima specific aeguments
    parser.add_argument('--reverse_complement', action='store_true')
    parser.add_argument('--add_mask', nargs='+', type=int)  
    parser.add_argument('--sample_idx', nargs='+', type=int)  

    args = parser.parse_args()

    # Make sure output path exists
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    edit_record = main(*args.model_paths, 
                       model_type=args.model_type, 
                       fasta_file=args.fasta_file, 
                       chromosome=args.chromosome, 
                       insert_coord=args.insert_coord, 
                       insert_sequence=args.insert_sequence, 
                       insert_offset_range=args.insert_offset_range,
                       n_iters=args.num_iters, output_path=args.output_path,
                       take_reverse_complement=args.reverse_complement,
                       add_mask=args.add_mask,
                       sample_idx=args.sample_idx)

    # Save output to file
    if args.output_path is not None:
        edit_record.to_csv(args.output_path, sep='\t')

        # Save plots
        fig, axs = plt.subplots(ncols=2, figsize=(15,7))
        plot_fitness(edit_record, ax=axs[0])
        plot_temp_scaling(edit_record, ax=axs[1])

        plt.savefig(args.output_path.split('.')[0]+'.png')




