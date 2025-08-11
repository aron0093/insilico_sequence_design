import os
import argparse

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from _utils import FastaStringExtractor, reverse_complement, saturation_mutagenesis, one_hot_encode
from _sequence import extract_refseq, edit_refseq

# Function for running simulated annealing to design a number of sequence edits based on model predictions
def main(*model_paths,
         fasta_file,
         chromosome,
         insert_coord,
         start_pos=None,
         model_type='chrombpnet',
         insert_sequence=None,
         insert_offset=0,
         overwritten_wt_bp=0, 
         output_path=None,
         strand=None,
         take_reverse_complement=False,
         add_mask=None,
         sample_idx=None,
         sample_type=None,
         bin_slice_idx=None,
         params_file=None,
         targets_file=None,
         ontology_terms=None):

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

        add_args = {'strand':strand}

    elif model_type=='decima':

        from models.decima_ import load_trained_model as load_decima_model
        from models.decima_ import predict_expression

        models = [load_decima_model(model_path) for model_path in model_paths]
        predict_func = predict_expression
        model_window = 524288
        add_args = {'add_mask':add_mask, 'sample_idx':sample_idx}

    elif model_type=='borzoi':

        from models.borzoi_ import load_trained_model as load_borzoi_model
        from models.borzoi_ import predict_RNA_expression, predict_CAGE_expression, predict_DNASE_accessibility

        models = [load_borzoi_model(model_path, params_file, targets_file, rc=True) \
                                    for model_path in model_paths]
        if sample_type=='RNA':
            predict_func = predict_RNA_expression
        elif sample_type=='CAGE':
            predict_func = predict_CAGE_expression
        elif sample_type=='DNASE':
            predict_func = predict_DNASE_accessibility

        model_window = 524288
        add_args = {'bin_slice_idx': bin_slice_idx, 'sample_idx':sample_idx}

    elif model_type=='alphagenome':
        from models.alphagenome_ import load_model
        from models.alphagenome_ import predict_func

        models = load_model(model_paths[0])
        model_window = 1048576
        add_args = {'strand': strand, 
                    'slice_idx':bin_slice_idx, 
                    'ontology_terms': ontology_terms, 
                    'output_type': sample_type}

    # Extract reference sequence

    fse = FastaStringExtractor(fasta_file)
    ref_seq = extract_refseq(chromosome, insert_coord, 
                             fasta=fse, model_window=model_window,
                             window_padding=0)
        
    # Make edit bundle
    edited_seq = edit_refseq(insert_sequence, ref_seq, insert_offset, overwritten_wt_bp)

    # Default start is center -50 (this is arbitrary)
    if start_pos is None:
        start_pos = model_window // 2 - 50

    # Score reversed sequences
    if take_reverse_complement:
        ref_seq = reverse_complement(ref_seq)
        edited_seq = reverse_complement(edited_seq)

        start_pos = model_window - start_pos -250 # Distance to end becomes distance from start

    edited_seq_onehot = one_hot_encode(edited_seq)

    base_pred, ism_preds = saturation_mutagenesis(predict_func, models, edited_seq_onehot, 
                                                  edit_window=250, start_coord=start_pos, 
                                                  batch_size=None, **add_args) # TODO: Decima mask adjustment if revcomp after edit
    ism_preds -= base_pred
    ism_preds = ism_preds.reshape(250,4)

    # ISM as DataFrame
    if take_reverse_complement:
        ism_preds=ism_preds[::-1]
        coord_offset = model_window //2 - start_pos - 250
        ism_preds = pd.DataFrame(ism_preds, columns=['T', 'G', 'C', 'A'], index=np.arange(insert_coord + coord_offset, insert_coord + coord_offset +250))
        ism_preds = ism_preds.loc[:,['A', 'C', 'T', 'G']]
        
    else:
        coord_offset = start_pos - model_window //2
        ism_preds = pd.DataFrame(ism_preds, columns=['A', 'C', 'T', 'G'], index=np.arange(insert_coord + coord_offset, insert_coord + coord_offset +250))

    # Save output at each iteration
    if output_path is not None:
        # Save output to file
        ism_preds.to_csv(output_path, sep='\t')
    
    return ism_preds

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_paths', nargs='+', type=str)
    parser.add_argument('--model_type', choices=['chrombpnet', 'procapnet', 'decima', 'borzoi', 'alphagenome'], default='chrombpnet')
    parser.add_argument('--fasta_file', type=str)
    parser.add_argument('--chromosome', type=str)    
    parser.add_argument('--insert_coord', type=int)
    parser.add_argument('--start_pos', type=int, default=None)
    parser.add_argument('--insert_sequence', default='', type=str)
    parser.add_argument('--insert_offset', default=0, type=int)
    parser.add_argument('--overwritten_wt_bp', default=0, type=int)
    parser.add_argument('-o','--output_path', default=None, type=str)

    # ProCAPNet specific arguments
    parser.add_argument('--strand', choices=[0, 1, -1], default=0, type=int)    

    # Decima specific aeguments
    parser.add_argument('--reverse_complement', action='store_true')
    parser.add_argument('--add_mask', nargs='+', type=int)  
    parser.add_argument('--sample_idx', nargs='+', type=int)  # Borzoi too

    # Borzoi specific arguments
    parser.add_argument('--sample_type', choices=['RNA', 'CAGE', 'DNASE'], default='RNA')  # AG too
    parser.add_argument('--bin_slice_idx', nargs='+', type=int)    # AG too
    parser.add_argument('--params_file', default=None, type=str)
    parser.add_argument('--targets_file', default=None, type=str)

    # Alphagenome specific arguments
    parser.add_argument('--ontology_terms', nargs='+', type=str)

    args = parser.parse_args()

    # Make sure output path exists
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    ism_preds = main(*args.model_paths, 
                       model_type=args.model_type, 
                       fasta_file=args.fasta_file, 
                       chromosome=args.chromosome, 
                       insert_coord=args.insert_coord, 
                       start_pos=args.start_pos,
                       insert_sequence=args.insert_sequence, 
                       insert_offset=args.insert_offset,
                       overwritten_wt_bp=args.overwritten_wt_bp,
                       output_path=args.output_path,
                       strand=args.strand,
                       take_reverse_complement=args.reverse_complement,
                       add_mask=args.add_mask,
                       sample_idx=args.sample_idx,
                       sample_type=args.sample_type,
                       bin_slice_idx=args.bin_slice_idx,
                       params_file=args.params_file,
                       targets_file=args.targets_file,
                       ontology_terms=args.ontology_terms)

    # Save output to file
    if args.output_path is not None:
        ism_preds.to_csv(args.output_path, sep='\t')






