import torch
import numpy as np

import sys # Adjust PATH or install 
sys.path.append('../../../decima/src/decima')

from lightning import LightningModel

def check_decima_sequence(decima_sequence_onehot, add_mask=None):

    n_bases=5
    if add_mask is not None:
        n_bases=4

    try:
        decima_sequence_onehot = np.array(decima_sequence_onehot)
        assert decima_sequence_onehot.shape[-2]==n_bases
        assert decima_sequence_onehot.shape[-1]==524288
    except:
        if decima_sequence_onehot.shape[-1]==n_bases:
            decima_sequence_onehot = np.swapaxes(decima_sequence_onehot, -1,-2)
            print('Last two axes were swapped to obtain (n_batches, n_bases, seq_len)')

        if decima_sequence_onehot.shape[-1]>524288:
            midpoint = int(decima_sequence_onehot.shape[-1]/2)
            decima_sequence_onehot = np.take(decima_sequence_onehot, np.arange(midpoint-int(524288/2), midpoint+int(524288/2)), -1)
            print('Sequence was subset to center 524288 bases')
        elif decima_sequence_onehot.shape[-1]==524288:
            pass
        else:
            raise ValueError('Input must be formatted as (n_batches, n_bases, seq_len)')
    
    if len(decima_sequence_onehot.shape)==2:
        decima_sequence_onehot = np.expand_dims(decima_sequence_onehot, 0)
    
    return decima_sequence_onehot

def load_trained_model(model_path, cuda=False):
    
    '''
    Load pretrained Decima model.
    
    '''
    return LightningModel.load_from_checkpoint(model_path).eval()

def predict_expression(decima_sequence_onehot, models=None, add_mask=None, sample_idx=None, mode='scoring'):

    '''
    Predict output.

    '''

    # Check formatting
    decima_sequence_onehot = check_decima_sequence(decima_sequence_onehot, add_mask)

    if add_mask is not None:
        if len(add_mask)!=2:
            raise ValueError('add_mask is improperly specified.')

        center_pos = int(decima_sequence_onehot.shape[-1]/2)

        mask_start = add_mask[0]
        mask_end = add_mask[1]
        mask_len = mask_start-mask_end

        mask = np.zeros(decima_sequence_onehot.shape[-1])
        mask[mask_start:mask_end] = 1
        mask = np.expand_dims(np.expand_dims(mask,0),0)
        mask = np.repeat(mask,decima_sequence_onehot.shape[0],0)

        decima_sequence_onehot = np.append(decima_sequence_onehot, 
                                           mask, 1)

    # This only works on a GPU
    # Make sure sequence is formatted correctly with the mask and strandedness accoutned for
    decima_sequence_onehot = torch.from_numpy(decima_sequence_onehot).float()
    decima_sequence_onehot = decima_sequence_onehot.cuda()

    # Make predictions
    prediction = [model(decima_sequence_onehot).cpu().detach().numpy() for model in models]
    prediction = np.mean(prediction, 0)

    if sample_idx is not None:
        prediction = prediction[:, sample_idx]
    
    prediction = np.squeeze(prediction,-1)

    # Average over sample heads
    if mode=='scoring':
        prediction = np.mean(prediction,-1)

    # Single-sample #TODO: Implement batch processing
    prediction = prediction[0]

    return prediction

def compute_attribution(sequence_onehot, models=None, strand=0, cuda=False):

    '''
    Read precomputed Input X Gradient attributions.
    
    '''
 
    #TODO: Extract per sequence (no h5) version
    #TODO: Add options to select which sample head
    # Adapt from src/decima/interpret.py

    raise NotImplementedError()




