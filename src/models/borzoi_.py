import json
from baskerville import seqnn

import numpy as np
import pandas as pd

import sys # Adjust this path
sys.path.append('../../../../../borzoi/borzoi/examples/')
#from borzoi_helpers import _prediction_input_grad
#from borzoi_helpers import get_prediction_gradient

def check_borzoi_sequence(sequence_onehot):

    try:
        sequence_onehot = np.array(sequence_onehot)
        assert sequence_onehot.shape[-1]==4
        assert sequence_onehot.shape[-2]==524288
    except:
        if sequence_onehot.shape[-2]==4:
            sequence_onehot = np.swapaxes(sequence_onehot, -1,-2)
            print('Last two axes were swapped to obtain (n_batches, n_bases, seq_len)')

        if sequence_onehot.shape[-2]>524288:
            midpoint = int(sequence_onehot.shape[-2]/2)
            sequence_onehot = np.take(sequence_onehot, np.arange(midpoint-int(524288/2), midpoint+int(524288/2)), -2)
            print('Sequence was subset to center 524288 bases')
        elif sequence_onehot.shape[-2]==524288:
            pass
        else:
            raise ValueError('Input must be formatted as (n_batches, n_bases, seq_len)')
    
    if len(sequence_onehot.shape)==2:
        sequence_onehot = np.expand_dims(sequence_onehot, 0)
    
    return sequence_onehot

def undo_transform(y, clip_soft=384., track_transform=3./4., track_scale=0.3):

    # Undo scale
    y /= track_scale

    if clip_soft is not None:
        y_unclipped = (y - clip_soft)**2 + clip_soft
        unclip_mask_wt = (y > clip_soft)

        y[unclip_mask_wt] = y_unclipped[unclip_mask_wt]

    # Undo sqrt
    y = y ** (1. / track_transform)

    return y

def load_trained_model(model_path, params_file, targets_file, rc=False):
    
    '''
    Load pretrained Borzoi model.
    
    '''
    #Read model parameters
    with open(params_file) as params_open :
        params = json.load(params_open)
        params_model = params['model']

    #Read targets
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    target_index = targets_df.index

    #Create local index of strand_pair (relative to sliced targets)
    if rc:
        strand_pair = targets_df.strand_pair
        target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        ], dtype='int32')

    model = seqnn.SeqNN(params_model)
    model.restore(model_path, 0)
    model.build_slice(target_index)
    if rc:
        model.strand_pair.append(slice_pair)
    model.build_ensemble(rc, [0])

    return model

def predict_func(sequence_onehot, models=None, bin_slice_idx=None, sample_idx=None, 
                 clip_soft=384., track_transform=3./4., track_scale=0.3, mode='scoring'):

    # Check formatting
    sequence_onehot = check_borzoi_sequence(sequence_onehot)

    # Make predictions
    prediction = np.concatenate([models[rep_idx](sequence_onehot)[:, None, ...].astype("float32") for rep_idx in range(len(models))], axis=1)

    if sample_idx is not None:
        prediction = prediction[..., sample_idx]

    prediction = undo_transform(prediction, clip_soft=clip_soft, track_transform=track_transform, track_scale=track_scale).mean(1)

    if bin_slice_idx is not None:
        for idx_ in bin_slice_idx:
            if idx_<0 or idx_>16383:
                raise ValueError('bin slice idx is negative or > 16383')
        prediction = prediction[:, bin_slice_idx]

    # Average over sample heads and return batch predictions
    if mode == 'scoring':
        # To return a cumulative score or keep batch
        prediction = np.mean(prediction, axis=-1).sum(axis=-1)

        # Convert to scalar if single sequence
        if prediction.shape[0]==1:
            prediction=prediction[0]

    return prediction

def predict_RNA_expression(sequence_onehot, models=None, bin_slice_idx=None, 
                           sample_idx=None, mode='scoring'):

    '''
    Predict RNA expression output.

    '''
    prediction = predict_func(sequence_onehot, models, bin_slice_idx, sample_idx, mode=mode,
                              clip_soft=384., track_transform=3./4., track_scale=0.3)

    return prediction

def predict_CAGE_expression(sequence_onehot, models=None, bin_slice_idx=None, 
                            sample_idx=None, mode='scoring'):

    '''
    Predict CAGE expression output.

    '''

    prediction = predict_func(sequence_onehot, models, bin_slice_idx, sample_idx, mode=mode,
                              clip_soft=384., track_transform=3./4., track_scale=1)

    return prediction

def predict_DNASE_accessibility(sequence_onehot, models=None, bin_slice_idx=None, 
                                sample_idx=None, mode='scoring'):

    '''
    Predict DNASE output.

    '''
    prediction = predict_func(sequence_onehot, models, bin_slice_idx, sample_idx, mode=mode,
                              clip_soft=32., track_transform=3./4., track_scale=2.)

    return prediction

def compute_attribution(sequence_onehot, models=None, bin_slice_idx=None, sample_idx=None, cuda=False):

    '''
    Computed Input X Gradient attributions.
    
    '''
    #https://github.com/calico/borzoi/blob/main/examples/borzoi_helpers.py
    # lines 370, 464

    raise NotImplementedError()
