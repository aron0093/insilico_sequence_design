import json
from baskerville import seqnn

import sys # Adjust this path
sys.path.append('../../../../../borzoi/borzoi/')

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
    if rc :
        strand_pair = targets_df.strand_pair
        target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        ], dtype='int32')

    model = seqnn.SeqNN(params_model)
    model.restore(model_path, 0)
    model.build_slice(target_index)
    if rc :
        model.strand_pair.append(slice_pair)
    model.build_ensemble(rc, [0])

    return model

def predict_RNA_expression(sequence_onehot, models=None, gene_slice_idx=None, rc=False, sample_idx=None, mode='scoring'):

    '''
    Predict RNA expression output.

    '''

    # Check formatting
    sequence_onehot = check_borzoi_sequence(sequence_onehot)

    # This only works on a GPU
    # Make sure sequence is formatted correctly with the mask and strandedness accoutned for

    # Make predictions
    prediction = np.concatenate([models[rep_idx](sequence_onehot)[:, None, ...].astype("float32") for rep_idx in np.arange(len(models))], axis=1)

    if sample_idx is not None:
        prediction = prediction[..., sample_idx]

    prediction = undo_transform(prediction, clip_soft=384., track_transform=3./4., track_scale=0.3).mean(1)

    if gene_slice_idx is not None:
        prediction = prediction[:, gene_slice_idx]

    # Average over sample heads and return scalar
    if mode=='scoring':
        prediction = np.mean(prediction,-1).sum(-1)

    # Single-sample #TODO: Implement batch processing
    prediction = prediction[0]

    return prediction

def compute_attribution(sequence_onehot, models=None, cuda=False):

    '''
    Computed Input X Gradient attributions.
    
    '''
    #https://github.com/calico/borzoi/blob/main/examples/borzoi_helpers.py
    # lines 370, 464

    raise NotImplementedError()