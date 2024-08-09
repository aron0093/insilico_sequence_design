import numpy as np

def check_bpnet_sequence(sequence_onehot):

    try:
        sequence_onehot = np.array(sequence_onehot)
        if sequence_onehot.ndim==2:
            sequence_onehot = np.expand_dims(sequence_onehot,0)
    except:
        raise ValueError('Input must be formatted as (n_batches, n_bases, 4)')

    try:
        assert sequence_onehot.ndim==3
        if sequence_onehot.shape[1]>2114:
            midpoint = int(sequence_onehot.shape[1]/2)
            sequence_onehot = sequence_onehot[:,midpoint-int(2114/2):midpoint+int(2114/2)]
        assert sequence_onehot.shape[1]==2114
        assert sequence_onehot.shape[2]==4
    except:
        raise ValueError('One hot DNA sequence is not properly formatted.')
    
    return sequence_onehot