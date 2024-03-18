import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

import chrombpnet.training.utils.losses as losses

def load_trained_model(model_path):
    
    '''
    Load pretrained ChromBPNet model.
    
    '''

    custom_objects={'tf': tf, 
                    'multinomial_nll': losses.multinomial_nll} 
    get_custom_objects().update(custom_objects)  
    
    model=load_model(model_path, compile=False)

    return model

def predict_accessibility(sequence_onehot, model):

    '''
    Predict mean profile head output.

    '''
    try:
        sequence_onehot = np.array(sequence_onehot)
        if sequence_onehot.ndim==2:
            sequence_onehot = np.expand_dims(sequence_onehot,0)
    except:
        raise ValueError('Input must be formatted as (n_batches, n_bases, 4)')

    try:
        assert sequence_onehot.ndim==3
        if sequence_onehot.shape[1]!=2114:
            midpoint = int(sequence_onehot.shape[1]/2)
            sequence_onehot = sequence_onehot[:,midpoint-int(2114/2):midpoint+int(2114/2)]
        assert sequence_onehot.shape[1]==2114
        assert sequence_onehot.shape[2]==4
    except:
        raise ValueError('One hot DNA sequence is not properly formatted.')

    prediction = model.predict_on_batch(sequence_onehot)[0]
    prediction = prediction.flatten().mean()

    return prediction