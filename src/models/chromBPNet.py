import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

import chrombpnet.training.utils.losses as losses

import shap
import tensorflow as tf
# disable eager execution so shap deep explainer wont break
tf.compat.v1.disable_eager_execution()

import chrombpnet.evaluation.interpret.shap_utils as shap_utils
import chrombpnet.evaluation.interpret.input_utils as input_utils
from chrombpnet.evaluation.interpret.interpret import generate_shap_dict

from .utils import check_bpnet_sequence

def load_trained_model(model_path):
    
    '''
    Load pretrained ChromBPNet model.
    
    '''

    custom_objects={'tf': tf, 
                    'multinomial_nll': losses.multinomial_nll} 
    get_custom_objects().update(custom_objects)  
    
    model=load_model(model_path, compile=False)

    return model

def predict_accessibility(sequence_onehot, models=None, mode='count'):

    '''
    Predict mean profile head output.

    '''
    sequence_onehot = check_bpnet_sequence(sequence_onehot)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    predictions=[]
    for model in models:

        if mode=='profile':
            prediction = model.predict_on_batch(sequence_onehot)[0]
            prediction = prediction.mean(axis=-1)
        elif mode=='count':
            prediction = model.predict_on_batch(sequence_onehot)[1]
            prediction=prediction.flatten()[0]
        predictions.append(np.exp(prediction))
    
    prediction = np.mean(predictions)

    return prediction

def compute_attribution(sequence_onehot, models=None, mode='scoring'):

    '''
    Compute DeepSHAP attribution.
    
    '''
    
    sequence_onehot = check_bpnet_sequence(sequence_onehot)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    profile_scores_dicts = []
    for model in models:

        outlen = model.output_shape[0][1]

        profile_model_input = model.input
        profile_input = sequence_onehot
        counts_model_input = model.input
        counts_input = sequence_onehot

        weightedsum_meannormed_logits = shap_utils.get_weightedsum_meannormed_logits(model)
        profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
                                        (profile_model_input, weightedsum_meannormed_logits),
                                        shap_utils.shuffle_several_times,
                                        combine_mult_and_diffref=shap_utils.combine_mult_and_diffref)

        profile_shap_scores = profile_model_profile_explainer.shap_values(profile_input, progress_message=100)
        profile_scores_dict = generate_shap_dict(sequence_onehot, profile_shap_scores)

        profile_scores_dicts.append(profile_scores_dict)

    if mode=='scoring': 
        return np.mean([profile_scores_dict['projected_shap']['seq'].sum(1).mean(-1) for profile_scores_dict in profile_scores_dicts])
    else: 
        return [profile_scores_dict['projected_shap']['seq'] for profile_scores_dict in profile_scores_dicts]