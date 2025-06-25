import torch
import numpy as np
from .utils import check_bpnet_sequence

import sys # Adjust PATH or install
sys.path.append('../../../procapnet/ProCapNet/src/2_train_models/')
sys.path.append('../../../procapnet/ProCapNet/src/4_interpret_models/')
sys.path.append('../../../procapnet/ProCapNet/src/utils/')

from BPNet_strand_merged_umap import Model
from deepshap_utils import *

def load_trained_model(model_path, cuda=False):
    
    '''
    Load pretrained ProCapNet model.
    
    '''

    if cuda:
        model = torch.load_from(model_path).cuda()
    else:
        model = torch.load(model_path)

    return model

def process_predictions(pred_profiles_logits, pred_logcounts):

    pred_profiles = np.exp(pred_profiles_logits)
    pred_profiles /= pred_profiles.sum()

    pred_counts = np.exp(pred_logcounts)

    return pred_profiles, pred_counts

def compute_strand_counts(pred_profiles, pred_counts, strand=0):
    
    # Distribute counts over strands
    pred_profiles_counts = pred_profiles*pred_counts
    positive_counts = pred_profiles_counts[:,0].sum()
    negative_counts = pred_profiles_counts[:,1].sum()

    # Return stranded counts
    if strand==0:
        pred_counts = positive_counts + negative_counts
    elif strand==-1:
        pred_counts = negative_counts
    elif strand==1:
        pred_counts = positive_counts

    return pred_counts

#TODO: Modify to work with batches
def predict_transcription(sequence_onehot, models=None, strand=0, rc=True, cuda=False):

    '''
    Predict output.

    '''
    sequence_onehot = check_bpnet_sequence(sequence_onehot)
    sequence_onehot = sequence_onehot.transpose(0,-1,1)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    predictions=[]
    for model in models:

        # Predict counts and profiles
        with torch.no_grad():
            sequence_onehot = torch.tensor(sequence_onehot, dtype=torch.float32)
            if cuda:
                sequence_onehot = sequence_onehot.cuda()
            pred_profiles_logits, pred_logcounts = model.predict(sequence_onehot)
            rc_pred_profiles_logits, rc_pred_logcounts = model.predict(torch.flip(sequence_onehot, [-1, -2]))

        # Process predictions
        pred_profiles, pred_counts = process_predictions(pred_profiles_logits, pred_logcounts)
        rc_pred_profiles, rc_pred_counts = process_predictions(rc_pred_profiles_logits, rc_pred_logcounts)

        # Compute stranded counts
        pred_counts = compute_strand_counts(pred_profiles, pred_counts, strand)
        rc_pred_counts = compute_strand_counts(rc_pred_profiles, rc_pred_counts, strand*-1)

        # Return predictions
        if not rc:      
            prediction = pred_counts
        elif rc:
            prediction = pred_counts + rc_pred_counts
            prediction = prediction/2

        predictions.append(prediction)
    
    prediction = np.mean(predictions)

    return prediction
    
#TODO: Modify to work with batches
def compute_attribution(sequence_onehot, models=None, num_shufs=25, 
                        mode='scoring', typ='counts', 
                        is_stranded=True, cuda=False):

    '''
    Compute DeepSHAP attribution.
    
    '''
 
    sequence_onehot = check_bpnet_sequence(sequence_onehot)
    sequence_onehot = sequence_onehot.transpose(0,-1,1)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    profile_scores_dicts, count_scores_dicts = [], []
    for model in models:

        assert len(sequence_onehot.shape) == 3 and sequence_onehot.shape[1] == 4, sequence_onehot.shape
        prof_attrs = []
        count_attrs = []

        with torch.no_grad():
            for i in range(len(sequence_onehot)):
                if is_stranded:
                    prof_explainer = DeepLiftShap(StrandedProfileModelWrapper(model))
                    count_explainer = DeepLiftShap(StrandedCountsModelWrapper(model))
                else:
                    prof_explainer = DeepLiftShap(ProfileModelWrapper(model))
                    count_explainer = DeepLiftShap(CountsModelWrapper(model))
            
                # use a batch of 1 so that reference is generated for each seq 
                seq = torch.tensor(sequence_onehot[i : i + 1]).float()

                # create a reference of dinucleotide shuffled sequences
                ref_seqs = dinuc_shuffle(seq[0], num_shufs).float()

                if cuda:
                    seq = seq.cuda()
                    ref_seqs = ref_seqs.cuda()
                # calculate attributions according to profile task (fwd and rev strands)
                prof_attrs_fwd = prof_explainer.attribute(seq, ref_seqs).cpu()
                prof_attrs_rev = prof_explainer.attribute(torch.flip(seq, [1,2]),
                                                          torch.flip(ref_seqs, [1,2])).cpu()

                prof_attrs_rev = torch.flip(prof_attrs_rev, [1,2])
                
                prof_attrs_batch = np.array([prof_attrs_fwd.numpy(), prof_attrs_rev.numpy()])
                prof_attrs.append(prof_attrs_batch.mean(axis=0))

                # calculate attributions according to counts task (fwd and rev strands)
                
                count_attrs_fwd = count_explainer.attribute(seq, ref_seqs).cpu()
                count_attrs_rev = count_explainer.attribute(torch.flip(seq, [1,2]),
                                                            torch.flip(ref_seqs, [1,2])).cpu()
                
                count_attrs_rev = torch.flip(count_attrs_rev, [1,2])
                
                count_attrs_batch = np.array([count_attrs_fwd.numpy(), count_attrs_rev.numpy()])
                count_attrs.append(count_attrs_batch.mean(axis=0))

        prof_attrs = np.concatenate(prof_attrs)
        count_attrs = np.concatenate(count_attrs)

        profile_scores_dicts.append(prof_attrs)
        count_scores_dicts.append(count_attrs)

    if typ=='counts':
        return_dicts = count_scores_dicts
    elif typ=='profile':
        return_dicts = profile_scores_dicts
    elif typ=='complete':
        return_dicts = (count_scores_dicts, profile_scores_dicts)

    if mode=='scoring': 
        raise NotImplemnetedError()
    else: 
        return return_dicts

