import torch
import numpy as np
from .utils import check_bpnet_sequence

import sys # Place PROCapNet repo in PATH
sys.path.append('procapnet/ProCapNet/src/2_train_models/')
sys.path.append('procapnet/ProCapNet/src/4_interpret_models/')
sys.path.append('procapnet/ProCapNet/src/utils/')


def load_trained_model(model_path, cuda=False):
    
    '''
    Load pretrained ProCapNet model.
    
    '''

    if cuda:
        model = torch.load(model_path).cuda()
    else:
        model = torch.load(model_path)

    return model

def predict_transcription(sequence_onehot, models=None, strand=0, cuda=False):

    '''
    Predict output.

    '''
    sequence_onehot = check_bpnet_sequence(sequence_onehot)
    sequence_onehot = sequence_onehot.transpose(0,-1,1)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    predictions=[]
    for model in models:

        with torch.no_grad():
            sequence_onehot = torch.tensor(sequence_onehot, dtype=torch.float32)
            if cuda:
                sequence_onehot = sequence_onehot.cuda()
            pred_profiles, pred_logcounts = model.predict(sequence_onehot)
            rc_pred_profiles, rc_pred_logcounts = model.predict(torch.flip(sequence_onehot, [-1, -2]))

        if strand>0:        
            prediction = np.exp(pred_logcounts)
        elif strand<0:
            prediction = np.exp(rc_pred_logcounts)
        else:
            prediction = np.exp(pred_logcounts) + np.exp(rc_pred_logcounts)
        prediction = prediction/2
        predictions.append(prediction)
    
    prediction = np.mean(predictions)

    return prediction

def compute_attribution(sequence_onehot, models=None, mode='scoring', is_stranded=True, cuda=False):

    '''
    Compute DeepSHAP attribution.
    
    '''
 
    sequence_onehot = check_sequence(sequence_onehot)
    sequence_onehot = sequence_onehot.transpose(0,-1,1)

    if not isinstance(models, (list, tuple, np.ndarray)):
        models = [models]

    profile_scores_dicts, count_scores_dicts = [], []
    for model in models:

        assert len(sequence_onehot.shape) == 3 and sequence_onehot.shape[1] == 4, sequence_onehot.shape
        prof_attrs = []
        count_attrs = []

        with torch.no_grad():
            for i in trange(len(sequence_onehot)):
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

    if mode=='scoring': 
        raise NotImplemnetedError()
    else: 
        return count_scores_dicts

