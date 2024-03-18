import numpy as np
from _sequence import extract_refseq, edit_refseq

# Time varying control process over making edits
# editing_action(t) = f(action_probabilties, editing_action(t-1)..., editing_action(0))
# edited_sequence(t) = f(editing_action(t), edited_sequence(t-1))

# Probability distribution over actions as func. of edit history
def control_edit(edit_history, 
                 action_probs={'SUB' : 0.7, 
                               'INS' : 0.15, 
                               'DEL' : 0.05, 
                               'SWITCH_POS' : 0.05, 
                               'INS_WT' : 0.025, 
                               'DEL_WT' : 0.025},
                 max_edited_bp=10, max_overwritten_wt_bp=3, 
                 insert_offset_range=(-2, 2)):

    '''
    Rescale edit action sampling probabilities based on history.

    ARGS
        edit_history: list
            [ref_seq, insert_seq, insert_offset, overwritten_wt_bp]
            ref_seq: 
                Reference sequence that was edited. 
            insert_seq:
                Use "" for no insert sequence.
            insert_offset: int
                insert seq offset from insert site.
            overwritten_wt_bp: int
                Number of WT bases that have been overwritten (deleted).
        action_probs: dict
            Dictionary of sampling probability per edit-action.
        max_edited_bp: int
            Maximum number of bp edits allowed.
        max_overwritten_wt_bp: int
            Maximum number of wild-type bp to overwrite (delete).
        insert_offset_range: tuple(int, int)
            Range of insert_seq offsets w.r.t. insert_site.
    RETURNS
        Edit-action probabilities.
    
    '''

    ref_seq, insert_seq, insert_offset, overwritten_wt_bp = edit_history
    
    # Enable/disable actions based on edit history
    allowed_actions = {'SUB' : 0, 
                       'INS' : 0, 
                       'DEL' : 0, 
                       'SWITCH_POS' : 0, 
                       'INS_WT' : 0, 
                       'DEL_WT' : 0}
    
    if insert_offset_range is not None:
        allowed_actions['SWITCH_POS'] = 1

    if overwritten_wt_bp + len(insert_seq) < max_edited_bp :
        if overwritten_wt_bp < max_overwritten_wt_bp :
            allowed_actions['DEL_WT'] = 1
        allowed_actions['INS'] = 1

    if overwritten_wt_bp > 0 :
        allowed_actions['INS_WT'] = 1

    if len(insert_seq) > 0 :
        allowed_actions['DEL'] = 1
        allowed_actions['SUB'] = 1
    
    # Recompute edit-action sampling probabilities
    for key in allowed_actions.keys():
        action_probs[key] *= allowed_actions[key]
    
    scaler = sum(list(action_probs.values()))
    for key in action_probs.keys():
        action_probs[key] /= scaler
             
    return action_probs

# Function for sampling an edit bundle
# edit_bundle(t) = f(edit_history(t))
# edit_history(t) = edit_bundle(t-1)
# f is a time varying function <- see control_edit() above
def gen_edit(edit_history, 
             action_probs={'SUB' : 0.7, 
                           'INS' : 0.15, 
                           'DEL' : 0.05, 
                           'SWITCH_POS' : 0.05, 
                           'INS_WT' : 0.025, 
                           'DEL_WT' : 0.025},
             max_edited_bp=10, max_overwritten_wt_bp=3, 
             insert_offset_range=(-2, 2)) :

    '''
    Sample new edit using edit-action probabilities.

    ARGS
        edit_history: list
            [ref_seq, insert_seq, insert_offset, overwritten_wt_bp]
            ref_seq: 
                Reference sequence that was edited. 
            insert_seq:
                Use "" for no insert sequence.
            insert_offset: int
                insert seq offset from insert site.
            overwritten_wt_bp: int
                Number of WT bases that have been overwritten (deleted).
        action_probs: dict
            Dictionary of sampling probability per edit-action.
        max_edited_bp: int
            Maximum number of bp edits allowed.
        max_overwritten_wt_bp: int
            Maximum number of wild-type bp to overwrite (delete).
        insert_offset_range: tuple(int, int)
            Range of insert_seq offsets w.r.t. insert_site.
    RETURNS
        Edit bundle/history for making next edit.
    
    '''

    # Uniform substitution probability
    substitution_nt_dict = {
        'A' : ['C', 'G', 'T'],
        'C' : ['A', 'G', 'T'],
        'G' : ['C', 'A', 'T'],
        'T' : ['C', 'G', 'A']
                           }

    # Uniform insert shift probability
    switch_pos_dict = {
        insert_pos : [new_insert_pos for new_insert_pos in range(insert_offset_range[0], insert_offset_range[-1]+1) \
                      if new_insert_pos != insert_pos] for insert_pos in range(insert_offset_range[0], insert_offset_range[-1]+1)
                      }

    ref_seq, insert_seq, insert_offset, overwritten_wt_bp = edit_history

    # Sample an edit-action 

    # Choose random action
    rand_action = np.random.choice(list(action_probs.keys()), 
                                   p=list(action_probs.values()))

    # Create new edit bundle
    if rand_action == 'SWITCH_POS' :
        insert_offset = np.random.choice(switch_pos_dict[insert_offset])
        # TODO: Add option to update refseq <- to always center insert_seq and not insert_site
        # Taking a padded sequence with extract_refseq+modifying make_edit should allow this.
        # ChromBPNet has some translational invariance by jitter <- should be fine as it is.
    elif rand_action == 'DEL_WT' :
        overwritten_wt_bp += np.random.randint(1, min(max_edited_bp - (overwritten_wt_bp + len(insert_seq)) + 1, max_overwritten_wt_bp + 1))
    elif rand_action == 'INS_WT' :
        overwritten_wt_bp -= np.random.randint(1, overwritten_wt_bp + 1)
    elif rand_action == 'INS' :
        ins_pos = np.random.randint(0, len(insert_seq) + 1)
        ins_nt = np.random.choice(['A', 'C', 'G', 'T'])
        insert_seq = insert_seq[:ins_pos] + ins_nt + insert_seq[ins_pos:]
    elif rand_action == 'DEL' :
        del_pos = np.random.randint(0, len(insert_seq))
        insert_seq = insert_seq[:del_pos] + insert_seq[del_pos+1:]
    elif rand_action == 'SUB' :
        sub_pos = np.random.randint(0, len(insert_seq))
        sub_nt = np.random.choice(substitution_nt_dict[insert_seq[sub_pos]])
        insert_seq = insert_seq[:sub_pos] + sub_nt + insert_seq[sub_pos+1:]

    # Make new edit bundle
    edit_bundle = [ref_seq, insert_seq, insert_offset, overwritten_wt_bp]

    return edit_bundle
    
def make_edit(edit_bundle):
    
    '''
    Produce an edited sequence using edit bundle/history.
    
    '''

    ref_seq, insert_seq, insert_offset, overwritten_wt_bp = edit_bundle
    edited_sequence = edit_refseq(insert_seq, ref_seq, insert_offset=insert_offset, 
                                  len_overwrite=overwritten_wt_bp)

    return edited_sequence
                     