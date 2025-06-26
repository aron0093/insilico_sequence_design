import numpy as np

from alphagenome.data import genome
from alphagenome.models import dna_client

import sys # Adjust this path
sys.path.append('../')
from _utils import one_hot_decode, reverse_complement

# Load model via API
def load_model(API_KEY):
    # Load model
    model = dna_client.create(API_KEY)
    return model

def predict_func(sequence, models, strand=0, slice_idx=None, ontology_terms=None, output_type=None, rc=True, mode='scoring'):

    # Check shape
    if len(sequence)>1048576:
        midpoint = int(len(sequence)/2)
        sequence = sequence[midpoint-int(1048576/2):midpoint+int(1048576/2)]
        print('Sequence was subset to center 1048576 bases')

    # Revert onehot to str
    if type(sequence) is not str:
        sequence = one_hot_decode(np.expand_dims(sequence,0))[0]

    # Take rc if enabled
    if rc:
        sequence_rc = reverse_complement(sequence)

    # Setup output types
    output_type_dict = {'RNA': dna_client.OutputType.RNA_SEQ,
                        'CAGE': dna_client.OutputType.CAGE,
                        'ATAC': dna_client.OutputType.ATAC,
                        'DNASE': dna_client.OutputType.DNASE}
    requested_outputs = [output_type_dict[output_type]]
    
    # Make prediction
    outputs = models.predict_sequence(sequence, ontology_terms=ontology_terms, requested_outputs=requested_outputs)

    # Extract signal # Assumes single output type
    if output_type=='RNA':
        prediction = outputs.rna_seq.values[slice_idx]
    elif output_type=='CAGE':
        prediction = outputs.cage.values[slice_idx]
    elif output_type=='ATAC':
        prediction = outputs.atac.values[slice_idx]
    elif output_type=='DNASE':
        prediction = outputs.dnase.values[slice_idx]

    # Choose strand for stranded signals
    if output_type=='RNA' or output_type=='CAGE':
        if strand==1:
            prediction = prediction[:,0]
        elif strand==-1:
            prediction = prediction[:,1]

    if rc:
        prediction_rc = predict_func(sequence_rc, models, strand, slice_idx, ontology_terms, output_type, rc=False, mode=None)
        prediction += prediction_rc[::-1]
        prediction /= 2

    if mode=='scoring':
        prediction = prediction.sum()

    return prediction
        