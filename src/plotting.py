import pandas as pd
import logomaker

import seaborn as sns
from matplotlib import pyplot as plt

# Plot fitness per iteration
def plot_fitness(*edit_records, ax=None):

    # Setup axes
    if ax is None:
        ax=plt.subplot()
    
    # Plot fitness per iteration
    for edit_record in edit_records:
        sns.lineplot(x='iteration', 
                     y='fitness', 
                     data=edit_record,
                     ax=ax)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
    
    return ax

# Plot temperature scaling
def plot_temp_scaling(edit_record, ax=None):

    # Setup axes
    if ax is None:
        ax=plt.subplot()
    
    # Plot fitness per iteration
    sns.lineplot(x='iteration', 
                    y='temperature', 
                    data=edit_record,
                    ax=ax)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Fitness')
    
    return ax

# Plot scores on sequence
def plot_scores(scores, ax=None):

    # Setup axes
    if ax is None:
        ax=plt.subplot()
    
    # Plot scores
    score_df = pd.DataFrame(scores, columns=['A', 'T', 'C', 'G'])

    # create Logo object
    logo = logomaker.Logo(score_df, ax=ax)

    # style using Logo methods
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True, 
                      bounds=[score_df.sum(1).min(), score_df.sum(1).max()])

    logo.ax.get_yaxis().set_ticks([score_df.sum(1).min(), 
                                   score_df.sum(1).max()])
    logo.ax.get_yaxis().set_ticklabels([score_df.sum(1).min().round(3), 
                                        score_df.sum(1).max().round(3)])                      

    logo.ax.get_xaxis().set_ticks([])
    logo.ax.get_xaxis().set_ticklabels([])

    logo.ax.set_xlim([logo.ax.get_xlim()[0]-5,
                      logo.ax.get_xlim()[1]])

    return ax


