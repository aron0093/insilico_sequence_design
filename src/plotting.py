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