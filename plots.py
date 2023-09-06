import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb

sns.set_theme(style='whitegrid')

def get_file_index(filename: str) -> int:
    """Extract the index from the filename."""
    return int(filename.replace('singular_vals', '').replace('.npy', ''))

def load_npy_file(filepath: str) -> dict:
    """Load a npy file and return the dictionary contained within."""
    return np.load(filepath, allow_pickle=True).item()

def get_data_from_files(path: str, key: str) -> pd.DataFrame:
    """Load all npy files from a given directory, extract the data corresponding to a given key, and return a dataframe."""
    data = {}
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            filepath = os.path.join(path, filename)
            file_data = load_npy_file(filepath)
            if key in file_data:
                index = get_file_index(filename)
                data[index] = file_data[key].numpy().flatten()
    return pd.DataFrame.from_dict(data, orient='index')



def plot_singular_vals(path, keys):
    fig, axes = plt.subplots(len(keys), 1, figsize=(15, 10))
                             #layout='constrained')

    for i, key in enumerate(keys):
        df = get_data_from_files(path, key)

        df = df.stack().reset_index(level=0)
        df.columns = ['index', 'value']
        df['index'] = df['index'] / 100
        
        sns.histplot(df, x='index', y='value', ax=axes[i])
        axes[i].xaxis.grid()
        axes[i].tick_params(axis='both', labelsize=18)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        if i == 0:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])

    fig.supxlabel('Environment steps (1e3)', fontsize=20,y=0.02)
    fig.supylabel('Singular values', fontsize=20, x=0.06)
    key = key.replace('/', "")
    plt.savefig(f'figures/singular_vals{key}.png', bbox_inches='tight')
    plt.close()
        

        
path = 'results/relocate'
keys = ['Critic/embed_obs.weight - singular vals',
       'Critic/post_pol.weight - singular vals']


plot_singular_vals(path, keys)
