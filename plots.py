import pandas as pd
import numpy as np
import os
import joypy
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb

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

def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)



path = 'results/relocate'
key = 'Critic/embed_obs.weight - singular vals'

df = get_data_from_files(path, key)


joypy.joyplot(df.transpose(), overlap=2,
              colormap=lambda x: color_gradient(x),
              linecolor='w', linewidth=.5)
plt.show()
