import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd


class SingularValsDataset:
    def __init__(self, dataset_id):
        data = torch.load(f'datasets/{dataset_id}')
        obs = data['observations']
        actions = data['actions']

        self.critic_arg = np.concatenate((obs, actions), axis=1)
        self.shuffle()
    
    def shuffle(self):
        idxs = np.arange(self.critic_arg.shape[0])
        np.random.shuffle(idxs)
        self.critic_arg = self.critic_arg[idxs, :].squeeze()

    def compute_svd(self, data):
        return np.linalg.svd(data, compute_uv=False)

    def sample(self):
        val = 64
        base_val = 4
        cases = 6

        singular_vals = []

        for i in range(cases):
            aux_data = self.critic_arg[:val, :]
            pdb.set_trace()
            S = self.compute_svd(aux_data)
            print(S[0]/S[-1])
            val = val * base_val

        
datasets = ['kitchen-mixed-v0.pt',
            'relocate-cloned-v1.pt',
            'relocate-expert-v1.pt',
            'antmaze-medium-diverse-v2.pt',
            'pen-cloned-v1.pt']

with open('datasets/class_04-09-2023-13_30_04_80000', 'rb') as file:
    data = pickle.load(file)

obs = data.experience_buffer.obs_buf
acts = data.experience_buffer.z_buf
done = data.experience_buffer.done_buf.squeeze()

idx_1 = np.where(done == 1)[0]

done_idx = np.arange(done.shape[0])

done_idx = done_idx[done.astype(bool)]

for idx in idx_1:
    done[idx: idx+4] = 1

done = done.astype(bool)

obs = obs[~done, :]
acts = acts[~done, :]


args = np.concatenate((obs, acts), axis=1)

small = args[:500, :]

large = args[1000:50000, :]


def sampling_process(data):
    trials = 100

    S_vals = []

    for i in range(trials):    
        idxs = np.random.randint(0, data.shape[0], 256)
        S = np.linalg.svd(data[idxs, :], compute_uv=False)
        S_vals.append(S)

    S_vals = np.vstack(S_vals).flatten()

    return S_vals

S_vals_small = sampling_process(small)
S_vals_large = sampling_process(large)

label1 = ['Small'] * S_vals_small.shape[0]
label2 = ['Large'] * S_vals_large.shape[0]

vals = np.concatenate((S_vals_small, S_vals_large), axis=0).reshape(-1, 1)
labels = np.concatenate((label1, label2), axis=0).reshape(-1, 1)
data = np.concatenate((vals, labels), axis=1)

df = pd.DataFrame(data, columns=['vals', 'labels'])

df['vals'] = pd.to_numeric(df['vals'])

sns.histplot(data=df, x='vals', hue='labels', bins=100)
plt.show()



# val = 64
# base_val = 2
# cases = 12

# for i in range(cases):
#     aux_data = args[:val, :]
#     S = np.linalg.svd(aux_data, compute_uv=False)
#     print(S[0]/S[-1])
#     print(val)
#     print('--------------------------------')
#     plt.hist(S, bins=25)
#     plt.show()
#     val = val * base_val

# pdb.set_trace()


# for data in datasets:
#     print(data)
#     analysis = SingularValsDataset(data)
#     analysis.sample()
