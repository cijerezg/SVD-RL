import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

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

for data in datasets:
    print(data)
    analysis = SingularValsDataset(data)
    analysis.sample()
