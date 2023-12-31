"""Implement neural nets for SiRL."""

import pdb
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

LOG_STD_MAX = 0
LOG_STD_MIN = -20

        
class Encoder(nn.Module):
    def __init__(self, z_dim, input_dim, seq_length, hidden_dim=128):
        super().__init__()

        self.embed_a = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            batch_first=True)

        self.hidden = nn.Linear(hidden_dim, 64)
        
        self.mu = nn.Linear(64, z_dim)
        self.log_std = nn.Linear(64, z_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        x = self.embed_a(x)

        x, _ = self.lstm(x)
        x = x[:, -1, :] # Take the output of the last encoder
        x = F.relu(self.hidden(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        density = Normal(mu, std)
        sample = density.rsample()
        
        return sample, density, mu, std

        
class StateConditionedDecoder(nn.Module):
    def __init__(self, z_dim, out_dim, obs_dim, seq_length, hidden_dim=128):
        super().__init__()
        self.embed_z = nn.Linear(z_dim, hidden_dim)
        self.embed_obs = nn.Linear(obs_dim, hidden_dim)

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            batch_first=True)

        self.hidden = nn.Linear(hidden_dim, 64)
        self.out = nn.Linear(64, out_dim)
        
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

    def forward(self, obs):
        # x: [batch_size, input_size (z_dim)]
        obs = F.relu(self.embed_obs(obs))

        x = obs + self.embedded_z[:, self.counter, :]

        out = F.relu(self.hidden(x))
        out = self.out(out)
        self.counter += 1
    
        return out

    def func_embed_z(self, z):
        x = F.relu(self.embed_z(z))
        x = x.reshape(-1, 1, x.shape[-1])
        x = x.repeat(1, self.seq_length, 1)
        self.embedded_z, _ = self.lstm(x)
        self.counter = 0


    def reset_hidden_state(self, x):
        self.hn = torch.zeros(1, x.shape[0], self.hidden_dim).to(x.device)
        self.cn = torch.zeros(1, x.shape[0], self.hidden_dim).to(x.device)
        



def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    use_batch_norm: bool = False,
    squash_output: bool = False,
) -> List[nn.Module]:
    """Create a multilayer perceptron (MLP).

    An MLP is a collection of fully-connected layers (linear
    transformations) followed by an activation function.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector
    output_dim : int
        Dimension of the output vector
    net_arch : List[int]
        Architecture of the neural net. It represents the number of
        units per layer. The length of this list is the number of
        layers.
    activation_fn : Type[nn.Module]
        The activation function to use after each layer.
    use_batch_norm : bool
        Whether to include a batch norm layer.
    squash_output : bool
        Whether to squash the output using a Tanh activation function.

    Returns
    -------
    List[nn.Module]
        List containing all layers, including activation functions.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(num_features=net_arch[idx]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class SkillPrior(nn.Module):
    def __init__(self, input_obs, net_arch=[128] * 6, latent_dim=12):
        super().__init__()
        
        self.embed_obs = nn.Linear(input_obs, 128)

        latent_policy = create_mlp(128, output_dim=-1,
                                   net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)
        self.mu = nn.Linear(net_arch[-1], latent_dim)
        self.log_std = nn.Linear(net_arch[-1], latent_dim)
        self.input_obs = input_obs
    
    def forward(self, obs):
        x = self.embed_obs(obs[:, :self.input_obs])
        policy_latent = self.latent_policy(x)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        std = torch.exp(torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX))
        
        density = Normal(mu, std)
        
        return density

class LengthPrior(nn.Module):
    def __init__(self, input_dim, net_arch=[128] * 6, lengths=3):
        super().__init__()
        latent_policy = create_mlp(input_dim, output_dim=-1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)
        self.midway = nn.Linear(net_arch[-1], 32)
        self.out = nn.Linear(32, lengths)

    def forward(self, observation):
        policy_latent = self.latent_policy(observation)
        policy_latent = F.relu(self.midway(policy_latent))
        prior = F.log_softmax(self.out(policy_latent), dim=1)
        return prior



class LengthPolicy(nn.Module):
    def __init__(self, input_dim, lengths, net_arch=[128] * 6):
        super().__init__()
        latent_policy = create_mlp(input_dim, -1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)

        self.preout = nn.Linear(net_arch[-1], 32)
        self.out = nn.Linear(32, lengths)

    def forward(self, obs):
        policy_latent = self.latent_policy(obs)
        z_l = F.relu(self.preout(policy_latent))
        length_prior = F.log_softmax(self.out(z_l), dim=1)

        dist = Categorical(logits=length_prior)
        samples = dist.sample()

        return length_prior, samples
        


class SkillPolicy(nn.Module):
    def __init__(self, input_obs, latent_dim, net_arch=[128] * 4):
        super().__init__()
        self.embed_obs = nn.Linear(input_obs, 128)

        latent_policy = create_mlp(128, -1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)

        self.mu = nn.Linear(net_arch[-1], latent_dim)
        self.log_std = nn.Linear(net_arch[-1], latent_dim)

    def forward(self, data):
        x = self.embed_obs(data)
        policy_latent = self.latent_policy(x)
                        
        mu = self.mu(policy_latent)
        mu = torch.tanh(mu)

        log_std = self.log_std(policy_latent)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        
        return sample, density, mu, std
    

class Critic(nn.Module):
    def __init__(self, obs_dim, z_dim, net_arch=[256] * 2):
        super().__init__()
        self.embed_obs = nn.Linear(obs_dim, 128)
        self.embed_z = nn.Linear(z_dim, 128)
        
        self.latent_policy = create_mlp(128, -1, net_arch=net_arch)
        self.latent_policy = nn.Sequential(*self.latent_policy)
        self.latent_policy2 = create_mlp(256, -1, net_arch=net_arch)
        self.latent_policy2 = nn.Sequential(*self.latent_policy2)
        self.out = nn.Linear(32, 1)
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.post_pol = nn.Linear(net_arch[-1], 32)
        

    def forward(self, data):
        e_obs = self.embed_obs(data[:, :self.obs_dim])
        e_z = self.embed_z(data[:, self.obs_dim:self.obs_dim + self.z_dim])
        x = e_obs + e_z

        qvalue = self.latent_policy(x)

        qvalue = self.latent_policy2(qvalue)
        qvalue = F.relu(self.post_pol(qvalue))
        qvalue = self.out(qvalue)

        return qvalue
