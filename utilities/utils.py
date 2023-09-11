"""Various useful functions."""

import torch
import torch.nn as nn
import gymnasium as gym
from collections import OrderedDict
import numpy as np
import os
import copy
import skvideo.io
import matplotlib.pyplot as plt
import pdb
from torch.optim import Adam


class hyper_params:
    """Set hyperparameters."""

    def __init__(self, args):
        """Val args comes from wanbd."""
        # General hyperparams
        for key, value in args.items():
            setattr(self, key, value)
        
        self.device = torch.device(args.device)
        self.action_dim, self.state_dim = hyper_params.env_dims(args.env_id)
        if 'Relocate' in self.env_id:
            self.env_key = 'relocate'
        elif 'Pen' in self.env_id:
            self.env_key = 'pen'
        elif 'Ant' in self.env_id:
            self.env_key = 'ant'
        elif 'Egg' in self.env_id:
            self.env_key = 'egg'
        elif 'ManipulatePen':
            self.env_key = 'pen_hand'
        # Assign env_key for all other environments.

    @staticmethod
    def env_dims(env_id):
        """Get action and observation dimensions."""
        env = gym.make(env_id)
        if 'Ant' in env_id:
            action_dim = env.action_space.sample().shape[0]
            state_dim = env.observation_space.sample()['observation'].shape[0]
        else:
            action_dim = env.action_space.shape[0]
            state_dim = env.observation_space.shape[0]
        env.close()
        del env
        return action_dim, state_dim


def params_extraction(models: list,
                      names: list,
                      pretrained_params,
                      ) -> dict:
    """Get and init params from model to use with functional call.

    The models list contains the pytorch model. The parameters are
    initialized with bias and std 0, and rest with orthogonal init.

    Parameters
    ----------
    models : list
        Each element contains the pytorch model.
    names : list
        Strings that contains the name that will be assigned.

    Returns
    -------
    dict
        Each dictionary contains params ready to use with functional
        call.
    """
    
    params = OrderedDict()
    for model, name_m, pre_params in zip(models, names, pretrained_params):
        par = {}
        gain = 1.0

        if name_m == 'Target_critic1':
            params[name_m] = copy.deepcopy(params['Critic1'])
            continue
        if name_m == 'Target_critic2':
            params[name_m] = copy.deepcopy(params['Critic2'])
            continue

        if pre_params is None:
            for name, param in model.named_parameters():
                if len(param.shape) == 1:
                    init = torch.nn.init.constant_(param, 0.0)
                else:
                    init = torch.nn.init.xavier_normal_(param, gain=gain)
                par[name] = nn.Parameter(init)

        else:
            for name, param in model.named_parameters():
                try:
                    init = pre_params[name]
                    par[name] = nn.Parameter(init)
                except KeyError:
                    pdb.set_trace()
        params[name_m] = copy.deepcopy(par)

    return params


def load_pretrained_models(args, folder, filename):
    """Load pretrained models."""
    pretrained_params = []
    
    path = f'{folder}/{filename}'
    if os.path.isfile(path):
        params = torch.load(path)
    else:
        return pretrained_params

    if args.load_rl_models:
        print('RL models were loaded.')
        pretrained_params.append(params['SkillPolicy'])
        pretrained_params.append(params['Critic1'])
        pretrained_params.append(params['Target_critic1'])
        pretrained_params.append(params['Critic2'])
        pretrained_params.append(params['Target_critic2'])
        
    return pretrained_params


def compute_cum_rewards(data, done_key='timeouts', max_reward=10.0):
    """Compute cumulative reward (non-discounted).

    In D4RL datasets, sometimes the episode continues even after the agent
    has reached the goal. In those cases, the cumulative reward sum is
    stopped at the step that the agent reached the goal.

    The cumulative rewards are normalized with 0 mean and 1 std.

    Parameters
    ----------
    data : dict
        This should be a dictionary containing the data. The key keys are observations,
        actions, rewards, timeouts/terminals.
    done : str
        This is the key that indicates if an episode is done. For D4RL it might terminals
        or timeouts.
    max_reward : float
        If this reward is obtained, then the cumulative reward stops adding. By definition,
        this is the reward at which the agent reaches the goal.
    """   
    done_idx = np.arange(data[done_key].shape[0])
    done_idx = done_idx[data[done_key]]
    done_idx = np.insert(done_idx, 0, 0)
    done_idx = np.append(done_idx, data[done_key].shape[0])

    cum_rewards = np.zeros(data[done_key].shape[0],dtype=np.float32)
    cum_rew_unique = []
    for j in range(done_idx.shape[0] - 1):
        aux_rew = data['rewards'][done_idx[j]:done_idx[j + 1]]
        mask = aux_rew == max_reward
        if len(np.argwhere(mask)) > 0:
            mask[np.argwhere(mask)[0]] = False  # This computes the index of the first True
        # Note that an inverse mask is being used.
        cum_reward = np.maximum(-20.0, np.sum(aux_rew[~mask]))
        cum_rewards[done_idx[j]:done_idx[j + 1]] = cum_reward
        cum_rew_unique.append(cum_reward)

    cum_rew_stats = np.array(cum_rew_unique)

    norm_cum_rewards = (cum_rewards - cum_rew_stats.mean()) / (cum_rew_stats.std() + 1e-4)

    return cum_rewards, norm_cum_rewards


def reset_params(params, keys, optimizers, lr):
    for key in keys:
        for name, param in params[key].items():
            if len(param.shape) == 1:
                init = torch.nn.init.constant_(param, 0.0)
            else:
                init = torch.nn.init.xavier_normal_(param)
            params[key][name] = nn.Parameter(init)

        optimizers[key] = Adam(params[key].values(), lr)

    return params, optimizers
        

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d
        

def process_frames(frames, env_id, foldername):
    if 'kitchen' in env_id:
        heigth_up, heigth_down = 620, 1180
        width_left, width_right = 960, 1680

    else:
        heigth_up, heigth_down = 0, 1800
        width_left, width_right = 480, 2000

    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for idx, frame in enumerate(frames):
        skill_length = len(frame)
        print(f'Skill length for {idx} is {skill_length}')
        if 'kitchen' not in env_id:
            if idx == 0:
                frame0 = np.flip(frame[0], 0)
                frame0 = frame0[heigth_up:heigth_down, width_left:width_right]
            frame = np.flip(frame[-1], 0)
            frame = frame[heigth_up:heigth_down, width_left:width_right]
        else:
            if idx == 0:
                frame0 = frame[0][heigth_up:heigth_down, width_left:width_right]
            frame = frame[-1][heigth_up:heigth_down, width_left:width_right]
        if idx == 0:
            plt.imshow(frame0)
            plt.axis('off')
            plt.savefig(f'{foldername}/init_frame',
                        bbox_inches='tight')
            plt.close()
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f'{foldername}/skill_number_{idx}_skil_length_{skill_length}',
                    bbox_inches='tight')
        plt.close()
        
    flat_frames = [np.stack(frame) for frame in frames]
    video = np.concatenate(flat_frames, axis=0)
    if 'kitchen' not in env_id:
        video = np.flip(video, 1) # For adroit envs, the image is upside down.

    skvideo.io.vwrite(f'{foldername}/video.mp4', video)
        
