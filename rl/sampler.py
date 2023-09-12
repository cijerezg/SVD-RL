"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict, compute_cum_rewards
import gymnasium as gym
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal

import pdb

WIDTH = 4 * 640
HEIGHT = 4 * 480


class Sampler(hyper_params):
    def __init__(self, skill_policy, args):
        super().__init__(args)

        self.skill_policy = skill_policy
        self.env = gym.make(self.env_id)


    def skill_step(self, params, obs):
        obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)

        with torch.no_grad():
            action, _, mu, std = functional_call(self.skill_policy,
                                                  params['SkillPolicy'],
                                                  obs_t)

            action = action.cpu().numpy()

        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, dict):
            obs = self.convert_obs_dict_to_array(obs)

        done = True if terminated or truncated else False

        if 'Fetch' in self.env_key:
            done = True if info['is_success'] > 0.0 else done

        elif 'Adroit' in self.env_key:
            done = True if info['success'] else done

        next_obs = obs
        next_obs_t = torch.from_numpy(next_obs).to(self.device).to(torch.float32)

        with torch.no_grad():
            next_action, _, _ , _ = functional_call(self.skill_policy,
                                                    params['SkillPolicy'],
                                                    next_obs_t)
            next_action = next_action.cpu().numpy()

        return next_obs, reward, action, next_action, done

    def skill_iteration(self, params, done=False, obs=None):
        if done or obs is None:
            obs, _ = self.env.reset()
            if isinstance(obs, dict):
                obs = self.convert_obs_dict_to_array(obs)           

        return obs, self.skill_step(params, obs)

    def convert_obs_dict_to_array(self, obs):
        if 'Franka' in self.env_key:
            observation = obs['observation']
            achieved = np.hstack(list(obs['achieved_goal'].values()))
            desired = np.hstack(list(obs['desired_goal'].values()))
            obs = np.concatenate((observation, achieved, desired), axis=0)
            return obs
        else:
            return np.hstack(list(obs.values()))

       
class ReplayBuffer(hyper_params):
    def __init__(self, env, args):
        super().__init__(args)

        self.obs_buf = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.z_buf = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.tracker = np.zeros((self.buffer_size,), dtype=bool)
        self.cum_reward = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.norm_cum_reward = np.zeros((self.buffer_size, 1), dtype=np.float32)
        
        self.size, self.ptr = 0, 0
        self.env = env


    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs],
                         cum_reward=self.cum_reward[idxs],
                         norm_cum_reward=self.norm_cum_reward[idxs])
        return batch


    def update_tracking_buffers(self, ep_reward):
        last_ep_idx = np.where(self.done_buf[0:self.ptr - 1])[0].max() + 1
        self.cum_reward[last_ep_idx:self.ptr, :] = ep_reward
        mean = self.cum_reward[0:self.ptr, :].mean()
        std = self.cum_reward[0:self.ptr, :].std()
        self.norm_cum_reward[0:self.ptr, :] = (self.cum_reward[0:self.ptr, :] - mean) / (std + 1e-4)
