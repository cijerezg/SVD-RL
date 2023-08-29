"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict
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
    def __init__(self, skill_policy, decoder, eval_decoder, args):
        super().__init__(args)

        self.skill_policy = skill_policy
        self.decoder = decoder
        self.eval_decoder = eval_decoder
        MAX_EPISODE_STEPS = 257 if self.env_key == 'adroit' else 256

        self.env = gym.make(self.env_id)
        self.env._max_episode_steps = MAX_EPISODE_STEPS
        
    def skill_execution(self, actions, frames=None):
        obs_trj, rew_trj, done_trj = [], [], []
        aux_frames = []
        
        for i in range(actions.shape[0]):
            next_obs, rew, done, info = self.env.step(actions[i, :])
            if frames is not None:
                if self.env_key != 'kitchen':
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT,
                                                mode='offscreen',
                                                camera_name='vil_camera')
                    aux_frames.append(frame)
                else:
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT)
                    aux_frames.append(frame)
                    
            if self.env_key != 'kitchen':
                done = info['goal_achieved'] if len(info) == 1 else True
            obs_trj.append(next_obs)
            rew_trj.append(rew)
            done_trj.append(done)
        if frames is not None:
            frames.append(aux_frames)

        return obs_trj, rew_trj, done_trj, frames

    def skill_step(self, params, obs, frames=None):
        obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)
        obs_t = obs_t.reshape(1, -1)
        obs_trj, rew_trj, done_trj = [], [], []

        with torch.no_grad():
            z_sample, _, _, _ = functional_call(self.skill_policy,
                                                params['SkillPolicy'],
                                                obs_t)

            self.decoder.reset_hidden_state(z_sample)
            self.decoder.func_embed_z(z_sample)

            for i in range(self.length):
                action = self.eval_decoder(obs_t, params)
                action = action.cpu().detach().numpy()
                action = action.squeeze()
                obs, rew, _, done, info = self.env.step(action)
                obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)
                
                # Collect trajectories
                obs_trj.append(obs)
                rew_trj.append(rew)
                done_trj.append(done)


        if frames is not None:
            done = True if sum(done_trj) > 0 else False
            return obs_trj[-1], done, frames

        next_obs_t = torch.from_numpy(obs_trj[-1]).to(self.device).to(torch.float32)
        next_obs_t = next_obs_t.reshape(1, -1)

        with torch.no_grad():
            next_z_sample, _, _, _ = functional_call(self.skill_policy,
                                                     params['SkillPolicy'],
                                                     next_obs_t)
           
        next_obs = obs_trj[-1]
        if self.env_key == 'kitchen':
            rew = sum(rew_trj)
        elif self.env_key == 'adroit':
            rew = rew_trj[-1]

        z = z_sample.cpu().numpy()
        next_z = next_z_sample.cpu().numpy()
        done = True if sum(done_trj) > 0 else False

        return next_obs, rew, z, next_z, done

    def skill_iteration(self, params, done=False, obs=None):
        if done or obs is None:
            obs, _ = self.env.reset()

        return obs, self.skill_step(params, obs)

    def skill_iteration_with_frames(self, params, done=False, obs=None, frames=None):
        if done or obs is None:
            obs, _ = self.env.reset()

        frames = self.skill_step(params, obs, frames)

        return frames
    
       
class ReplayBuffer(hyper_params):
    def __init__(self, size, env, lat_dim, reset_ratio, args):
        super().__init__(args)

        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.tracker = np.zeros((size,), dtype=bool)
        self.cum_reward = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.idx_sampler = np.arange(size)
        self.idx_tracker = np.zeros((size, 1), dtype=np.float32)
        self.threshold = 0.0

        self.sampling_ratio = reset_ratio
        self.env = env
        self.lat_dim = lat_dim

    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tracker[self.ptr] = True
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32, s_ratio=1.0):
        # idxs = np.random.randint(0, self.size, size=batch_size)
        # idxs = np.random.power(1 + .25 * self.ptr / 1e5, size=batch_size)
        # idxs = np.array(idxs * self.size, dtype=np.int32)

        idxs = np.random.randint(0, self.size, size=int((1 - s_ratio) * 256))
        
        self.idx_tracker[idxs] += 1

        idxs_off = np.random.randint(0, self.size, size=int(s_ratio * 256))

        obs = np.concatenate((self.obs_buf[idxs], self.offline_obs_buf[idxs_off]), axis=0)
        z = np.concatenate((self.z_buf[idxs], self.offline_z_buf[idxs_off]), axis=0)
        next_z = np.concatenate((self.next_z_buf[idxs], self.offline_next_z_buf[idxs_off]), axis=0)
        next_obs = np.concatenate((self.next_obs_buf[idxs], self.offline_next_obs_buf[idxs_off]), axis=0)
        rew = np.concatenate((self.rew_buf[idxs], self.offline_rew_buf[idxs_off]), axis=0)
        done = np.concatenate((self.done_buf[idxs], self.offline_done_buf[idxs_off]), axis=0)
        cum_r = np.concatenate((self.cum_reward[idxs], self.offline_cum_reward_buf[idxs_off]), axis=0)

        total_idxs = np.concatenate((idxs, idxs_off), axis=0)

        batch = AttrDict(observations=obs,
                         next_observations=next_obs,
                         z=z,
                         next_z=next_z,
                         rewards=rew,
                         dones=done,
                         cum_reward=cum_r,
                         idxs=total_idxs)
        return batch

    def sample_recent_eps(self, batch_size=128):
        idx = self.size
        batch = AttrDict(observations=self.obs_buf[idx - batch_size:idx],
                         next_observations=self.next_obs_buf[idx - batch_size:idx],
                         z=self.z_buf[idx - batch_size:idx],
                         next_z=self.next_z_buf[idx - batch_size:idx],
                         rewards=self.rew_buf[idx - batch_size:idx],
                         dones=self.done_buf[idx - batch_size:idx],
                         tracker=self.tracker[idx - batch_size:idx])

        return batch

    def update_tracking_buffers(self, ep_reward):
        last_ep_idx = np.where(self.done_buf[0:self.ptr - 1])[0].max() + 1
        self.cum_reward[last_ep_idx:self.ptr, :] = ep_reward


    def prune_buffers(self, all_rewards):
        all_rewards = np.array(all_rewards)[-500:]
        self.threshold = np.percentile(all_rewards, 60)
        self.tracker[self.cum_reward < self.threshold] = False
        
        vals = self.tracker.sum()
        
        self.obs_buf[0: vals] = self.obs_buf[self.tracker[:, 0], :]
        self.next_obs_buf[0: vals] = self.next_obs_buf[self.tracker[:, 0], :]
        self.z_buf[0: vals] = self.z_buf[self.tracker[:, 0], :]
        self.next_z_buf[0: vals] = self.next_z_buf[self.tracker[:, 0], :]        
        self.rew_buf[0: vals] = self.rew_buf[self.tracker[:, 0], :]
        self.done_buf[0: vals] = self.done_buf[self.tracker[:, 0], :]
        self.cum_reward[0: vals] = self.cum_reward[self.tracker[:, 0], :]

        self.ptr, self.size = vals, vals
        
        return self.threshold

    def log_offline_dataset(self, path, params, eval_encoder, device):
        dataset = torch.load(path)

        dataset['rewards'] = self.d4rl_reward_map(dataset['rewards'])
        
        total_size = dataset['actions'].shape[0]
        self.offline_size = int(total_size / self.length)
        iters = 5000

        self.offline_next_obs_buf = np.zeros((self.offline_size, *self.env.observation_space.shape),
                                             dtype=np.float32)
        self.offline_z_buf = np.zeros((self.offline_size, self.lat_dim), dtype=np.float32)
        self.offline_next_z_buf = np.zeros((self.offline_size, self.lat_dim), dtype=np.float32)

        keys = ['observations', 'actions', 'rewards', 'timeouts']
        dataset = {key: dataset[key] for key in keys}

        actions = torch.from_numpy(dataset['actions']).to(device)

        batch_size = int(total_size / iters)
        chunk = int(batch_size / self.length)

        for i in range(iters):
            with torch.no_grad():
                z, pdf, mu, std = eval_encoder(actions[batch_size * i: batch_size * (i + 1), :], params)
            mu = mu.cpu().numpy()
            self.offline_z_buf[chunk * i: chunk * (i + 1), :] = mu
            
        self.offline_obs_buf = dataset['observations'][0::self.length]
        self.offline_rew_buf = dataset['rewards'][0::self.length]
        self.offline_done_buf = dataset['timeouts'][self.length-1::self.length]

        timeouts = dataset['timeouts']
        timeouts[-1] = True

        aux_val = dataset['rewards'][dataset['timeouts']]

        done_idx = np.arange(dataset['timeouts'].shape[0])
        done_idx = done_idx[dataset['timeouts']]
        done_idx = np.insert(done_idx, 0, 0)
        self.offline_cum_reward_buf = np.repeat(aux_val, np.diff(done_idx))
        self.offline_cum_reward_buf = self.offline_cum_reward_buf[0::self.length]

        self.offline_next_obs_buf[0:self.offline_size - 1, :] = self.offline_obs_buf[1:, :]
        self.offline_next_z_buf[0:self.offline_size - 1, :] = self.offline_z_buf[1:, :]

        self.offline_obs_buf = self.offline_obs_buf[~self.offline_done_buf, :]
        self.offline_z_buf = self.offline_z_buf[~self.offline_done_buf, :]
        self.offline_next_obs_buf = self.offline_next_obs_buf[~self.offline_done_buf, :]
        self.offline_next_z_buf = self.offline_next_z_buf[~self.offline_done_buf, :]
        self.offline_rew_buf = self.offline_rew_buf[~self.offline_done_buf].reshape(-1, 1)
        self.offline_cum_reward_buf = self.offline_cum_reward_buf[~self.offline_done_buf].reshape(-1, 1)
        self.offline_done_buf = self.offline_done_buf[~self.offline_done_buf].reshape(-1, 1)
        self.offline_size = self.offline_obs_buf.shape[0]
        
    def d4rl_reward_map(self, reward):
        if self.env_key == 'adroit':
            return np.where(reward < 9, -.1, 10.0).astype(np.float32)

        else:
            return reward
