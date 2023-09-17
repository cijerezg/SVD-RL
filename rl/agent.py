"""Training RL algorithm."""

import sys
sys.path.insert(0, '../')

import torch
from utilities.optimization import GD_full_update, Adam_update
from utilities.utils import hyper_params, process_frames, reset_params
from torch.func import functional_call
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import wandb
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.autograd as autograd
import os
import pdb
import pickle
import time
from stable_baselines3.common.utils import polyak_update
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import math


MAX_SKILL_KL = 100
INIT_LOG_ALPHA = 0

class VaLS(hyper_params):
    def __init__(self,
                 sampler,
                 experience_buffer,
                 skill_policy,
                 critic,
                 args):

        super().__init__(args)

        self.sampler = sampler
        self.critic = critic
        self.skill_policy = skill_policy
        self.experience_buffer = experience_buffer
        
        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True,
                                            device=self.device)
        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.learning_rate)

        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = []
        self.log_data = 0
        self.log_data_freq = 2000
        self.prior = Normal(0, 1)
        
    def training(self, params, optimizers, path, name):
        self.iterations = 0
        ref_params = copy.deepcopy(params)

        obs = None    # These two lines are to setup the RL env.
        done = False  # Only need to be called once.

        while self.iterations < self.max_iterations:

            params, obs, done = self.training_iteration(params, done,
                                                        optimizers,
                                                        self.learning_rate,
                                                        ref_params,
                                                        obs=obs)

            if self.iterations % self.test_freq == 0 and self.iterations > 0:
                dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                print(f'Current iteration is {self.iterations}')
                print(dt_string)
                fullpath = f'{path}/{name}'
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)
                filename = f'{path}/{name}/params_rl_{dt_string}_iter{self.iterations}.pt'
                torch.save(params, filename)
                with open(f'{fullpath}/class_{dt_string}_{self.iterations}', 'wb') as file:
                    pickle.dump(self, file)

            if self.iterations % self.log_data_freq == 0:
                wandb.log({'Iterations': self.iterations})
            self.iterations += 1

            if self.SVD or self.Replayratio:
                if self.iterations % self.reset_frequency == 0:
                    if self.SVD:
                        self.reset_frequency = 2 * self.reset_frequency
                    # self.gradient_steps = math.ceil(self.gradient_steps / 2) # To evaluate
                    self.interval_iteration = 0
                    keys = ['SkillPolicy', 'Critic1', 'Critic2']
                    ref_params = copy.deepcopy(params)
                    
                    if self.Replayratio:
                        params, optimizers = reset_params(params, keys, optimizers, self.learning_rate)
                    elif self.SVD:
                        params, optimizers = self.rescale_singular_vals(params, keys, optimizers, self.learning_rate)
                        self.singular_val_k = 1.5 * self.singular_val_k
                        
                    self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                                        requires_grad=True,
                                                        device=self.device)
                    self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=self.learning_rate)
                                    
        return params

    def training_iteration(self,
                           params,
                           done,
                           optimizers,
                           lr,
                           ref_params,
                           obs=None):
               
        obs, data = self.sampler.skill_iteration(params, done, obs)

        next_obs, rew, z, next_z, done = data

        self.reward_per_episode += rew

        self.experience_buffer.add(obs, next_obs, z, next_z, rew, done)

        if done:
            if self.total_episode_counter > 1:
                self.experience_buffer.update_tracking_buffers(self.reward_per_episode)
            wandb.log({'Reward per episode': self.reward_per_episode,
                       'Total episodes': self.total_episode_counter})

            self.reward_logger.append(self.reward_per_episode)
            self.reward_per_episode = 0
            self.total_episode_counter += 1

        log_data = True if self.log_data % self.log_data_freq == 0 else False

        if len(self.reward_logger) > 8 and log_data:
            wandb.log({'Cumulative reward dist': wandb.Histogram(np.array(self.reward_logger))})
            wandb.log({'Average reward over 100 eps': np.mean(self.reward_logger[-100:])}, step=self.iterations)

        self.log_data = (self.log_data + 1) % self.log_data_freq

        if self.experience_buffer.size > self.log_data_freq - 1:
            
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False # Only log data once for multi grad steps.
                policy_losses, critic1_loss, critic2_loss = self.losses(params, log_data, ref_params)
                losses = [*policy_losses, critic1_loss, critic2_loss]
                names = ['SkillPolicy', 'Critic1', 'Critic2']
                params = Adam_update(params, losses, names, optimizers, lr)
                polyak_update(params['Critic1'].values(),
                              params['Target_critic1'].values(), 0.005)
                polyak_update(params['Critic2'].values(),
                              params['Target_critic2'].values(), 0.005)

                if log_data:
                    with torch.no_grad():
                        dist_init1 = self.distance_to_params(params, ref_params, 'Critic1', 'Critic1')
                        dist_init_pol = self.distance_to_params(params, ref_params, 'SkillPolicy', 'SkillPolicy')
                    
                        wandb.log({'Critic/Distance to init weights': dist_init1,
                                   'Policy/Distance to init weights Skills': dist_init_pol}) 
           
        return params, next_obs, done

    def losses(self, params, log_data, ref_params):
        batch = self.experience_buffer.sample(batch_size=self.batch_size)

        obs = torch.from_numpy(batch.observations).to(self.device)
        next_obs = torch.from_numpy(batch.next_observations).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        next_z = torch.from_numpy(batch.next_z).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        cum_reward = torch.from_numpy(batch.cum_reward).to(self.device)
        norm_cum_reward = torch.from_numpy(batch.norm_cum_reward).to(self.device)
        
        if log_data:
            singular_vals = self.compute_singular_vals(params)
            if self.folder_sing_vals is not None:
                dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                path = f'results/{self.env_key}/{self.folder_sing_vals}/{self.iterations}'
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(f'{path}/{dt_string}.npy', singular_vals, allow_pickle=True)

            for log_name, log_val in singular_vals.items():
                wandb.log({log_name: wandb.Histogram(log_val)})
                        
            # # Critic analysis
            critic_test_arg = torch.cat([obs, z], dim=1)

            trials = 32

            new_z = z.reshape(z.shape[0], 1, -1).repeat(1, trials, 1)
            new_z = new_z.reshape(-1, new_z.shape[-1])
            z_rand = torch.rand(new_z.shape).to(self.device)
            new_z = new_z + torch.randn(new_z.shape).to(self.device) / 5

            new_obs = obs.reshape(obs.shape[0], 1, -1).repeat(1, trials, 1)
            new_obs = new_obs.reshape(-1, new_obs.shape[-1])

            new_critic_arg = torch.cat([new_obs, new_z], dim=1)
            new_critic_arg_rand = torch.cat([new_obs, z_rand], dim=1)
        
            with torch.no_grad():
                q1_r, q2_r = self.eval_critic(critic_test_arg, params)
                
                q1_rep, q2_rep = self.eval_critic(new_critic_arg, params)
                q1_rep, q2_rep = q1_rep.reshape(-1, trials, 1), q2_rep.reshape(-1, trials, 1)

                q1_rand, _ = self.eval_critic(new_critic_arg_rand, params)
                q1_rand = q1_rand.reshape(-1, trials, 1)

                mean_diff_rand = q1_r - q1_rand.mean(1)

            eval_test_ave = self.log_scatter_3d(q1_r, q1_rand.mean(1), cum_reward, rew,
                                                'Q val', 'Q random', 'Cum reward', 'Reward')

            wandb.log({'Critic/Mean diff dist rand': wandb.Histogram(mean_diff_rand.cpu()),
                       'Critic/Mean diff average rand': mean_diff_rand.mean().cpu(),
                       'Policy/Eval policy critic_random': eval_test_ave,
                       'Gradient updates': self.gradient_steps,
                       'Rewards batch': wandb.Histogram(rew.cpu())
                       })

                                                                 
        ####
        target_critic_arg = torch.cat([next_obs, next_z], dim=1)
        critic_arg = torch.cat([obs, z], dim=1)
        
        with torch.no_grad():                                
            q_target1, q_target2 = self.eval_critic(target_critic_arg, params,
                                                    target_critic=True)
            
        q_target = torch.cat((q_target1, q_target2), dim=1)
        q_target, _ = torch.min(q_target, dim=1)

        q1, q2 = self.eval_critic(critic_arg, params)
        
        if log_data:
            with torch.no_grad():
                dist1 = self.distance_to_params(params, params, 'Critic1', 'Target_critic1')

                q_refs, _ = self.eval_critic(critic_arg, ref_params)
                q_refs_target, _ = self.eval_critic(target_critic_arg, ref_params,
                                                    target_critic=True)

            bellman_terms = self.log_scatter_3d(q1, q_target.unsqueeze(dim=1), rew, cum_reward,
                                                'Q val', 'Q target', 'Reward', 'Cum reward')
            
            wandb.log({'Critic/Distance critic to target 1': dist1,
                       'Critic/Bellman terms': bellman_terms})

        q_target = rew + (0.97 * q_target).reshape(-1, 1) * (1 - dones)
        q_target = torch.clamp(q_target, min=-100, max=100)

        critic1_loss = F.mse_loss(q1.squeeze(), q_target.squeeze(),
                                  reduction='none')
        critic2_loss = F.mse_loss(q2.squeeze(), q_target.squeeze(),
                                  reduction='none')        

        if self.SVD:
            with torch.no_grad():
                weights = F.sigmoid(norm_cum_reward).squeeze()
        else:
            weights = torch.ones_like(critic1_loss)
            
        critic1_loss = critic1_loss * weights
        critic2_loss = critic2_loss * weights
                
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()
        
        if log_data:
            wandb.log(
                {'Critic/Critic1 Grad Norm': self.get_gradient(critic1_loss, params, 'Critic1')})
        
        z_sample, pdf, mu, std = self.eval_skill_policy(obs, params)

        q_pi_arg = torch.cat([obs, z_sample], dim=1)
        
        q_pi1, q_pi2 = self.eval_critic(q_pi_arg, params)
        q_pi = torch.cat((q_pi1, q_pi2), dim=1)
        q_pi, _ = torch.min(q_pi, dim=1)
        
        #skill_prior = -torch.clamp(pdf.entropy(), max=MAX_SKILL_KL).mean()
        skill_prior = torch.clamp(kl_divergence(pdf, self.prior), max=MAX_SKILL_KL).mean()

        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        skill_prior_loss = alpha_skill * skill_prior

        q_pi = q_pi * weights
        
        q_val_policy = -torch.mean(q_pi)
        skill_policy_loss = q_val_policy + skill_prior_loss

        policy_losses = [skill_policy_loss]
            
        loss_alpha_skill = torch.exp(self.log_alpha_skill) * \
            (self.delta_skill - skill_prior).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss_alpha_skill.backward()
        self.optimizer_alpha_skill.step()
          
        if log_data:
            with torch.no_grad():
                z_ref, _, mu_ref, _ = self.eval_skill_policy(obs, ref_params)
                q_pi_ref_arg = torch.cat([obs, z_ref], dim=1)
                q_pi_ref, _ = self.eval_critic(q_pi_ref_arg, params)
                diff = q_pi.reshape(-1, 1) - q1
                mu_diff_as = F.l1_loss(mu, z, reduction='none').mean(1)

            pi_diff = self.log_scatter_3d(q_pi.reshape(-1, 1), diff, mu_diff_as.unsqueeze(dim=1), cum_reward,
                                          'Q pi', 'Diff Q pi and Q', 'Diff mu pi and z', 'Cum reward')

            pi_reward = self.log_scatter_3d(q_pi.reshape(-1, 1), rew, mu_diff_as.unsqueeze(dim=1), cum_reward,
                                            'Q pi', 'Reward', 'Diff mu pi and z', 'Cum reward')

                
            wandb.log(
                {'Policy/current_q_values': wandb.Histogram(q_pi.detach().cpu()),
                 'Policy/current_q_values_average': q_pi.detach().mean().cpu(),
                 'Policy/Z abs value mean': z_sample.abs().mean().detach().cpu(),
                 'Policy/Z std': z_sample.std().detach().cpu(),
                 'Policy/Z distribution': wandb.Histogram(z_sample.detach().cpu()),
                 'Policy/Mean STD': std.mean().detach().cpu(),
                 'Policy/Mu dist': wandb.Histogram(mu.detach().cpu()),
                 'Policy/Pi diff data': pi_diff,
                 'Policy/Pi reward': pi_reward,
                 })

            wandb.log(
                {'Priors/Alpha skill': alpha_skill.detach().cpu(),
                 'Priors/skill_prior_loss': skill_prior.detach().cpu()})

            wandb.log(
                {'Critic/Critic loss': critic1_loss,
                 'Critic/Q values': wandb.Histogram(q1.detach().cpu())})
        
        return policy_losses, critic1_loss, critic2_loss

    def eval_skill_prior(self, state, params):
        """Evaluate the policy.

        It takes the current state and params. It evaluates the
        policy.

        Parameters
        ----------
        state : Tensor
            The current observation of agent
        params : dictionary with all parameters for models
            It contains all relevant parameters, e.g., policy, critic,
            etc.
        """
        z_prior = functional_call(self.vae.models['SkillPrior'],
                                  params['SkillPrior'], state)
        return z_prior
    

    def eval_skill_policy(self, state, params):
        sample, pdf, mu, std = functional_call(self.skill_policy,
                                               params['SkillPolicy'],
                                               state)
        return sample, pdf, mu, std

    def eval_critic(self, arg, params, target_critic=False):
        if target_critic:
            name1, name2 = 'Target_critic1', 'Target_critic2'
        else:
            name1, name2 = 'Critic1', 'Critic2'

        q1 = functional_call(self.critic, params[name1], arg)
        q2 = functional_call(self.critic, params[name2], arg)

        return q1, q2

    def log_histogram_2d(self, x, y, xlabel, ylabel):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        data = np.concatenate([x, y], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel])

        fig_heatmap = px.density_heatmap(df, x=xlabel, y=ylabel,
                                         marginal_x='histogram',
                                         marginal_y='histogram',
                                         nbinsx=60,
                                         nbinsy=60)

        return fig_heatmap

    def log_scatter_3d(self, x, y, z, color, xlabel, ylabel, zlabel, color_label):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

        data = np.concatenate([x, y, z, color], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel, zlabel, color_label])
        
        fig_scatter = px.scatter_3d(df, x=xlabel, y=ylabel,
                                    z=zlabel, color=color_label)
        fig_scatter.update_layout(scene=dict(aspectmode='cube'))

        return fig_scatter

    def compute_singular_vals(self, params):
        models = ['Critic1', 'SkillPolicy']
        nicknames = ['Critic', 'Policy']

        singular_vals = {}
        
        with torch.no_grad():
            for name, mods in zip(nicknames, models):
                for key, param in params[mods].items():
                    if len(param.shape) < 2:
                        continue
                    S = torch.linalg.svdvals(param)
                    singular_vals[f'{name}/{key} - singular vals'] = S.cpu()

        return singular_vals

    def rescale_singular_vals(self, params, keys, optimizers, lr):
        k = self.singular_val_k
        
        with torch.no_grad():
            for model in keys:
                for key, param in params[model].items():
                    if len(param.shape) < 2:
                        continue
                    U, S, Vh = torch.linalg.svd(param, full_matrices=False)
                    bounded_S = k * (1 - torch.exp(-S / k))
                    new_param = U @ torch.diag(bounded_S) @ Vh
                    params[model][key] = nn.Parameter(new_param)

                optimizers[model] = Adam(params[model].values(), lr)

        return params, optimizers

    def compute_singular_vals_loss(self, params):
        models = ['Critic1', 'Critic2', 'SkillPolicy']
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        

    def get_gradient(self, x, params, key):
        grads = autograd.grad(x, params[key].values(), retain_graph=True,
                              allow_unused=True)

        grads = [grad for grad in grads if grad is not None]
        try:
            grads_vec = nn.utils.parameters_to_vector(grads)
        except RuntimeError:
            pdb.set_trace()
        return torch.norm(grads_vec).detach().cpu()

    def distance_to_params(self, params1, params2, name1, name2):
        with torch.no_grad():
            vec1 = nn.utils.parameters_to_vector(params1[name1].values())
            target_vec1 = nn.utils.parameters_to_vector(params2[name2].values())
        return torch.norm(vec1 - target_vec1)

    def percentile_hist(self, x):
        x = x.detach().cpu().numpy()
        x = np.minimum(x, 2)

        return x              

    def testing(self, params):
        done = False
        obs = None

        rewards = []
        episodes_w_reward = 0
        test_episodes = 500
        length_samples = []
        length_over_time = []

        
        for j in range(test_episodes):
            self.sampler.env.reset()
            check_episode = True
            step = 0
            while not done:
                _, data = self.sampler.skill_iteration(params,
                                                       done,
                                                       obs)

                obs, reward, _, _, l_samp, _, done, _, _ = data

                if step < 64:
                    length_over_time.append(['0-63', l_samp[0]])
                elif step < 128:
                    length_over_time.append(['64-127', l_samp[0]])
                elif step < 196:
                    length_over_time.append(['128-195', l_samp[0]])
                elif step < 257:
                    length_over_time.append(['196-255', l_samp[0]])

                length_samples.append(l_samp[0])
                step += self.level_lengths[l_samp[0]]
                if check_episode and reward > 0.0:
                    episodes_w_reward += 1
                    check_episode = False
                
                rewards.append(reward)

            done = False

        pdb.set_trace()
        print(np.unique(length_samples, return_counts=True))
        evol_lengths = np.array(length_over_time)        
        df = pd.DataFrame(evol_lengths, columns=['Step', 'Length'])
        df.to_csv('runs/case0.csv', index=False)
        average_reward = sum(rewards) / test_episodes
        return average_reward

    def render_results(self, params, foldername):
        test_episodes = 10
        
        for j in range(test_episodes):
            done = False
            obs = None

            frames = []
            self.sampler.env.reset()

            while not done:
                obs, done, frames = self.sampler.skill_iteration_with_frames(params,
                                                                             done=done,
                                                                             obs=obs,
                                                                             frames=frames)

            process_frames(frames, self.env_id, f'{foldername}/test_{j}')
