"""Train all models."""

from offline.offline_train import HIVES
from utilities.utils import params_extraction, load_pretrained_models
from utilities.optimization import set_optimizers
from rl.agent import VaLS
from rl.sampler import Sampler, ReplayBuffer
from datetime import datetime
from models.nns import Critic, SkillPolicy, StateEncoder, StateDecoder
import wandb
import os
import torch
import numpy as np
import copy
import pickle

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

import pdb

os.environ['WANDB_SILENT'] = "true"

wandb.login()

ENV_NAME = 'AdroitHandRelocateSparse-v1'
PARENT_FOLDER = 'checkpoints_relocate'
CASE_FOLDER = 'New_Baseline'

config = {
    # General hyperparams
    'device': 'cuda',
    'hidden_dim': 128,
    'env_id': ENV_NAME,
    
    # Offline hyperparams
    'vae_batch_size': 1024,
    'vae_lr': 6e-4,
    'priors_lr': 6e-4,
    'epochs': 501,
    'beta': 0.01,
    'length': 10,
    'z_skill_dim': 12,

    # Online hyperparams  
    'batch_size': 256,
    'action_range': 4,
    'critic_lr': 3e-4,
    'actor_lr': 3e-4,
    'discount': 0.97,
    'delta_skill': 32,
    'delta_length': 32,
    'z_state_dim': 8,
    'gradient_steps': 16,
    'max_iterations': int(200000 + 1),
    'buffer_size': int(200000 + 1),
    'test_freq': 100000,
    'reset_frequency': 25000,
    'singular_val_k': 1,

    # Run params
    'train_VAE_models': True,
    'train_priors': False,
    'train_rl': False,
    'load_VAE_models': False,
    'load_prior_models': False,
    'load_rl_models': False,
    'use_SAC': False,
    'render_results': False
}


path_to_data = f'datasets/{ENV_NAME}.pt'


def main(config=None):
    """Train all modules."""
    with wandb.init(project='SVD-Relocate-Offline', config=config,
                    notes='VAE training',
                    name='VAE'):

        config = wandb.config

        path = PARENT_FOLDER
        hives = HIVES(config)

        if not config.train_rl:
            hives.dataset_loader(path_to_data)

        skill_policy = SkillPolicy(hives.state_dim, hives.action_range,
                                   latent_dim=hives.z_skill_dim).to(hives.device)

        critic = Critic(hives.state_dim, hives.z_skill_dim).to(hives.device)
        
        sampler = Sampler(skill_policy, hives.models['Decoder'], hives.evaluate_decoder, config)

        experience_buffer = ReplayBuffer(hives.buffer_size, sampler.env,
                                         hives.z_skill_dim, config.reset_frequency,
                                         hives.length)

        vals = VaLS(sampler,
                    experience_buffer,
                    hives,
                    skill_policy,
                    critic,
                    config)
        
        hives_models = list(hives.models.values())

        models = [*hives_models, vals.skill_policy,
                  vals.critic, vals.critic,
                  vals.critic, vals.critic]
        
        names = [*hives.names, 'SkillPolicy', 'Critic1', 'Target_critic1',
                 'Critic2', 'Target_critic2']
    
        params_path = 'params_rl.pt'
        
        pretrained_params = load_pretrained_models(config, PARENT_FOLDER, params_path)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))
        
        params = params_extraction(models, names, pretrained_params)

        vals.experience_buffer.log_offline_dataset(f'datasets/{ENV_NAME}.pt',
                                                   params, hives.evaluate_encoder, hives.device)
        
        test_freq = config.epochs // 4
        test_freq = test_freq if test_freq > 0 else 1
    
        keys_optims = ['VAE_skills']
        keys_optims.extend(['SkillPrior', 'SkillPolicy'])
        keys_optims.extend(['Critic1', 'Critic2'])

        optimizers = set_optimizers(params, keys_optims, config.critic_lr)

        print('Training is starting')
    
        if config.train_VAE_models:
            for e in range(config.epochs):
                params = hives.train_vae(params,
                                         optimizers,
                                         config.vae_lr,
                                         config.beta)
                if e % test_freq == 0:
                    print(f'Epoch is {e}')
                    dt = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(params, f'{path}/params_{dt}_epoch{e}.pt')
                
        if config.train_priors:
            hives.set_skill_lookup(params)
            for i in range(config.epochs):
                params = hives.train_prior(params, optimizers,
                                           config.priors_lr)

            folder = 'Prior'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            fullpath = f'{path}/{folder}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
                
            torch.save(params, f'{path}/{folder}/params_{dt_string}_offline.pt')

        if config.train_rl:
            params = vals.training(params, optimizers, path, CASE_FOLDER)

        if config.render_results:
            vals.render_results(params, f'videos/{config.env_id}/{CASE_FOLDER}')

            
main(config=config)

