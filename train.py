"""Train all models."""

from offline.offline_train import HIVES
from utilities.utils import params_extraction, load_pretrained_models, hyper_params
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



# When using kitchen, remember in D4RL the tasks are open microwave,
# move kettle, flip light switch, and open (slide) cabinet.

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

import pdb

os.environ['WANDB_SILENT'] = "true"

wandb.login()

# The ids for envs are:
# AdroitHandRelocateSparse-v1
# AdroitHandPenSparse-v1
# AntMaze_Medium-v3
# FrankaKitchen-v1
# FetchPickAndPlace-v2
# FetchPush-v2

ENV_NAME = 'FrankaKitchen-v1'

PARENT_FOLDER = f'checkpoints/{ENV_NAME}'        
CASE_FOLDER = 'Baseline'

config = {
    # General hyperparams
    'device': 'cuda',
    'hidden_dim': 128,
    'env_id': ENV_NAME,
    
    # Online hyperparams  
    'batch_size': 256,
    'action_range': 4,
    'learning_rate': 3e-4,
    'discount': 0.99,
    'delta_skill': 48,
    'gradient_steps': 1,
    'max_iterations': int(8e5 + 1),
    'buffer_size': int(8e5 + 1),
    'test_freq': 100000,
    'reset_frequency': 12500,
    'singular_val_k': 1,

    # Algo selection params
    'SVD': False,
    'Replayratio': False,
    'Underparameter': False,
    'SAC': False,

    'folder_sing_vals': 'SPiRL-16',
    
    # Run params
    'train_rl': True,
    'load_rl_models': False,
    'render_results': False
}


path_to_data = f'datasets/{ENV_NAME}.pt'


def main(config=None):
    """Train all modules."""
    with wandb.init(project=f'SVD-{ENV_NAME}', config=config,
                    notes='SPiRL to compare evolution of singular vals.',
                    name='Random policy'):

        config = wandb.config

        path = PARENT_FOLDER

        action_dim, state_dim = hyper_params.env_dims(config.env_id)

        skill_policy = SkillPolicy(state_dim, config.action_range,
                                   latent_dim=action_dim).to(config.device)

        critic = Critic(state_dim, action_dim).to(config.device)
        
        sampler = Sampler(skill_policy, config)

        experience_buffer = ReplayBuffer(sampler.env, config)

        vals = VaLS(sampler,
                    experience_buffer,
                    skill_policy,
                    critic,
                    config)
        
        models = [vals.skill_policy,
                  vals.critic, vals.critic,
                  vals.critic, vals.critic]
        
        names = ['SkillPolicy', 'Critic1', 'Target_critic1',
                 'Critic2', 'Target_critic2']
    
        params_path = 'Prior/params_04-09-2023-02_43_11_offline.pt'
        
        pretrained_params = load_pretrained_models(config, PARENT_FOLDER, params_path)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))
        
        params = params_extraction(models, names, pretrained_params)

        optimizers = set_optimizers(params, names, config.learning_rate)

        print('Training is starting')
    
        if config.train_rl:
            params = vals.training(params, optimizers, path, CASE_FOLDER)

        if config.render_results:
            vals.render_results(params, f'videos/{config.env_id}/{CASE_FOLDER}')

            
main(config=config)

