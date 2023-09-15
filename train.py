"""Train all models."""

from utilities.utils import params_extraction, load_pretrained_models, hyper_params
from utilities.optimization import set_optimizers
from rl.agent import VaLS
from rl.sampler import Sampler, ReplayBuffer
from models.nns import Critic, SkillPolicy
import wandb
import os
import torch
import numpy as np


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

ANT = 'AntMaze_Medium-v3'
KITCHEN = 'FrankaKitchen-v1'
RELOCATE = 'AdroitHandRelocateSparse-v1'
PEN = 'AdroitHandPenSparse-v1'

ENV_NAME = RELOCATE

PARENT_FOLDER = f'checkpoints/{ENV_NAME}'        
CASE_FOLDER = 'Baseline'

if 'Ant' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(4e6) + 1,
                         'buffer_size': int(4e6) + 1,
                         'reset_frequency': 50000,
                         'test_freq': 400000}

elif 'Adroit' in ENV_NAME or 'Franka' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(4e5) + 1,
                         'buffer_size': int(4e5) + 1,
                         'reset_frequency': 25000,
                         'test_freq': 100000}

elif 'Franka' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(4e5) + 1,
                         'buffer_size': int(4e5) + 1,
                         'reset_frequency': 25000,
                         'test_freq': 100000}
    

elif 'Fetch' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(2e5) + 1,
                         'buffer_size': int(2e5) + 1,
                         'reset_frequency': 6250,
                         'test_freq': 50000}

else:
    raise ValueError('This environment is not registered in the code')


config = {
    # General hyperparams
    'device': 'cuda',
    'hidden_dim': 128,
    'env_id': ENV_NAME,
    
    # Online hyperparams  
    'batch_size': 256,
    'learning_rate': 3e-4,
    'discount': 0.99,
    'delta_skill': 8,
    'gradient_steps': 4,
    'singular_val_k': 1,

    # Algo selection params
    'SVD': True,
    'Replayratio': False,
    'Underparameter': False,
    'SAC': False,

    'folder_sing_vals': 'SVD',
    
    # Run params
    'train_rl': True,
    'load_rl_models': False,
    'render_results': False
}


config.update(hyperparams_dict)

path_to_data = f'datasets/{ENV_NAME}.pt'


def main(config=None):
    """Train all modules."""
    with wandb.init(project=f'SVD-{ENV_NAME}', config=config,
                    notes='SVD baseline.',
                    name='SVD'):

        config = wandb.config

        path = PARENT_FOLDER

        action_dim, state_dim = hyper_params.env_dims(config.env_id)

        skill_policy = SkillPolicy(state_dim, action_dim).to(config.device)

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

