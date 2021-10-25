# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import multiprocessing as mp
from datetime import datetime
from fsaf.policies.policies import NeuralAF
from fsaf.RL.DQN import DQN
from fsaf.RL.plot_learning_curve_online import plot_learning_curve_online
from gym.envs.registration import register, registry
import torch

def metaAdapt(best_iter, iter, dim, kernel, shot,logpath,shot_path,env_spec=None):
    # specify parameters
    n_iterations = iter

    batch_size = 50
    n_workers = shot
    arch_spec = 4 * [200]
    dqn_spec = {
        "batch_size": batch_size,
        "max_steps": n_iterations * batch_size,
        "lr": 1e-3,
        "inner_lr":1e-2,
        "gamma": 0.98,
        "buffer_size":1e3,
        "prior_alpha":0.3,
        "prior_beta":0.6,
        "outer_w":0.01,
        "n_steps":3,
        "task_size":3,
        "max_norm":40,
        "target_update_interval":5,
        "n_workers": n_workers,
        "env_id": env_spec["env_id"],
        "seed": 1000,
        "env_seeds": list(range(n_workers)),
        "policy_options": {
            "activations": "relu",
            "arch_spec": arch_spec,
            "use_value_network": True,
            "t_idx": -2,
            "T_idx": -1,
            "arch_spec_value": arch_spec,
        },
        "ML" : True,
        "load_itr" : best_iter,
        "load" : True,
        "load_path":logpath,
        "inner_loop_steps":1,
        "using_chaser":True,
        "use_multi_step_loss_optimization":False
    }
    if env_spec["env_id"] in registry.env_specs:
        del registry.env_specs[env_spec["env_id"]]
    # register environment
    register(
        id=env_spec["env_id"],
        entry_point="fsaf.environment.function_gym:FSAF",
        max_episode_steps=env_spec["T"] if "T" in env_spec else env_spec["T_max"],
        reward_threshold=None,
        kwargs=env_spec
    )
    os.makedirs(shot_path, exist_ok=True)


    # set up policy
    policy_fn = lambda observation_space, action_space, deterministic: NeuralAF(observation_space=observation_space,
                                                                                action_space=action_space,
                                                                                deterministic=deterministic,
                                                                                options=dqn_spec["policy_options"])

    # do training
    print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    dqn = DQN(policy_fn=policy_fn, params=dqn_spec, logpath=shot_path, save_interval=1)
    dqn.train()
    return shot_path

