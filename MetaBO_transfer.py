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
from fsaf.policies.policies import iclr2020_NeuralAF
from fsaf.RL.ppo import PPO
from gym.envs.registration import register, registry
import shutil

rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "iclr2020_weight")

def MetaBO_transfer(env_spec,iter):
    shot_path = rootdir+"/{}".format(env_spec["env_id"])
    os.makedirs(rootdir+"/{}".format(env_spec["env_id"]), exist_ok=True)
    shutil.copy("{}/weights_{}".format(rootdir,1200),"{}/weights_{}".format(shot_path,1200))
    shutil.copy("{}/stats_{}".format(rootdir,1200),"{}/stats_{}".format(shot_path,1200))
    shutil.copy("{}/params_{}".format(rootdir,1200),"{}/params_{}".format(shot_path,1200))
    n_iterations = iter
    batch_size = 300
    n_workers = 2
    arch_spec = 4 * [200]
    ppo_spec = {
        "batch_size": batch_size,
        "max_steps": n_iterations * batch_size,
        "minibatch_size": batch_size // 20,
        "n_epochs": 4,
        "lr": 1e-4,
        "epsilon": 0.15,
        "value_coeff": 1.0,
        "ent_coeff": 0.01,
        "gamma": 0.98,
        "lambda": 0.98,
        "loss_type": "GAElam",
        "normalize_advs": True,
        "n_workers": n_workers,
        "env_id": env_spec["env_id"],
        "seed": 0,
        "env_seeds": list(range(n_workers)),
        "policy_options": {
            "activations": "relu",
            "arch_spec": arch_spec,
            "exclude_t_from_policy": True,
            "exclude_T_from_policy": True,
            "use_value_network": True,
            "t_idx": -2,
            "T_idx": -1,
            "arch_spec_value": arch_spec
        },
        "load":True,
        "load_path":shot_path,
        "param_iter":1200
    }

    # register environment
    if env_spec["env_id"] in registry.env_specs:
        del registry.env_specs[env_spec["env_id"]]
    register(
        id=env_spec["env_id"],
        entry_point="fsaf.environment.function_gym:FSAF",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # log data and weights go here, use this folder for evaluation afterwards
    logpath = os.path.join(rootdir, env_spec["env_id"])

    # set up policy
    policy_fn = lambda observation_space, action_space, deterministic: iclr2020_NeuralAF(observation_space=observation_space,
                                                                                action_space=action_space,
                                                                                deterministic=deterministic,
                                                                                options=ppo_spec["policy_options"])

    # do training
    print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
    ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=1)
    # learning curve is plotted online in separate process
    ppo.train()