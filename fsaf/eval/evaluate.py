# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

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
import json
import numpy as np
import gym
import pickle as pkl
import torch
from datetime import datetime
from collections import namedtuple
from fsaf.policies.policies import *
from fsaf.RL.test_recorder import BatchRecorder, Transition

Result = namedtuple("Result",
                    "logpath env_id env_specs policy policy_specs deterministic load_iter T n_episodes rewards")


def write_overview_logfile(savepath, timestamp, env, env_seeds, policy, verbose=False):
    fname = "000_eval_overview_{}.txt".format(policy)
    s = ""
    s += "********* OVERVIEW ENVIRONMENT PARAMETERS *********\n"
    s += "Evaluation timestamp: {}\n".format(timestamp)
    s += "Environment-ID: {}\n".format(env.spec.id)
    s += "Environment-kwargs:\n"
    s += json.dumps(env.spec._kwargs, indent=2)
    s += "\n"
    s += "Environment-seeds:\n"
    s += str(env_seeds)
    with open(os.path.join(savepath, fname), "w") as f:
        print(s, file=f)
    if not verbose:
        print(s)


def load_fsaf_policy(logpath, load_iter, env, device, deterministic):
    with open(os.path.join(logpath, "params_" + str(load_iter)), "rb") as f:
        train_params = pkl.load(f)

    pi = NeuralAF(observation_space=env.observation_space,
                  action_space=env.action_space,
                  deterministic=deterministic,
                  options=train_params["policy_options"]).to(device)
    with open(os.path.join(logpath, "weights_" + str(load_iter)), "rb") as f:
        pi.load_state_dict(torch.load(f,map_location="cpu"))
    with open(os.path.join(logpath, "stats_" + str(load_iter)), "rb") as f:
        stats = pkl.load(f)

    return pi, train_params, stats
def load_iclr2020_metabo_policy(logpath, load_iter, env, device, deterministic):
    with open(os.path.join(logpath, "params_" + str(load_iter)), "rb") as f:
        train_params = pkl.load(f)

    pi = iclr2020_NeuralAF(observation_space=env.observation_space,
                  action_space=env.action_space,
                  deterministic=deterministic,
                  options=train_params["policy_options"]).to(device)
    with open(os.path.join(logpath, "weights_" + str(load_iter)), "rb") as f:
        pi.load_state_dict(torch.load(f,map_location="cpu"))
    with open(os.path.join(logpath, "stats_" + str(load_iter)), "rb") as f:
        stats = pkl.load(f)

    return pi, train_params, stats

def eval_experiment(eval_spec,name=""):
    env_id = eval_spec["env_id"]
    env_seed_offset = eval_spec["env_seed_offset"]
    policy = eval_spec["policy"]
    logpath = eval_spec["logpath"]
    policy_specs = eval_spec["policy_specs"]
    savepath = eval_spec["savepath"]
    n_workers = eval_spec["n_workers"]
    n_episodes = eval_spec["n_episodes"]
    assert n_episodes % n_workers == 0
    T = eval_spec["T"]
    if policy != "FSAF":
        pi = None
        deterministic = None
        load_iter = None

    os.makedirs(savepath, exist_ok=True)

    env_seeds = env_seed_offset + np.arange(n_workers)
    dummy_env = gym.make(env_id)
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
    write_overview_logfile(savepath=savepath, timestamp=timestamp, env=dummy_env, policy=policy,
                           env_seeds=env_seeds)
    env_specs = dummy_env.spec._kwargs

    # prepare the policies
    if policy == "GP-UCB":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        D = dummy_env.unwrapped.D
        policy_fn = lambda *_: UCB(feature_order=feature_order,
                                   kappa=policy_specs["kappa"],
                                   D=D,
                                   delta=policy_specs["delta"])
    elif policy == "EI":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        policy_fn = lambda *_: EI(feature_order=feature_order)
    elif policy == "PI":
        feature_order = dummy_env.unwrapped.feature_order_eval_envs
        policy_fn = lambda *_: PI(feature_order=feature_order, xi=policy_specs["xi"])
    elif policy == "MES":
        policy_fn = lambda *_: MES(dim=policy_specs["dim"])
    elif policy == "FSAF":
        load_iter = eval_spec["load_iter"]
        deterministic = eval_spec["deterministic"]
        pi, policy_specs, _ = load_fsaf_policy(logpath=logpath, load_iter=load_iter, env=dummy_env,
                                                 device="cpu", deterministic=deterministic)
        latent = None
        theta = None
        if "bmaml" in eval_spec and eval_spec["bmaml"]:
            with open(os.path.join(logpath, "theta_" + str(load_iter)), "rb") as f:
                theta = torch.load(f,map_location="cpu").detach().cpu()
        if "mmaml" in eval_spec and eval_spec["mmaml"] and load_iter < 100:
            with open(os.path.join(logpath, "latent_" + str(load_iter)), "rb") as f:
                latent = torch.load(f,map_location="cpu")

        policy_fn = lambda osp, asp, det: NeuralAF(observation_space=osp,
                                                   action_space=asp,
                                                   deterministic=det,
                                                   options=policy_specs["policy_options"])
    elif policy == "iclr2020_MetaBO":
        load_iter = eval_spec["load_iter"]
        deterministic = eval_spec["deterministic"]
        pi, policy_specs, _ = load_iclr2020_metabo_policy(logpath="iclr2020_weight/{}".format(eval_spec["env_id"]), load_iter=load_iter, env=dummy_env,
                                                 device="cpu", deterministic=deterministic)

        policy_fn = lambda osp, asp, det: iclr2020_NeuralAF(observation_space=osp,
                                                   action_space=asp,
                                                   deterministic=det,
                                                   options=policy_specs["policy_options"])
    elif policy == "Random":
        pass  # will be dealt with separately below
    elif policy == "TAF-ME":
        policy_fn = lambda *_: TAF(datafile=policy_specs["TAF_datafile"], mode="me")
    elif policy == "TAF-RANKING":
        policy_fn = lambda *_: TAF(datafile=policy_specs["TAF_datafile"], mode="ranking", rho=1.0)
    else:
        raise ValueError("Unknown policy!")
    dummy_env.close()

    # evaluate the experiment
    if policy != "Random":
        br = BatchRecorder(size=T * n_episodes, env_id=env_id, env_seeds=env_seeds, policy_fn=policy_fn,
                           n_workers=n_workers, deterministic=deterministic)
        if policy == "FSAF":
            br.set_worker_weights(pi=pi,theta=theta, latent=latent)
        if "iclr2020_MetaBO" in policy:
            br.set_worker_weights(pi=pi)
        br.record_batch(gamma=1.0, lam=1.0)  # gamma, lam do not matter for evaluation
        transitions = Transition(*zip(*br.memory.copy()))
        rewards = transitions.reward
        br.cleanup()
    else:
        env = gym.make(env_id)
        env.seed(env_seed_offset)
        rewards = []
        for _ in range(n_episodes):
            rewards = rewards + env.unwrapped.get_random_sampling_reward()
        env.close()

    # save result
    policy=policy+"_iter{}".format(eval_spec["load_iter"]) if policy == "FSAF" else policy
    if policy == "iclr2020_MetaBO":
        policy += str(eval_spec["load_iter"]) if eval_spec["load_iter"] == 1200 else "T"
    result = Result(logpath=logpath, env_id=env_id, env_specs=env_specs, policy=policy, 
                    policy_specs=policy_specs,
                    deterministic=deterministic, load_iter=load_iter, T=T, n_episodes=n_episodes, rewards=rewards)
    fn = "result_{}_{}".format(policy,name)
    with open(os.path.join(savepath, fn), "wb") as f:
        pkl.dump(result, f)