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
from fsaf.eval.evaluate import eval_experiment
from fsaf.eval.plot_results import plot_results
from gym.envs.registration import register, registry
from datetime import datetime
from fsaf.RL.util import get_best_iter_idx,get_best_iter_idx_meta,get_last_iter_idx,get_best_iter_idx_reward

from fsaf.policies.taf.generate_taf_data import *
import shutil
import torch
import copy
from metaAdapt import metaAdapt
from MetaBO_transfer import MetaBO_transfer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="mix_all_card1000",
                    help="model name in log file")
parser.add_argument("-lsH", "--ls_high",type=float,default=0.55,
                    help="ength scale high")
parser.add_argument("-lsL", "--ls_low",type=float,default=0.5,
                    help="length scale low")
parser.add_argument("-det", "--deterministic",type=bool,default=True,
                    help="max or categorical")        
parser.add_argument("-cuda", "--cuda",type=int,default=0,
                    help="gpu num")        
parser.add_argument("-is", "--init_sample",type=int,default=10,
                    help="gp init samples")                 
args = parser.parse_args()
if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

# set evaluation parameters
shot_step = 5
afs_to_evaluate = ["FSAF","iclr2020_MetaBO", "EI", "PI","GP-UCB","MES","TAF-ME"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "log","FSAF-GP-v0")
logpath = os.path.join(rootdir,args.model)
dims = [3,5,3,5,3,5,3,5]
kernels = ["Matern32","Matern32","RBF","RBF","SM","SM","SM","SM"]
periods = [[None]]*4+[[0.3,0.6,0.9]]*2+[[0.2,0.4,0.8]]*2
idx = -1
for kernel, dim in zip(kernels, dims):
    idx += 1
    det = args.deterministic
    period = periods[idx]
    shot_path = os.path.join(logpath,"shot{}D_{}_{}_{}".format(dim,kernel,str(args.ls_low).replace(".",""),period[0]))

    best_iter = 710 #get_best_iter_idx_reward(logpath,get_last_iter_idx(logpath),init_iter=100)
    shot=5
    env_spec = {
        "env_id": "FSAF-{}D{}_{}-{}-v0".format(dim,kernel,str(args.ls_high).replace(".",""),str(args.ls_low).replace(".","")),
        "D": dim,
        "f_type": "GP",
        "f_opts": {
                "kernel": kernel,
                "periods":period,
                "lengthscale_low": args.ls_low,
                "lengthscale_high": args.ls_high,
                "noise_var_low": 0.1,
                "noise_var_high": 0.1,
                "signal_var_low": 1.0,
                "signal_var_high": 1.0,
                "min_regret": 1e-20, 
                "mix_kernel": False,
                "metaTrainShot":shot},
        "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"],
        "T": 50,
        "n_init_samples": args.init_sample,
        "pass_X_to_pi": False,
        "kernel": kernel,
        "kernel_lengthscale": None,
        "kernel_variance": None,
        "noise_variance": None,
        "use_prior_mean_function": False,
        "local_af_opt": False,
        "cardinality_domain": 2000,
        "reward_transformation": "neg_log10"  
    }
    env_spec_ppo = copy.deepcopy(env_spec)
    env_spec_ppo["features"] = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep","budget"]
    metaAdapt(best_iter=best_iter,iter=shot_step, dim=dim, kernel=kernel,shot=shot,logpath=logpath,shot_path=shot_path,env_spec=env_spec)
    os.makedirs(shot_path, exist_ok=True)
    shutil.copy("{}/weights_{}".format(logpath,best_iter),"{}/weights_{}".format(shot_path,best_iter))
    shutil.copy("{}/stats_{}".format(logpath,best_iter),"{}/stats_{}".format(shot_path,best_iter))
    shutil.copy("{}/params_{}".format(logpath,best_iter),"{}/params_{}".format(shot_path,best_iter))
    shutil.copy("{}/theta_{}".format(logpath,best_iter),"{}/theta_{}".format(shot_path,best_iter))
    MetaBO_transfer_iter = 100
    MetaBO_transfer(env_spec = env_spec_ppo,iter=MetaBO_transfer_iter)
    n_workers = 10
    n_episodes = 100
    savepath = os.path.join(shot_path, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%Sdet={}".format(det)))

    test_iters = [0,5]

    for li in test_iters:
        # evaluate all afs
        for af in afs_to_evaluate:
            # set af-specific parameters
            if af == "FSAF":
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"] # dimensionality agnostic
                pass_X_to_pi = False
                if li == 0:
                    load_iter = best_iter
                else:
                    load_iter = li-1 
                T_training = None
                deterministic = det
                policy_specs = {} 
            elif af == "iclr2020_MetaBO":
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep","budget"]
                pass_X_to_pi = False
                T_training = None
                if li == 0:
                    load_iter = 1200
                else:
                    load_iter = get_best_iter_idx_reward("iclr2020_weight/{}".format(env_spec["env_id"]),MetaBO_transfer_iter-1,init_iter=1)
                deterministic = False
                policy_specs = {}  # will be loaded from the logfiles
            elif af == "MES":
                features = ["posterior_mean", "posterior_std"]
                T_training = None
                pass_X_to_pi = True
                load_iter = None 
                deterministic = None 
                policy_specs = {"dim":dim}
            elif af == "TAF-ME" or af == "TAF-RANKING":
                generate_taf_data_gps(M=5,kernel=kernel,ls=[args.ls_low,args.ls_high],D=dim,periods=periods[idx])
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
                T_training = None
                pass_X_to_pi = True
                load_iter = None  
                deterministic = None 
                policy_specs = {"TAF_datafile": os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "policies", "taf", "taf_{}{}D.pkl".format(kernel,dim))}
            else:
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"]
                T_training = None
                pass_X_to_pi = False
                load_iter = None 
                deterministic = None 
                if af == "EI":
                    policy_specs = {}
                elif af == "PI":
                    policy_specs = {"xi" : 0.5}
                elif af == "GP-UCB":
                    policy_specs = {"kappa":"gp_ucb", "delta": 0.0001}
                elif af == "Random":
                    policy_specs = {}
                else:
                    raise ValueError("Unknown AF!")

            # define environment
            env_spec = {
                "env_id": "FSAF-{}D{}_{}-{}-v0".format(dim,kernel,str(args.ls_high).replace(".",""),str(args.ls_low).replace(".","")),
                "D": dim,  # FSAF is dimensionality agnostic and can be evaluated for any D
                "f_type": "GP",
                "f_opts": {
                        "periods":period,
                        "kernel": kernel,
                        "lengthscale_low": args.ls_low,
                        "lengthscale_high": args.ls_high,
                        "noise_var_low": 0.1,
                        "noise_var_high": 0.1,
                        "signal_var_low": 1.0,
                        "signal_var_high": 1.0,
                        "min_regret": 0,
                        "mix_kernel": False},
                "features": features,
                "T": 100,
                "T_training": T_training,
                "n_init_samples": args.init_sample,
                "pass_X_to_pi": pass_X_to_pi,
                # will be set individually for each new function to the sampled hyperparameters
                "kernel": kernel,
                "kernel_lengthscale": None,
                "kernel_variance": None,
                "noise_variance": None,
                "use_prior_mean_function": False,
                "local_af_opt": True,
                    "N_MS": 10000,
                    "N_S":2000,
                    "N_LS": 1000,
                    "k": 10,
                "reward_transformation": "none",
            }

            # register gym environment
            if env_spec["env_id"] in registry.env_specs:
                del registry.env_specs[env_spec["env_id"]]
            register(
                id=env_spec["env_id"],
                entry_point="fsaf.environment.function_gym:FSAF",
                max_episode_steps=env_spec["T"],
                reward_threshold=None,
                kwargs=env_spec
            )

            # define evaluation run
            eval_spec = {
                "env_id": env_spec["env_id"],
                "env_seed_offset": 100,
                "policy": af,
                "logpath": shot_path,
                "load_iter": load_iter,
                "deterministic": det,
                "policy_specs": policy_specs,
                "savepath": savepath,
                "n_workers": n_workers,
                "n_episodes": n_episodes,
                "T": env_spec["T"],
                "bmaml":True,
            }

            # perform evaluation
            print("Evaluating {} on {}...".format(af, env_spec["env_id"]))
            if li == 0 or af == "FSAF" or af == "iclr2020_MetaBO":
                eval_experiment(eval_spec)
            print("Done! Saved result in {}".format(savepath))
            print("**********************\n\n")

        # plot (plot is saved to savepath)
        print("Plotting...")
        plot_results(path=savepath, logplot=True if not env_spec["reward_transformation"] == "cumulative" else False
                        ,name="_{}update".format(li),
                        limit_y = None if not env_spec["reward_transformation"] == "cumulative" else [0,140])
        print("Done! Saved plot in {}".format(savepath))


