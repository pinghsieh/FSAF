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
import copy
from metaAdapt import metaAdapt
from MetaBO_transfer import MetaBO_transfer
from fsaf.policies.taf.generate_taf_data import generate_taf_data

import shutil
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="mix_all_card1000",
                    help="model name in log file")
parser.add_argument("-d", "--data", default="HPO_data/critical_temperature",
                    help="data name in log file")
parser.add_argument("-dl", "--dataLen", type=int, default=20,
                    help="data name in log file")
parser.add_argument("-dim", "--dimension", type=int, default=8,
                    help="data dimension")
parser.add_argument("-cuda", "--cuda",type=int,default=1,
                    help="gpu num")   
parser.add_argument("-det", "--deterministic",type=bool,default=True,
                    help="max or categorical")   
parser.add_argument("-is", "--init_sample",type=int,default=0,
                    help="gp init samples") 
parser.add_argument("-ls", "--lengthScale",type=float, nargs="+",default=[1],
                    help="length Scale")  
args = parser.parse_args()

# set evaluation parameters
shot_step = 5
afs_to_evaluate = ["FSAF", "EI", "PI","MES","GP-UCB", "iclr2020_MetaBO","TAF-ME"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "log","FSAF-GP-v0")
logpath = os.path.join(rootdir,args.model)
dim = args.dimension
kernel = "RBF"
shot_path = os.path.join(logpath,"shotHPO{}".format(args.data).format(dim))

if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda)


best_iter = 710 #get_best_iter_idx_reward(logpath,get_last_iter_idx(logpath),init_iter=100)
features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"]
env_spec = {
    "env_id": "FSAF-{}-v0".format(args.data).replace('/',''),
    "D": dim,  # FSAF is dimensionality agnostic and can be evaluated for any D
    "f_type": "HPO",
    "f_opts": {
        "adapting":True,
        "min_regret": 1e-20,
        "data":["HPO_data/{}_0.pkl".format(args.data)]},
    "features": features,
    "T": 50,
    "T_training": None,
    "n_init_samples": args.init_sample,
    "pass_X_to_pi": False,
    # will be set individually for each new function to the sampled hyperparameters
    "kernel": kernel,
    "kernel_lengthscale": args.lengthScale,
    "kernel_variance": 1,
    "noise_variance": 0.1,
    "use_prior_mean_function": False,
    "local_af_opt": False,
    "cardinality_domain": 1,
    "reward_transformation": "neg_log10",
}
env_spec_ppo = copy.deepcopy(env_spec)
env_spec_ppo["features"] = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep","budget"]
metaAdapt(best_iter=best_iter,iter=shot_step, dim=dim, kernel=kernel,shot=1,logpath=logpath,shot_path=shot_path,env_spec=env_spec)
os.makedirs(shot_path, exist_ok=True)
shutil.copy("{}/weights_{}".format(logpath,best_iter),"{}/weights_{}".format(shot_path,best_iter))
shutil.copy("{}/stats_{}".format(logpath,best_iter),"{}/stats_{}".format(shot_path,best_iter))
shutil.copy("{}/params_{}".format(logpath,best_iter),"{}/params_{}".format(shot_path,best_iter))
shutil.copy("{}/theta_{}".format(logpath,best_iter),"{}/theta_{}".format(shot_path,best_iter))
MetaBO_transfer_iter = 100
MetaBO_transfer(env_spec = env_spec_ppo,iter=MetaBO_transfer_iter)

n_workers = 1
datas = ["HPO_data/{}_{}.pkl".format(args.data,i) for i in range(1,args.dataLen)]
n_episodes = len(datas)
savepath = os.path.join(shot_path, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%Sdet={}".format(args.deterministic)))

test_iters = [0,5]

for li in test_iters:
    # evaluate all afs
    for af in afs_to_evaluate:
        # set af-specific parameters
        if af == "FSAF":
            features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"]#, "timestep","budget"]  # dimensionality agnostic
            pass_X_to_pi = False
            if li == 0:
                load_iter = best_iter
            else:
                load_iter = li-1 
            T_training = None
            deterministic = args.deterministic
            policy_specs = {}  # will be loaded from the logfiles
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
            generate_taf_data(M=5,f_type=args.data,ls=args.lengthScale,D=dim)
            T_training = None
            features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
            pass_X_to_pi = True
            load_iter = None  # does only apply for MetaBO
            deterministic = None  # does only apply for MetaBO
            policy_specs = {"TAF_datafile": os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "policies", "taf", "taf_{}.pkl".format(args.data))}
        elif af == "MetaBO_original":
            features = ["posterior_mean", "posterior_std","x" , "incumbent", "timestep_perc", "timestep","budget"]
            pass_X_to_pi = False
            T_training = None
            load_iter = get_best_iter_idx_reward("MetaBO_original/{}/MetaBO".format(env_spec["env_id"]),last_iter=1999,init_iter=0,winsow_size=10)
            deterministic = False
            policy_specs = {}  # will be loaded from the logfiles
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
            "env_id": "FSAF-{}-v0".format(args.data).replace('/',''),
            "D": dim,  # FSAF is dimensionality agnostic and can be evaluated for any D
            "f_type": "HPO",
            "f_opts": {
                # "gp":"GPytorch",
                "adapting":False,
                "min_regret": 0,
                "data":datas},
            "features": features,
            "T": 140,
            "T_training": T_training,
            "n_init_samples": args.init_sample,
            "pass_X_to_pi": pass_X_to_pi,
            # will be set individually for each new function to the sampled hyperparameters
            "kernel": kernel,
            "kernel_lengthscale": args.lengthScale,
            "kernel_variance": 1,
            "noise_variance": 0.1,
            "use_prior_mean_function": False,
            "local_af_opt": False,
            "cardinality_domain": 1,
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
            "deterministic": deterministic,
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
        plot_results(path=savepath, logplot=True,name="_{}update".format(li))
        print("Done! Saved plot in {}".format(savepath))