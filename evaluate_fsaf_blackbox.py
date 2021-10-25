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
parser.add_argument("-m", "--model", 
                    help="model name in log file")
parser.add_argument("-det", "--deterministic",type=bool,default=True,
                    help="max or categorical")  
parser.add_argument("-cuda", "--cuda",type=int,default=0,
                    help="gpu num")    
args = parser.parse_args()

# set evaluation parameters
shot_step = 5
afs_to_evaluate = ["FSAF", "EI","iclr2020_MetaBO", "PI","MES","GP-UCB","TAF-ME"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "log","FSAF-GP-v0")
logpath = os.path.join(rootdir,args.model)
f_types = [
            "STYBLINSKI_TANG",
            "egg",
            "ackley",
            "DIXON_PRICE",
            "POWELL"
            ]
length_scales = [
                [0.175,0.175],
                [0.034,0.085],
                [0.07,0.018],
                [0.922, 0.820, 0.778, 0.647, 0.394, 0.355, 0.279],
                [5.582, 0.786, 4.595, 2.815, 1.068, 0.851, 3.076, 2.796, 0.860, 5.623]
                ]
dims = [2,2,2,7,10]
n_inits = [5,5,0,0,10]#[5,5,0,5,0,10]
assert len(dims) == len(length_scales) == len(f_types)
if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda)
for idx in range(len(dims)):
    shot_path = os.path.join(logpath,"shot{}D_{}".format(dims[idx],f_types[idx]))


    kernel = "RBF"

    best_iter = 710 #get_best_iter_idx_reward(logpath,get_last_iter_idx(logpath),init_iter=100)
    shot=5
    env_spec = {
        "env_id": "FSAF-{}D{}-v0".format(dims[idx],f_types[idx]),
        "D": dims[idx],
        "f_type": f_types[idx],
        "f_opts": {
                "bound_translation":0.1,
                "bound_scaling":0.1,
                "kernel": kernel,
                "min_regret": 1e-20,
                "mix_kernel": False,
                "metaTrainShot":shot},
        "features": ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"],
        "T": 50,
        "n_init_samples": n_inits[idx],
        "pass_X_to_pi": False,
        "kernel": kernel,
        "kernel_lengthscale": length_scales[idx],
        "kernel_variance": 1,
        "noise_variance": 8.9e-16,
        "use_prior_mean_function": False,
        "local_af_opt": True,
            "N_MS": 10000,
            "N_S":2000,
            "N_LS": 1000,
            "k": 10,
        "reward_transformation": "neg_log10"  # true maximum not known
    }
    env_spec_ppo = copy.deepcopy(env_spec)
    env_spec_ppo["features"] = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc", "timestep","budget"]
    metaAdapt(best_iter=best_iter,iter=shot_step, dim=dims[idx], kernel=kernel,shot=shot,logpath=logpath,shot_path=shot_path,env_spec=env_spec)
    shutil.copy("{}/weights_{}".format(logpath,best_iter),"{}/weights_{}".format(shot_path,best_iter))
    shutil.copy("{}/stats_{}".format(logpath,best_iter),"{}/stats_{}".format(shot_path,best_iter))
    shutil.copy("{}/params_{}".format(logpath,best_iter),"{}/params_{}".format(shot_path,best_iter))
    shutil.copy("{}/theta_{}".format(logpath,best_iter),"{}/theta_{}".format(shot_path,best_iter))
    MetaBO_transfer_iter = 100
    MetaBO_transfer(env_spec = env_spec_ppo,iter=MetaBO_transfer_iter)

    # exit()
    n_workers = 10
    n_episodes = 200
    savepath = os.path.join(shot_path, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%Sdet={}".format(args.deterministic)))

    test_iters = [0,1,2,3,4,5]

    for li in test_iters:
        # evaluate all afs
        for af in afs_to_evaluate:
            # set af-specific parameters
            if af == "FSAF":
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep_perc"]
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
                policy_specs = {"dim":dims[idx]}
            elif af == "TAF-ME" or af == "TAF-RANKING":
                generate_taf_data_blackbox(M=5,f_type=f_types[idx],ls=length_scales[idx],D=dims[idx])
                T_training = None
                features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
                pass_X_to_pi = True
                load_iter = None  # does only apply for MetaBO
                deterministic = None  # does only apply for MetaBO
                policy_specs = {"TAF_datafile": os.path.join(os.path.dirname(os.path.realpath(__file__)), "fsaf", "policies", "taf", "taf_{}.pkl".format(f_types[idx]))}
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
                "env_id": "FSAF-{}D{}-v0".format(dims[idx],f_types[idx]),
                "D": dims[idx],  # FSAF is dimensionality agnostic and can be evaluated for any D
                "f_type": f_types[idx],
                "f_opts": {
                        "bound_translation":0.3,
                        "bound_scaling":0.3,
                        "kernel": kernel,
                        "min_regret": 0,
                        "mix_kernel": False},
                "features": features,
                "T": 80,
                "T_training": T_training,
                "n_init_samples": n_inits[idx],
                "pass_X_to_pi": pass_X_to_pi,
                # will be set individually for each new function to the sampled hyperparameters
                "kernel": kernel,
                "kernel_lengthscale": length_scales[idx],
                "kernel_variance": 1,
                "noise_variance": 8.9e-16,
                "use_prior_mean_function": False,
                "local_af_opt": True,
                "N_MS": 10000,#N_MSs[idx],
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