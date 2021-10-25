# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

import os
import numpy as np
from fsaf.environment.util import scale_from_unit_square_to_domain
from fsaf.environment.objectives import *
import sobol_seq
import pickle


def generate_taf_data(M,f_type,ls,D):
    # dimension of task
    rng = np.random.RandomState(seed = 99999)

    pkl = pickle.load(open("HPO_data/{}_0.pkl".format(f_type),"rb"))

    domain = np.array(pkl["domain"])
    datalen = len(domain)
    N = 250
    input_grid = []
    input_idxs = []
    for i in range(M):
        # input_idxs.append(rng.choice(datalen,N,replace=False))#sobol_seq.i4_sobol_generate(dim_num=D, n=N)
        # input_grid.append(domain[input_idxs[i]]) 
        input_grid.append(domain[int(i*50):int(i*50+50)]) 

    # generate data
    kernel_lengthscale = ls
    kernel_variance = 1.0
    noise_variance = 8.9e-16
    use_prior_mean_function = False
    data = {"D": D,
            "M": M,
            "X": input_grid,
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [kernel_lengthscale],
            "kernel_variance": M * [kernel_variance],
            "noise_variance": M * [noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function],
            "kernel":M*["RBF"]}

    for i in range(M):
        # data["Y"][i] = np.array(pkl["accs"])[input_idxs[i]].reshape(-1,1)
        data["Y"][i] = np.array(pkl["accs"])[int(i*50):int(i*50+50)].reshape(-1,1)

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_{}.pkl".format(f_type))
    with open(datafile, "wb") as f:
        pickle.dump(data, f)


def generate_taf_data_blackbox(M,f_type,ls,D):
    # dimension of task
    rng = np.random.RandomState(seed = 99999)
    # generate grid of function parameters for source tasks
    # number of source tasks
    bound_scaling = 0.1
    bound_translation = 0.1
    fct_params_domain = []
    for i in range(D):
        fct_params_domain.append([-bound_translation, bound_translation])
    fct_params_domain.append([1 - bound_scaling, 1 + bound_scaling])
    fct_params_domain = np.array(fct_params_domain)
    fct_params_grid = rng.rand(M,D+1)#sobol_seq.i4_sobol_generate(dim_num=D+1, n=M)  # D translations, 1 scaling
    fct_params_grid = scale_from_unit_square_to_domain(X=fct_params_grid, domain=fct_params_domain)

    # generate grid of control parameters
    # number of parameter configurations
    N = 250
    input_grid = rng.rand(N,D)#sobol_seq.i4_sobol_generate(dim_num=D, n=N)

    # generate data
    kernel_lengthscale = ls
    kernel_variance = 1.0
    noise_variance = 8.9e-16
    use_prior_mean_function = False
    data = {"D": D,
            "M": M,
            "X": M * [input_grid],
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [kernel_lengthscale],
            "kernel_variance": M * [kernel_variance],
            "noise_variance": M * [noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function],
            "kernel":M*["RBF"]}

    for i, fct_params in enumerate(fct_params_grid):
        t = np.array(fct_params[:-1])
        s = fct_params[D]
        if f_type == "ackley":
            fct_eval = ackley_var(x=input_grid, t=t, s=s)
        elif f_type == "STYBLINSKI_TANG":
            fct_eval = STYBLINSKI_TANG_var(x=input_grid, t=t, s=s)
        elif f_type == "egg":
            fct_eval = Eggholder_var(x=input_grid, t=t, s=s)
        elif f_type == "GRIEWANK":
            fct_eval = GRIEWANK_var(x=input_grid, t=t, s=s)
        elif f_type == "DIXON_PRICE":
            fct_eval = DIXON_PRICE_var(x=input_grid, t=t, s=s)
        elif f_type == "POWELL":
            fct_eval = POWELL_var(x=input_grid, t=t, s=s)
        data["Y"][i] = fct_eval

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_{}.pkl".format(f_type))
    with open(datafile, "wb") as f:
        pickle.dump(data, f)


def generate_taf_data_gps(M,kernel,ls,D,periods=None):
    rng = np.random.RandomState(seed=99999)
    seed = rng.randint(100000)
    n_features = 500
    lengthscale = rng.uniform(low=ls[0],
                                    high=ls[1])
    

    ssgp = SparseSpectrumGP(seed=seed, input_dim=D, noise_var=0.1, 
                            length_scale=lengthscale,
                            signal_var=1, n_features=n_features, kernel=kernel,periods=periods)
    x_train = np.array([]).reshape(0, D)
    y_train = np.array([]).reshape(0, 1)
    ssgp.train(x_train, y_train, n_samples=1)
    f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1)

    # load gp-hyperparameters
    kernel_lengthscale = lengthscale if not kernel == "SM" else lengthscale/4
    kernel_variance = 1
    noise_variance = 8.9e-16

    N = 250
    input_grid = rng.rand(N,D)#sobol_seq.i4_sobol_generate(dim_num=D, n=N)
    y_vec = f(input_grid)

    f = lambda x: ssgp.sample_posterior_handle(x).reshape(-1, 1)
    kernel_lengthscale = lengthscale
    kernel_variance = 1.0
    noise_variance = 8.9e-16
    use_prior_mean_function = False
    data = {"D": D,
            "M": M,
            "X": M * [input_grid],
            "Y": M * [None],  # is computed below
            "kernel_lengthscale": M * [kernel_lengthscale],
            "kernel_variance": M * [kernel_variance],
            "noise_variance": M * [noise_variance],
            "use_prior_mean_function": M * [use_prior_mean_function],
            "kernel":kernel}

    for i in range(M):
        fct_eval = f(x=input_grid)
        data["Y"][i] = fct_eval

    this_path = os.path.dirname(os.path.realpath(__file__))
    datafile = os.path.join(this_path, "taf_{}{}D.pkl".format(kernel,D))
    with open(datafile, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    generate_taf_data(1,"hpobenchXGB",[11.870, 0.787, 6.060, 10.142, 11.142, 10.255],D=6)