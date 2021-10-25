# FewShotAF
This repository contains implementations of the paper, Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization. It includes code for running the training and evaluation all experiments in the paper.
To get our initial model,  run python train_fsaf.py.
To comparison with other afs, we implemented evaluate_fsaf_blackbox.py for global optimization benchmark functions, evaluate_fsaf_gps.py for GP kernel functions and evaluate_fsaf_hpo.py for real task and hpo.

# Quick run
Examples of FSAF running script can see in the run_all_exp.sh
We tested this code on the following versions of libraries:
- gpy 1.9.9
- pytorch 1.7.0
- numpy 1.18.1
- python 3.7.6
- scipy 1.4.1
- sympy 1.5.1
- sobol-seq 0.1.2
- gym 0.17.1
- namedlist 1.7
- matplotlib 3.1.3

# Refernece

- https://github.com/metabo-iclr2020/MetaBO

