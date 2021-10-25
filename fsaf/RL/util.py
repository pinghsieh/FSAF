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

import numpy as np
import os
import pickle as pkl


def plot_learning_curve(ax, data, smoothing_range, color, plot_only_smoothed=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data_smooth = np.array([])
    for i in range(len(data)):
        if i >= smoothing_range:
            data_smooth = np.append(data_smooth,
                                    np.mean(data[i - smoothing_range:i + 1]))
        else:
            data_smooth = np.append(data_smooth, None)

    assert data.size == data_smooth.size
    if not plot_only_smoothed:
        ax.plot(data)
    ax.plot(data_smooth, color=color, lw=4)


def get_last_iter_idx(logpath):
    # returns the index of the last iteration stored in logpath
    files = os.listdir(logpath)
    last_iter = -np.inf
    for file in files:
        if file.startswith("stats_"):
            pos = file.find("_")
            iter = int(file[pos + 1:])
            if iter > last_iter:
                last_iter = iter

    return last_iter


def get_best_iter_idx(logpath):
    # returns the index of the best iteration (w.r.t. average step reward) stored in logpath
    last_iter = get_last_iter_idx(logpath)
    best_iter = -np.inf
    best_avg_step_rew = -np.inf
    with open(os.path.join(logpath, "stats_{:d}".format(last_iter)), "rb") as f:
        stats = pkl.load(f)
    for iter, avg_step_rew in enumerate(stats["avg_step_rews"]):
        if avg_step_rew > best_avg_step_rew:
            best_avg_step_rew = avg_step_rew
            best_iter = iter
    # consistency check
    with open(os.path.join(logpath, "stats_{:d}".format(best_iter)), "rb") as f:
        best_stats = pkl.load(f)
    assert best_stats["batch_stats"]["avg_step_reward"] == best_avg_step_rew

    return best_iter

def get_best_iter_idx_meta(logpath,last_iter,init_point = 0):
    # returns the index of the best iteration (w.r.t. average step reward) stored in logpath
    best_iter = np.inf
    best_rew = np.inf
    best_iter_rew = -np.inf
    winsow_size = 10
    with open(os.path.join(logpath, "stats_{:d}".format(last_iter)), "rb") as f:
        stats = pkl.load(f)
    # print(stats)
    for iter in range(init_point,len(stats["avg_ep_rews"]),winsow_size):
        total = 0
        # print(max(stats["avg_term_rews"]))
        for j in range(winsow_size):
            if iter+j == last_iter:
                break
            total += np.abs(stats["losses"][iter+j])#+stats["avg_term_rews"][iter+j]
        total /= (j+1)
        if total < best_rew:
            best_rew = total
            best_iter = iter
    last = last_iter if best_iter+winsow_size>last_iter else best_iter+winsow_size
    return best_iter

def get_best_iter_idx_reward(logpath,last_iter,winsow_size=10,init_iter = 0):
    # returns the index of the best iteration (w.r.t. average step reward) stored in logpath
    best_iter = np.inf
    best_rew = -np.inf
    best_iter_rew = -np.inf
    with open(os.path.join(logpath, "stats_{:d}".format(last_iter)), "rb") as f:
        stats = pkl.load(f)
    # print(stats)
    for iter in range(init_iter,len(stats["avg_ep_rews"]),winsow_size):
        total = 0
        # print(max(stats["avg_term_rews"]))
        for j in range(winsow_size):
            if iter+j == last_iter:
                break
            total += np.abs(stats["avg_step_rews"][iter+j])
        total /= (j+1)
        if total > best_rew:
            best_rew = total
            best_iter = iter
    return best_iter