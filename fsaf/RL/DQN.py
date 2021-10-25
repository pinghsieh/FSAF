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

os.environ["OMP_NUM_THREADS"] = "1"  # on some machines this is needed to restrict torch to one core
CUDA_LAUNCH_BLOCKING="1"
import random
import torch
import torch.optim
import time
import numpy as np
import gym
import os
import pickle as pkl
import copy
import json
from datetime import datetime
import collections
from fsaf.RL.batchrecorder import BatchRecorder
from fsaf.RL.util import get_best_iter_idx
from fsaf.policies.policies import NeuralAF
from torch.nn.utils import parameters_to_vector
from fsaf.RL.memory import CustomPrioritizedReplayBuffer
from torch.distributions import Categorical

class DQN:
    def __init__(self, policy_fn, params, logpath, save_interval, load_path="", verbose=False):
        self.params = params
        max_iter = self.params["max_steps"]/self.params["batch_size"]
        # set up the environment (only for reading out observation and action spaces)
        self.env = gym.make(self.params["env_id"])
        self.set_all_seeds()

        # logging
        self.logpath = logpath
        if not self.params["ML"]:
            os.makedirs(logpath, exist_ok=True)
        self.save_interval = save_interval
        self.verbose = verbose

        # policies, optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

        if self.params["ML"]:
            self.pi = policy_fn(observation_space=self.env.observation_space,
                  action_space=self.env.action_space,
                  deterministic=False).to(self.device)
            self.old_pi = policy_fn(observation_space=self.env.observation_space,
                  action_space=self.env.action_space,
                  deterministic=False).to(self.device)
            if self.params["load"]:
                param = self.params["load_itr"]
                with open(os.path.join(self.params["load_path"], "weights_" + str(param)), "rb") as f:
                    print("loading weights_{}".format(param))
                    self.pi.load_state_dict(torch.load(f,map_location="cpu"))
                    self.old_pi.load_state_dict(self.pi.state_dict())
                with open(os.path.join(self.params["load_path"], "theta_" + str(param)), "rb") as f:
                    self.theta = torch.load(f,map_location="cpu")
            else:
                param = self.params["load_itr"]
                with open(os.path.join(logpath, "weights_" + str(param)), "rb") as f:
                    print("loading weights_{}".format(param))
                    self.pi.load_state_dict(torch.load(f,map_location="cpu"))
                    self.old_pi.load_state_dict(self.pi.state_dict())
                with open(os.path.join(logpath, "theta_" + str(param)), "rb") as f:
                    self.theta = torch.load(f,map_location="cpu")
            self.num_particles = len(self.theta)
            self.theta = self.theta.to(self.device).detach()

        else:
            self.pi = policy_fn(self.env.observation_space, self.env.action_space,
                                deterministic=False).to(self.device)
            self.old_pi = policy_fn(self.env.observation_space, self.env.action_space,
                                    deterministic=False).to(self.device)

            names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
            self.theta = []
            self.num_particles = self.params["num_particles"]
            

            for _ in range(self.num_particles):
                theta_flatten = []
                for key in names_weights_copy.keys():
                    if len(names_weights_copy[key].shape)>=2:
                        theta_temp = torch.empty(names_weights_copy[key].shape, device=self.device)
                        torch.nn.init.xavier_normal_(tensor=theta_temp)
                    else:
                        theta_temp = torch.zeros(names_weights_copy[key].shape, device=self.device)
                    theta_flatten.append(torch.flatten(theta_temp, start_dim=0, end_dim=-1))
                    
                self.theta.append(torch.cat(theta_flatten))
            self.theta = torch.stack(self.theta)
        self.theta.requires_grad_()
        self.optimizer = torch.optim.Adam([self.theta], lr=self.params["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                              T_max= max_iter,
                                                              eta_min=self.params["lr"]/100)
                                                              
        self.total_num_inner_loop_steps = self.params["inner_loop_steps"]
        self.using_chaser = self.params["using_chaser"]
        self.multi_step_loss_num_epochs = max_iter
        # set up the batch recorder

        buffer_size = self.params["buffer_size"]
        prior_alpha = self.params["prior_alpha"]
        self.outer_w = self.params["outer_w"] if not self.params["ML"] else 0
        self.prior_beta = self.params["prior_beta"]
        self.n_steps = self.params["n_steps"]
        self.gamma = self.params["gamma"]
        self.buffer = []
        if self.params["ML"]:
            self.buffer.append(CustomPrioritizedReplayBuffer(size=buffer_size,alpha=prior_alpha))
        else:
            assert len(self.params["lengthScale"]) == len(self.params["kernels"])
            for _ in range(int(len(self.params["lengthScale"]))):
                self.buffer.append(CustomPrioritizedReplayBuffer(size=buffer_size,alpha=prior_alpha))
                self.buffer.append(CustomPrioritizedReplayBuffer(size=buffer_size,alpha=prior_alpha))

        self.demo_prob=self.params["demo_prob"] if "demo_prob" in self.params else 0

        self.batch_recorder = BatchRecorder(size=self.params["batch_size"],
                                            env_id=self.params["env_id"],
                                            env_seeds=self.params["env_seeds"],
                                            policy_fn=policy_fn,
                                            n_workers=self.params["n_workers"],
                                            buffer= self.buffer,
                                            n_steps=self.n_steps, gamma=self.gamma,metaAdapt=self.params["ML"],
                                            demo_prob=self.demo_prob)

        self.stats = dict()
        self.stats["n_timesteps"] = 0
        self.stats["n_optsteps"] = 0
        self.stats["n_iters"] = 0
        self.stats["t_train"] = 0
        self.stats["avg_step_rews"] = np.array([])
        self.stats["avg_init_rews"] = np.array([])
        self.stats["avg_term_rews"] = np.array([])
        self.stats["avg_ep_rews"] = np.array([])
        self.stats["losses"] = np.array([])
        self.stats["step15_reward"] = np.array([])
        self.stats["step20_reward"] = np.array([])
        self.stats["step30_reward"] = np.array([])
        self.sampling_cnt = 0
        self.avg_step_reward = 0
        self.avg_initial_reward = 0
        self.avg_terminal_reward = 0
        self.avg_ep_reward = 0
        self.step15_reward = 0
        self.step20_reward = 0
        self.step30_reward = 0
            

        self.t_batch = None

        self.rew_buffer = collections.deque(maxlen=50)

        self.write_overview_logfile()
    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name,param in self.pi.named_parameters():
            if param.requires_grad:
                yield param
    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                param_dict[name] = param.to(device=self.device)

        return param_dict
    def set_all_seeds(self):
        self.rng = np.random.RandomState(seed=self.params["seed"])
        np.random.seed(self.params["seed"])
        random.seed(self.params["seed"])
        torch.manual_seed(self.params["seed"])
        torch.cuda.manual_seed_all(self.params["seed"])

    def write_overview_logfile(self):
        s = ""
        s += "********* OVERVIEW OF RUN *********\n"
        s += "Logpath        : {}\n".format(self.logpath)
        s += "Logfile created: {}\n".format(datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
        s += "Environment-ID:  {}\n".format(self.params["env_id"])
        s += "Environment-kwargs:\n"
        s += json.dumps(self.env.unwrapped.kwargs, indent=2)
        s += "\n"
        s += "DQN-parameters:\n"
        s += json.dumps(self.params, indent=2)
        s += "\n"
        s += "Batchrecorder:\n"
        s += json.dumps(self.batch_recorder.overview_dict(), indent=2)
        fname = "000_overview.txt"
        with open(os.path.join(self.logpath, fname), "w") as f:
            print(s, file=f)
        if not self.verbose:
            print(s)

    def optimize_on_batch(self,index):
        '''
        sample transition from replay and get loss
        index:replay buffer index
        return:loss value for backward
        '''
        now = time.time()
        usingDemo = np.random.rand() > 1-self.demo_prob
        with torch.no_grad():
            while(self.params["batch_size"] > len(self.buffer[index])):
                self.sampling_data(index=index)
            states_ori, actions_ori, rewards_ori, next_states_ori, dones_ori, weights_ori, idxes = self.buffer[index].sample(self.params["batch_size"], self.prior_beta)

            states = torch.FloatTensor(states_ori).to(self.device)
            actions = torch.LongTensor(actions_ori).to(self.device)
            rewards = torch.FloatTensor(rewards_ori).to(self.device)
            next_states = torch.FloatTensor(next_states_ori).to(self.device)
            dones = torch.FloatTensor(dones_ori).to(self.device)
            weights = torch.FloatTensor(weights_ori).to(self.device)


            q_values, _ = self.pi(states)
            next_q_values, _ = self.pi(next_states)
            tgt_next_q_values, _ = self.old_pi(next_states)

            q_a_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_a_values = tgt_next_q_values.gather(1, next_actions).squeeze(1)
            expected_q_a_values = rewards + (self.gamma ** self.n_steps) * next_q_a_values * (1 - dones)

            td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
            prios = (0.9*torch.max(td_error)+0.1*td_error + 1e-6).cpu().numpy()
            
            self.buffer[index].update_priorities(idxes, prios)
        
        
        if(usingDemo and not self.params["ML"]):
            # print("is demo")
            with torch.no_grad():
                states_demo, actions_demo, rewards_demo, next_states_demo, dones_demo, weights_demo, idxes_demo = self.buffer[index+1].sample(self.params["batch_size"], self.prior_beta)

                states = np.concatenate((states_ori,states_demo))
                actions = np.concatenate((actions_ori, actions_demo))
                rewards = np.concatenate((rewards_ori,rewards_demo))
                next_states = np.concatenate((next_states_ori,next_states_demo))
                dones = np.concatenate((dones_ori,dones_demo))
                weights = np.concatenate((weights_ori,weights_demo))


                states_demo = torch.FloatTensor(states_demo).to(self.device)
                actions_demo = torch.LongTensor(actions_demo).to(self.device)
                rewards_demo = torch.FloatTensor(rewards_demo).to(self.device)
                next_states_demo = torch.FloatTensor(next_states_demo).to(self.device)
                dones_demo = torch.FloatTensor(dones_demo).to(self.device)
                weights_demo = torch.FloatTensor(weights_demo).to(self.device)

                q_values_demo, _ = self.pi(states_demo)
                next_q_values_demo, _ = self.pi(next_states_demo)
                tgt_next_q_values_demo, _ = self.old_pi(next_states_demo)

                q_a_values_demo = q_values_demo.gather(1, actions_demo.unsqueeze(1)).squeeze(1)
                next_actions_demo = next_q_values_demo.max(1)[1].unsqueeze(1)
                next_q_a_values_demo = tgt_next_q_values_demo.gather(1, next_actions_demo).squeeze(1)
                expected_q_a_values_demo = rewards_demo + (self.gamma ** self.n_steps) * next_q_a_values_demo * (1 - dones_demo)

                td_error_demo = torch.abs(expected_q_a_values_demo.detach() - q_a_values_demo)
                prios_demo = (0.9*torch.max(td_error_demo)+0.1*td_error_demo + 1e-6).data.cpu().numpy()
                self.buffer[index+1].update_priorities(idxes_demo, prios_demo)

                

                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.FloatTensor(dones).to(self.device)
                weights = torch.FloatTensor(weights).to(self.device)
        

        q_values, _ = self.pi(states)
        next_q_values, _ = self.pi(next_states)
        tgt_next_q_values, _ = self.old_pi(next_states)

        q_a_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_q_a_values = tgt_next_q_values.gather(1, next_actions).squeeze(1)
        expected_q_a_values = rewards + (self.gamma ** self.n_steps) * next_q_a_values * (1 - dones)

        td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
        demo_actions = (self.pi.evalDemo(states)).to(self.device)
        dist = Categorical(logits=q_values)
        loss = -torch.sum(dist.log_prob(demo_actions)) + (torch.where(td_error < 1, 0.5 * td_error ** 2, td_error - 0.5)).mean()

        t_optim = time.time() - now

        return t_optim, loss



    def sampling_data(self,index,record = False):
        '''
        call each worker to sample data
        index:replay buffer index
        record:whether to record the log
        '''
        t_batch = self.batch_recorder.record_batch(gamma=self.params["gamma"],index=index)
        self.stats["t_train"] += t_batch
        batch_stats = self.batch_recorder.get_batch_stats()
        self.sampling_cnt += 1
        self.avg_step_reward += batch_stats["avg_step_reward"]
        self.avg_initial_reward += batch_stats["avg_initial_reward"]
        self.avg_terminal_reward += batch_stats["avg_terminal_reward"]
        self.step15_reward += batch_stats["step15_reward"]
        self.step20_reward += batch_stats["step20_reward"]
        self.step30_reward += batch_stats["step30_reward"]

        self.avg_ep_reward += batch_stats["avg_ep_reward"]
        if record:
            self.stats["batch_stats"] = batch_stats
            self.stats["avg_step_rews"] = np.append(self.stats["avg_step_rews"], self.avg_step_reward/self.sampling_cnt)
            self.stats["avg_init_rews"] = np.append(self.stats["avg_init_rews"], self.avg_initial_reward/self.sampling_cnt)
            self.stats["avg_term_rews"] = np.append(self.stats["avg_term_rews"], self.avg_terminal_reward/self.sampling_cnt)
            self.stats["avg_ep_rews"] = np.append(self.stats["avg_ep_rews"], self.avg_ep_reward/self.sampling_cnt)

            self.stats["step15_reward"] = np.append(self.stats["step15_reward"], self.step15_reward/self.sampling_cnt)
            self.stats["step20_reward"] = np.append(self.stats["step20_reward"], self.step20_reward/self.sampling_cnt)
            self.stats["step30_reward"] = np.append(self.stats["step30_reward"], self.step30_reward/self.sampling_cnt)

            self.stats["perc"] = 100 * self.stats["n_timesteps"] / self.params["max_steps"]
            batch_stats["t_batch"] = t_batch
            batch_stats["sps"] = batch_stats["size"] / batch_stats["t_batch"]
            self.add_iter_log_str(batch_stats=batch_stats)
            self.sampling_cnt = 0
            self.avg_step_reward = 0
            self.avg_initial_reward = 0
            self.avg_terminal_reward = 0
            self.avg_ep_reward = 0
            self.step15_reward = 0
            self.step20_reward = 0
            self.step30_reward = 0


    def kernel(self,particle_tensor):
        '''
        compute kernel covariance
        particle_tensor:all particles
        retrun:covariance matrix and kernel gradient
        '''

        e_dists = torch.nn.functional.pdist(input=particle_tensor, p=2).to(self.device)
        d_matrix = torch.zeros((self.num_particles, self.num_particles), device=self.device)

        triu_indices = torch.triu_indices(row=self.num_particles, col=self.num_particles, offset=1)
        d_matrix[triu_indices[0], triu_indices[1]] = e_dists

        d_matrix = torch.transpose(d_matrix, dim0=0, dim1=1)
        d_matrix[triu_indices[0], triu_indices[1]] = e_dists

        mean_dist = torch.mean(e_dists)**2 
        h = mean_dist / np.log(self.num_particles)

        kernel_matrix = torch.exp(-d_matrix / (h+1))
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, particle_tensor)
        grad_kernel += particle_tensor * kernel_sum
        grad_kernel /= h
        if self.num_particles == 1:
            kernel_matrix = torch.eye(1,device=self.device)
            grad_kernel = 0
            h = 0
        return kernel_matrix, grad_kernel, h

    def train(self):
        '''
        main train
        inner and outer loop
        '''
        def dict2tensor(dict_obj):
            d2tensor = []
            for key in dict_obj.keys():
                tensor_temp = torch.flatten(dict_obj[key], start_dim=0, end_dim=-1)
                d2tensor.append(tensor_temp)
            d2tensor = torch.cat(d2tensor)
            return d2tensor
        def get_weights_target_net(w_generated, row_id, w_target_shape):
            w = {}
            if type(w_generated) is torch.Tensor:
                temp = 0
                for key in w_target_shape.keys():
                    w_temp = w_generated[row_id, temp:(temp + np.prod(w_target_shape[key].shape))]
                    if 'b' in key:
                        w[key] = w_temp
                    else:
                        w[key] = w_temp.view(w_target_shape[key].shape)
                    temp += np.prod(w_target_shape[key].shape)
            elif type(w_generated) is dict:
                for key in w_generated.keys():
                    w[key] = w_generated[key][row_id]

            return w
        
        if "kernels" in self.params:
            kernels =  self.params["kernels"]
            lengthScale = self.params["lengthScale"]
            task_num = len(lengthScale)
            assert len(lengthScale) == len(kernels) # check length
            tasks = []
            for i in range(len(lengthScale)):
                tasks.append([kernels[i],lengthScale[i]])
            task_size = self.params["task_size"]
        else:
            task_num = 1
            task_size = 1
        while self.stats["n_timesteps"] < self.params["max_steps"]:
            self.stats["n_timesteps"] += self.params["batch_size"]
            total_losses = []
            self.iter_log_str = ""
            if self.stats["n_optsteps"] % self.params["target_update_interval"] == 0:
                self.old_pi.load_state_dict(self.pi.state_dict())
            loss_final = 0
            if "kernels" in self.params: # only on training
                task_idx = 0
                for k,l in tasks:
                    with torch.no_grad():
                        self.batch_recorder.switch_task(kernel=k,lengthScale=l)
                        self.sampling_data(index=task_idx*2)
                    task_idx += 1
            random_task = self.rng.choice(np.arange(task_num),size=task_size,replace=False)
            
            for i,r_task in enumerate(random_task):
                if "kernels" in self.params: # only on training
                    self.batch_recorder.switch_task(kernel=tasks[r_task][0],lengthScale=tasks[r_task][1])
                    # print(tasks[r_task][0],tasks[r_task][1])
                chaser_loss = 0
                distance_NLL = []
                for particle_id in range(self.num_particles):
                    
                    names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
                    w = get_weights_target_net(w_generated=self.theta, row_id=particle_id, w_target_shape=names_weights_copy)
                    with torch.no_grad():
                        for name,p in self.pi.named_parameters():
                            p.data.copy_(w[name])
                    t_weights = self.batch_recorder.set_worker_weights(copy.deepcopy(self.pi))
                    self.sampling_data(index=r_task*2,record= self.params["ML"] and particle_id == self.num_particles-1)
                    t_optim, loss_NLL = self.optimize_on_batch(index=r_task*2)

                    loss_NLL_grads = torch.autograd.grad(outputs=loss_NLL,
                        inputs=self.pi.parameters(),
                        create_graph=False if not self.params["ML"] else False
                    )
                    names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
                    loss_NLL_gradients_dict = dict(zip(names_weights_copy.keys(), loss_NLL_grads))
                    loss_NLL_gradients = dict2tensor(dict_obj=loss_NLL_gradients_dict)
                    distance_NLL.append(loss_NLL_gradients)
                distance_NLL = torch.stack(distance_NLL)
                kernel_matrix, grad_kernel, _ = self.kernel(particle_tensor=self.theta)

                if not self.params["ML"]:
                    q = self.theta - self.params["inner_lr"] * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)/len(names_weights_copy)
                else:
                    self.theta = self.theta - self.params["inner_lr"] * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)/len(names_weights_copy)

                for num_step in range(1,self.total_num_inner_loop_steps):
                        
                    distance_NLL = []
                    for particle_id in range(self.num_particles):
                        
                        names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
                        w = get_weights_target_net(w_generated=q, row_id=particle_id, w_target_shape=names_weights_copy)
                        with torch.no_grad():
                            for name,p in self.pi.named_parameters():
                                p.data.copy_(w[name])
                        t_weights = self.batch_recorder.set_worker_weights(copy.deepcopy(self.pi))
                        self.sampling_data(index=r_task*2)
                        t_optim, loss_NLL = self.optimize_on_batch(index=r_task*2)

                        loss_NLL_grads = torch.autograd.grad(outputs=loss_NLL,
                            inputs=self.pi.parameters(),
                            create_graph=False if not self.params["ML"] else False
                        )
                        names_weights_copy = self.get_inner_loop_parameter_dict(self.pi.named_parameters())
                        loss_NLL_gradients_dict = dict(zip(names_weights_copy.keys(), loss_NLL_grads))
                        loss_NLL_gradients = dict2tensor(dict_obj=loss_NLL_gradients_dict)
                        distance_NLL.append(loss_NLL_gradients)
                    distance_NLL = torch.stack(distance_NLL)
                    kernel_matrix, grad_kernel, _ = self.kernel(particle_tensor=q)

                    q = q - self.params["inner_lr"] * (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)/len(names_weights_copy)
                
                if not self.params["ML"]:
                    t_weights = self.batch_recorder.set_worker_weights(copy.deepcopy(self.pi),theta=q.detach().cpu())
                    self.sampling_data(index=r_task*2,record=(i+1) >= task_size)
                    t_optim, loss = self.optimize_on_batch(index=r_task*2)
                    if self.using_chaser:
                        for j in range(self.num_particles):
                            w_old = get_weights_target_net(w_generated=self.theta, row_id=j, w_target_shape=names_weights_copy)
                            w_new = get_weights_target_net(w_generated=q, row_id=j, w_target_shape=names_weights_copy)
            
                            for paramsvec, paramsvec_true in zip(w_old.values(),w_new.values()):
                                vec = parameters_to_vector(paramsvec)
                                vec_true = parameters_to_vector(paramsvec_true).detach()
                                chaser_loss = chaser_loss + torch.dot((vec - vec_true),(vec - vec_true) )#.clamp(-1e3,1e3)
                    else:
                        loss_final += loss

            
            

            if not self.params["ML"]:
                self.optimizer.zero_grad()
                chaser_loss = chaser_loss / self.num_particles
                chaser_loss = chaser_loss + loss_final/task_size
                chaser_loss.backward()
                self.stats["losses"] = np.append(self.stats["losses"], chaser_loss.detach().cpu().numpy())
                torch.nn.utils.clip_grad.clip_grad_norm_(self.pi.parameters(), self.params["max_norm"])
                self.optimizer.step()
                self.scheduler.step(epoch=self.stats["n_iters"])
                self.stats["n_optsteps"] += 1
                self.optimizer.zero_grad()


            if ( self.stats["n_iters"] % self.save_interval == 0 or \
                        self.stats["n_iters"] == 0 or \
                        self.stats["n_timesteps"] >= self.params["max_steps"]):
                self.store_weights(self.theta)
                self.store_log()
                
            self.stats["t_train"] += t_optim
            self.stats["t_optim"] = t_optim
                

            self.stats["n_iters"] += 1
        self.batch_recorder.cleanup()

    def store_log(self):
        with open(os.path.join(self.logpath, "log"), "a") as f:
            print(self.iter_log_str, file=f)
        if not self.verbose:
            print(self.iter_log_str)
        with open(os.path.join(self.logpath, "stats_" + str(self.stats["n_iters"])), "wb") as f:
            pkl.dump(self.stats, f)
        with open(os.path.join(self.logpath, "params_" + str(self.stats["n_iters"])), "wb") as f:
            pkl.dump(self.params, f)

    def store_weights(self,theta):
        with open(os.path.join(self.logpath, "weights_" + str(self.stats["n_iters"])), "wb") as f:
            torch.save(self.pi.state_dict(), f)
        with open(os.path.join(self.logpath, "theta_" + str(self.stats["n_iters"])), "wb") as f:
            torch.save(theta, f)

    def add_iter_log_str(self, batch_stats):
        self.iter_log_str += "\n******************** ITERATION {:2d} ********************\n".format(
            self.stats["n_iters"])
        self.iter_log_str += " RUN STATISTICS (BEFORE OPTIMIZATION):\n"
        self.iter_log_str += "   environment          = {}\n".format(self.params["env_id"])
        self.iter_log_str += "   n_timesteps  = {:d} ({:.2f}%)\n".format(self.stats["n_timesteps"], self.stats["perc"])
        self.iter_log_str += "   n_optsteps   = {:d}\n".format(self.stats["n_optsteps"])
        self.iter_log_str += "   t_total      = {:.2f}s\n".format(self.stats["t_train"])
        self.iter_log_str += " BATCH STATISTICS (BEFORE OPTIMIZATION):\n"
        self.iter_log_str += "   n_workers    = {:d}\n".format(self.batch_recorder.n_workers)
        self.iter_log_str += "   worker_seeds = {}\n".format(self.batch_recorder.env_seeds)
        self.iter_log_str += "   size         = {:d}\n".format(batch_stats["size"])
        self.iter_log_str += "   avg_step_rew = {:.4g}\n".format(batch_stats["avg_step_reward"])
        self.iter_log_str += "   avg_init_rew = {:.4g}\n".format(batch_stats["avg_initial_reward"])
        self.iter_log_str += "   avg_term_rew = {:.4g}\n".format(batch_stats["avg_terminal_reward"])

        self.iter_log_str += "   step15_reward = {:.4g}\n".format(batch_stats["step15_reward"])
        self.iter_log_str += "   step20_reward = {:.4g}\n".format(batch_stats["step20_reward"])
        self.iter_log_str += "   step30_reward = {:.4g}\n".format(batch_stats["step30_reward"])

        self.iter_log_str += "   avg_ep_rew   = {:.4g}\n".format(batch_stats["avg_ep_reward"])
        self.iter_log_str += "   n_new        = {:d}\n".format(batch_stats["n_new"])
        self.iter_log_str += "    per_worker  = {:}\n".format(batch_stats["worker_n_news"])
        self.iter_log_str += "   avg_ep_len   = {:.2f}\n".format(batch_stats["avg_ep_len"])
        self.iter_log_str += "   t_batch      = {:.2f}s ({:.0f}sps)\n".format(batch_stats["t_batch"],
                                                                              batch_stats["sps"])


