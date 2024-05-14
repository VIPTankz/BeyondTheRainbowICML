import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PERNoAlpha import UniformReplayBuffer, PrioritizedReplayBuffer, prep_observation_for_qnet
from memory import ReplayMemory
from ExperienceReplay import RegularReplayMemory
import numpy as np
from collections import deque
import pickle
import matplotlib.pyplot as plt
#from torchsummary import summary
import math
from networks import ImpalaCNNLarge, ImpalaCNNLargeIQN, NatureIQN
import networks
import traceback

class EpsilonGreedy():
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)


class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=256
                 , rr=1, maxpool_size=6, lr=1e-4, ema=False, trust_regions=False, target_replace=500, ema_tau=0.001,
                 noisy=True, spectral=True, munch=True, iqn=True, double=False, dueling=True, impala=True, discount=0.997,
                 adamw=False, ede=False, sqrt=False, discount_anneal=False, lr_decay=False, per=True, taus=8,
                 model_size=2, linear_size=512, spectral_lin=False, ncos=64, rainbow=False, maxpool=True):

        if rainbow:
            lr = 6.25e-5
            spectral = False
            munch = False
            iqn = False
            c51 = True
            double = True
            dueling = True
            impala = False
            discount = 0.99
            adamw = False
            per = True
            noisy = True
            linear_size = 512
            self.per_alpha = 0.4
        else:
            self.per_alpha = 0.2
            c51 = False

        self.procgen = True if input_dims[1] == 64 else False

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.loading_checkpoint = False

        if not self.loading_checkpoint:
            self.per_beta = 0.4
        else:
            self.per_beta = 0.7

        self.replay_ratio = int(rr) if rr > 0.99 else float(rr)
        self.total_frames = total_frames
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 8000
        else:
            self.min_sampling_size = 200000

        self.lr = lr

        self.total_grad_steps = (self.total_frames - self.min_sampling_size) / (self.num_envs / self.replay_ratio)

        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.lr_decay = lr_decay
        if self.lr_decay:
            self.lambda_lr = lambda frame: max(1.0 - frame / self.total_grad_steps, 0)

        self.chkpt_dir = ""

        # IMPORTANT params, check these

        self.n = 3
        if discount_anneal:
            self.discount_anneal = True
            self.gamma = 0.97
            self.final_gamma = 0.997
            self.annealing_period = self.total_grad_steps // 2  # first half of training
            self.gamma_inc = (self.final_gamma - self.gamma) / self.annealing_period
        else:
            self.gamma = discount
            self.discount_anneal = False
        self.batch_size = batch_size

        self.model_size = model_size  # Scaling of IMPALA network
        self.maxpool_size = maxpool_size

        self.spectral_norm = spectral  # rememberance of the bug that passed gpu tensor into env
        # and caused nans which somehow showed up in the PER sample function.

        self.spectral_lin = spectral_lin
        self.noisy = noisy

        self.per_splits = 1
        if self.per_splits > num_envs:
            self.per_splits = num_envs

        self.impala = impala  # non impala only implemented for iqn
        self.dueling = dueling

        # Don't use both of these, they are mutually exclusive
        self.c51 = c51
        self.iqn = iqn

        self.ncos = ncos

        self.ede = ede  # NOT FINISHED
        self.adamw = adamw
        self.sqrt = sqrt

        self.double = double  # Not implemented for IQN and Munchausen
        self.maxpool = maxpool
        self.munchausen = munch

        if self.munchausen:
            self.entropy_tau = 0.03
            self.lo = -1
            self.alpha = 0.9

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = 1048576

        # target_net, ema, trust_region
        if ema:
            self.stabiliser = "ema"
        elif trust_regions:
            self.stabiliser = "trust_regions"
        else:
            self.stabiliser = "target"

        if self.stabiliser == "ema":
            self.soft_updates = True
        else:
            self.soft_updates = False

        # NOT IMPLEMENTED
        if self.stabiliser == "trust_regions":
            self.trust_regions = True
            self.running_std = -999
        else:
            self.trust_regions = False

        self.soft_update_tau = ema_tau  # 0.001 for non-sample-eff
        self.replace_target_cnt = target_replace  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio

        self.tr_alpha = 1
        self.tr_period = 1500

        self.loss_type = "huber"  # NOT IMPLEMENTED

        if self.iqn:
            self.num_tau = taus

        if self.loading_checkpoint:
            self.per_beta = 0.8
            self.min_sampling_size = 300000

        #c51
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51

        if not self.loading_checkpoint and not self.testing:
            self.eps_start = 1.0
            # divided by 4 is due to frameskip
            self.eps_steps = 2000000
            self.eps_final = 0.01
        else:
            self.eps_start = 0.01
            self.eps_steps = 250000
            self.eps_final = 0.01

        self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)

        self.per = per

        self.outputs = {}
        self.dormant_tau = 0.025

        self.linear_size = linear_size

        # code for no alpha PER - appears to perform slightly worse
        # self.use_amp = False
        # if self.per:
        #     self.buffer = PrioritizedReplayBuffer(self.min_sampling_size, self.max_mem_size, self.gamma, self.n,
        #                                           self.num_envs, use_amp=self.use_amp)
        # else:
        #     self.buffer = UniformReplayBuffer(self.min_sampling_size, self.max_mem_size, self.gamma, self.n,
        #                                       self.num_envs, use_amp=self.use_amp)

        self.memories = []
        if self.per:
            for i in range(num_envs):
                self.memories.append(ReplayMemory(self.max_mem_size // num_envs, self.n, self.gamma, device, alpha=self.per_alpha, beta=self.per_beta, procgen=self.procgen))
        else:
            for i in range(num_envs):
                self.memories.append(RegularReplayMemory(self.max_mem_size // num_envs, self.n, self.gamma, device))

        if self.impala:
            if not self.iqn:
                self.net = ImpalaCNNLarge(self.input_dims[0], self.n_actions,spectral=self.spectral_norm, device=self.device,
                                             noisy=self.noisy, maxpool=self.maxpool, model_size=self.model_size, maxpool_size=self.maxpool_size,
                                          linear_size=self.linear_size)

                self.tgt_net = ImpalaCNNLarge(self.input_dims[0], self.n_actions,spectral=self.spectral_norm, device=self.device,
                                             noisy=self.noisy, maxpool=self.maxpool, model_size=self.model_size, maxpool_size=self.maxpool_size,
                                              linear_size=self.linear_size)
            else:
                # This is the BTR Network
                self.net = ImpalaCNNLargeIQN(self.input_dims[0], self.n_actions,spectral=self.spectral_norm,
                                                 device=self.device, noisy=self.noisy, maxpool=self.maxpool,
                                                 model_size=self.model_size, num_tau=self.num_tau,
                                                 maxpool_size=self.maxpool_size, dueling=dueling,
                                                 linear_size=self.linear_size, spectral_lin=spectral_lin, ncos=self.ncos,)

                self.tgt_net = ImpalaCNNLargeIQN(self.input_dims[0], self.n_actions,spectral=self.spectral_norm,
                                                 device=self.device, noisy=self.noisy, maxpool=self.maxpool,
                                                 model_size=self.model_size, num_tau=self.num_tau,
                                                 maxpool_size=self.maxpool_size, dueling=dueling,
                                                 linear_size=self.linear_size, spectral_lin=spectral_lin, ncos=self.ncos)


        else:
            self.net = NatureIQN(self.input_dims[0], self.n_actions, device=self.device,
                                     noisy=self.noisy, num_tau=self.num_tau, linear_size=self.linear_size)

            self.tgt_net = NatureIQN(self.input_dims[0], self.n_actions, device=self.device,
                                     noisy=self.noisy, num_tau=self.num_tau, linear_size=self.linear_size)

        if self.adamw:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size,
                                         weight_decay=1e-4)  # weight decay taken from museli
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015

        self.net.train()
        #if self.noisy:
        #self.tgt_net.train()

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        if self.lr_decay:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_lr)

        self.env_steps = 0
        self.grad_steps = 0

        self.replay_ratio_cnt = 0
        self.eval_mode = False

        if self.loading_checkpoint:
            self.load_models()

    def calculate_parameter_norms(self, norm_type=2):
        self.test_net.load_state_dict(self.net.state_dict())
        # Dictionary to store the norms
        norms = {}
        # Iterate through all named parameters
        for name, param in self.test_net.named_parameters():
            if 'weight' in name:
                # Calculate the norm of the parameter
                norm = torch.norm(param, p=norm_type).item()  # .item() converts a one-element tensor to a scalar
                # Store the norm in the dictionary
                norms[name] = norm

        norms_tot = 0
        count = 0
        for key, value in norms.items():
            count += 1
            norms_tot += value

        norms_tot /= count

        return norms_tot

    def get_grad_steps(self):
        return self.grad_steps

    def set_eval_mode(self):
        self.disable_noise(self.net)
        self.net.eval()
        self.tgt_net.eval()
        self.eval_mode = True

    def set_train_mode(self):
        self.net.train()
        self.tgt_net.train()
        self.eval_mode = False

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation):
        with T.no_grad():
            if self.noisy and not self.eval_mode:
                self.reset_noise(self.net)

            #state = prep_observation_for_qnet(torch.from_numpy(np.stack(observation)), self.use_amp)
            state = T.tensor(observation, dtype=T.float).to(self.net.device)
            qvals = self.net.qvals(state, advantages_only=True)

            x = T.argmax(qvals, dim=1).cpu()
            # this should contain (num_envs) different actions

            if self.eval_mode:
                for i in range(len(observation)):
                    if np.random.random() > 0.99:
                        x[i] = np.random.choice(self.action_space)

            elif self.env_steps < self.min_sampling_size or not self.noisy or self.env_steps < self.total_frames / 2:
                for i in range(len(observation)):
                    action = self.epsilon.choose_action()
                    if action is not None:
                        x[i] = action

            return x

    def store_transition(self, state, action, reward, done, stream, prio=None):
        if prio is None:
            self.memories[stream].append(torch.from_numpy(state), action, reward, done)
        else:
            self.memories[stream].append(torch.from_numpy(state), action, reward, done, 0)

        #self.buffer.put(state, action, reward, done, j=stream)
        self.epsilon.update_eps()
        self.env_steps += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)


    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        with torch.no_grad():
            for target_param, local_param in zip(self.tgt_net.parameters(), self.net.parameters()):
                target_param.data.copy_(self.soft_update_tau*local_param.data + (1.0-self.soft_update_tau)*target_param.data)

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % (int(1 / self.replay_ratio))
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def sample(self):
        if self.per:
            # get total priority from each tree
            buffer_totals = np.empty(self.num_envs, dtype=float)
            for i in range(self.num_envs):
                buffer_totals[i] = self.memories[i].transitions.total()

            # create probability distribution based on prios
            buffer_dist = buffer_totals / buffer_totals.sum()

            if self.per_splits > 1:
                mem = 0
                mems = np.random.choice(self.num_envs, self.per_splits, replace=False, p=buffer_dist)

                idxs, states, actions, rewards, next_states, dones, weights = self.memories[mems[0]].sample(
                    self.batch_size // self.per_splits)

                for i in range(self.per_splits - 1):

                    idxsN, statesN, actionsN, rewardsN, next_statesN, donesN, weightsN = self.memories[mems[i + 1]].sample(
                        self.batch_size // self.per_splits)

                    idxs = np.concatenate((idxs, idxsN))
                    states = torch.cat((states, statesN))
                    actions = torch.cat((actions, actionsN))
                    rewards = torch.cat((rewards, rewardsN))
                    next_states = torch.cat((next_states, next_statesN))
                    dones = torch.cat((dones, donesN))
                    weights = torch.cat((weights, weightsN))

            else:
                mems = 0
                # this is a slight simplification of PER, but runs MUCH faster with very little memory
                mem = np.random.choice(self.num_envs, self.per_splits, replace=False, p=buffer_dist)[0]
                idxs, states, actions, rewards, next_states, dones, weights = self.memories[mem].sample(
                    self.batch_size)

            dones = dones.squeeze()

        else:
            weights = 0
            idxs = 0
            mem = 0
            mems = 0
            # this gets how many we should sample from each experience replay
            to_sample_from_each = generate_random_sum_array(len(self.memories), self.batch_size)

            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for i in range(len(self.memories)):
                if to_sample_from_each[i] > 0:
                    statesN, actionsN, rewardsN, next_statesN, donesN = self.memories[i].sample(to_sample_from_each[i])
                    states.append(statesN)
                    actions.append(actionsN)
                    rewards.append(rewardsN)
                    next_states.append(next_statesN)
                    dones.append(donesN)

            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
            rewards = torch.cat(rewards, dim=0)
            next_states = torch.cat(next_states, dim=0)
            dones = torch.cat(dones, dim=0)

            dones = dones.squeeze()

        return states, rewards, actions, next_states, dones, weights, idxs, mems, mem

    def learn_call(self):

        if self.env_steps < self.min_sampling_size:
            return

        if self.per:
            for i in range(self.num_envs):
                self.memories[i].priority_weight = min(self.memories[i].priority_weight + self.priority_weight_increase, 1)

        #self.per_beta = min(1, self.per_beta + self.priority_weight_increase)

        if self.discount_anneal:
            self.gamma = min(self.gamma + self.gamma_inc, self.final_gamma)
            for i in self.memories:
                i.discount = self.gamma

        if self.noisy:
            self.reset_noise(self.tgt_net)

        if not self.soft_updates:
            if self.trust_regions:
                if self.grad_steps % self.tr_period == 0:
                    self.replace_target_network()
            else:
                if self.grad_steps % self.replace_target_cnt == 0:
                    self.replace_target_network()
        else:
            self.soft_update()

        states, rewards, actions, next_states, dones, weights, idxs, mems, mem = self.sample()

        # if self.per:
        #     indices, weights, (states, next_states, actions, rewards, dones) = self.buffer.sample(self.batch_size, self.per_beta)
        #     weights = torch.from_numpy(weights).cuda()
        # else:
        #     states, next_states, actions, rewards, dones = self.buffer.sample(self.batch_size)

        self.optimizer.zero_grad()

        # use this code to check your states are correct
        """
        plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()

        plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()

        plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        plt.show()
        """

        if self.c51:
            distr_v, qvals_v = self.net.both(states)
            state_action_values = distr_v[range(self.batch_size), actions.data]
            state_log_sm_v = F.log_softmax(state_action_values, dim=1)

            with torch.no_grad():
                next_distr_v, next_qvals_v = self.tgt_net.both(next_states)
                action_distr_v, action_qvals_v = self.net.both(next_states)

                next_actions_v = action_qvals_v.max(1)[1]

                next_best_distr_v = next_distr_v[range(self.batch_size), next_actions_v.data]
                next_best_distr_v = self.tgt_net.apply_softmax(next_best_distr_v)
                next_best_distr = next_best_distr_v.data.cpu()

                proj_distr = distr_projection(next_best_distr, rewards.cpu(), dones.cpu(), self.Vmin, self.Vmax, self.N_ATOMS,
                                              self.gamma ** self.n)

                proj_distr_v = proj_distr.to(self.net.device)

            loss_v = -state_log_sm_v * proj_distr_v
            if self.per:
                weights = T.squeeze(weights)
                loss_v = weights.to(self.net.device) * loss_v.sum(dim=1)

            loss = loss_v.mean()

        elif not self.iqn and not self.c51 and not self.munchausen:  # non distributional

            indices = np.arange(self.batch_size)

            q_pred = self.net.forward(states)
            q_pred = q_pred[indices, actions]

            with torch.no_grad():
                q_targets = self.tgt_net.forward(next_states)
                if self.double:
                    q_actions = self.net.forward(next_states)
                else:
                    q_actions = q_targets.clone().detach()

                max_actions = T.argmax(q_actions, dim=1)
                q_targets[dones] = 0.0

                q_target = rewards + (self.gamma ** self.n) * q_targets[indices, max_actions]

            # loss_v should be absolute error for PER
            td_error = q_target - q_pred
            loss_v = torch.abs(td_error)

            if self.per:
                loss_squared = (td_error.pow(2) * weights.to(self.net.device))
            else:
                loss_squared = td_error.pow(2)

            loss = loss_squared.mean().to(self.net.device)

        elif not self.iqn and not self.c51 and self.munchausen:  # non-distributional but with munchausen

            with torch.no_grad():

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                # if self.per:
                #     weights = weights.unsqueeze(1)

                Q_targets_next = self.tgt_net.forward(next_states)

                logsum = torch.logsumexp((Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)

                tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum

                # target policy
                pi_target = F.softmax(Q_targets_next / self.entropy_tau, dim=1)
                #Q_target = (self.gamma * (pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(1)).unsqueeze(-1)
                Q_target = (self.gamma ** self.n * (
                            pi_target * (Q_targets_next - tau_log_pi_next) * (~dones)).sum(1)).unsqueeze(1)

                # calculate munchausen addon with logsum trick
                q_k_targets = self.tgt_net(states)
                v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
                logsum = torch.logsumexp((q_k_targets - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)
                log_pi = q_k_targets - v_k_target - self.entropy_tau * logsum
                munchausen_addon = log_pi.gather(1, actions)

                # calc munchausen reward:
                munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0))

                Q_targets = munchausen_reward + Q_target

            q_k = self.net(states)
            Q_expected = q_k.gather(1, actions)

            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).squeeze()

            if self.per:
                loss_squared = (td_error.pow(2) * weights.to(self.net.device))
            else:
                loss_squared = td_error.pow(2)

            loss = loss_squared.mean().to(self.net.device)

        elif self.iqn and not self.munchausen:

            with torch.no_grad():

                if self.trust_regions:
                    Q_targets_next, _ = self.net(next_states)
                else:
                    Q_targets_next, _ = self.tgt_net(next_states)

                if self.double: #this may be wrong - seems to perform better without. Could just be chance though
                    indices = np.arange(self.batch_size)
                    q_actions = self.net.qvals(next_states)
                    max_actions = T.argmax(q_actions, dim=1)
                    Q_targets_next = Q_targets_next[indices, :, max_actions].detach().unsqueeze(1)
                else:
                    Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                if self.per:
                    weights = weights.unsqueeze(1)

                # Compute Q targets for current states
                Q_targets = rewards.unsqueeze(-1) + (
                            self.gamma ** self.n * Q_targets_next * (~dones.unsqueeze(-1)))


            # Get expected Q values from local model
            Q_expected, taus = self.net(states)
            Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

            # Quantile Huber loss
            td_error = Q_targets - Q_expected

            # get absolute losses for all taus
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            # assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"

            # calculate huber loss between prediction and target
            huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)  # note this gives all positive values

            # Multiply by the taus - this is what actually makes the quantiles, and also applies the sign
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            # sum the losses
            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # keepdim=True if using PER

            if self.per:
                loss = loss * weights.to(self.net.device)

            if self.trust_regions:
                loss = self.calculate_trust_regions(loss, loss_v, states, actions, Q_expected, Q_targets)

            loss = loss.mean()

        elif self.iqn and self.munchausen:
            with torch.no_grad():
                Q_targets_next, _ = self.tgt_net(next_states)

                # (batch, num_tau, actions)
                q_t_n = Q_targets_next.mean(dim=1)

                actions = actions.unsqueeze(1)
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                if self.per:
                    weights = weights.unsqueeze(1)

                # calculate log-pi
                logsum = torch.logsumexp(
                    (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
                #assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)

                pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

                Q_target = (self.gamma ** self.n * (
                            pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)

                #assert Q_target.shape == (self.batch_size, 1, self.num_tau)

                q_k_target = self.net.qvals(states)
                v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
                tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp(
                    (q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

                #assert tau_log_pik.shape == (self.batch_size, self.n_actions), "shape instead is {}".format(
                    #tau_log_pik.shape)
                munchausen_addon = tau_log_pik.gather(1, actions)

                # calc munchausen reward:
                munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
                #assert munchausen_reward.shape == (self.batch_size, 1, 1)
                # Compute Q targets for current states
                Q_targets = munchausen_reward + Q_target

            # Get expected Q values from local model
            q_k, taus = self.net(states)
            Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))
            #assert Q_expected.shape == (self.batch_size, self.num_tau, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).data
            #assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)
            quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl

            if self.per:
                loss = loss * weights.to(self.net.device)

            if self.trust_regions:
                loss = self.calculate_trust_regions(loss, loss_v, states, actions, Q_expected, Q_targets)

            loss = loss.mean()

        if self.per:
            if self.num_envs > 1 and self.per_splits > 1:
                idxs = np.split(idxs, self.per_splits)
                loss_v = torch.split(loss_v, self.batch_size // self.per_splits)
                for i in range(self.per_splits):
                    self.memories[mems[i]].update_priorities(idxs[i], loss_v[i].cpu().detach().numpy())
            else:
                self.memories[mem].update_priorities(idxs, loss_v.cpu().detach().numpy())

        # if self.per:
        #     new_priorities = np.abs(loss_v.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
        #     self.buffer.update_priorities(indices, new_priorities)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()

        if self.lr_decay:
            self.scheduler.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")


    def calculate_trust_regions(self, loss, loss_v, states, actions, Q_expected, Q_targets):
        with torch.no_grad():
            if self.running_std != -999:
                current_std = torch.std(loss_v).item()
                self.running_std += current_std

                q_k_tgt_net, taus = self.tgt_net(states)
                target_network_pred = q_k_tgt_net.gather(2,
                                                         actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))

                # get average across quantiles
                target_pred_mean = target_network_pred.mean(dim=1)
                Q_expected_mean = Q_expected.mean(dim=1)

                # q_targets has shape (bs, 1, num_taus), so need to squeeze
                Q_targets_mean = Q_targets.squeeze().mean(dim=1).unsqueeze(1)

                #  sigma_j calculations
                sigma_j = self.running_std / self.grad_steps

                sigma_j = max(sigma_j, current_std)
                sigma_j = max(sigma_j, 0.01)

                # These all need shape checking
                outside_region = torch.abs(Q_expected_mean - target_pred_mean) > \
                                 self.tr_alpha * sigma_j

                diff_sign = torch.sign(Q_expected_mean - target_pred_mean) != \
                            torch.sign(Q_expected_mean - Q_targets_mean)

                # create mask if conditions are true
                mask = torch.logical_and(outside_region, diff_sign)
                loss[mask] = 0

                # Some Testing Code
                """
                if np.random.random() > 0.995:
                    print("Mask")
                    print(mask)

                    # mask out losses
                    loss[mask] = 0
                    print(loss)

                    print(Q_expected_mean)

                    x = input(";lol")
                """
                return loss

            else:
                self.running_std = torch.std(loss_v).detach().cpu()
                return loss


def calculate_huber_loss(td_errors, k=1.0,taus=8):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], taus, taus), "huber loss has wrong shape"
    return loss

def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = T.zeros((batch_size, n_atoms), dtype=T.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).type(T.int64)
        u = np.ceil(b_j).type(T.int64)
        eq_mask = u == l
        eq_dones = T.clone(dones)
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = T.clone(dones)
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr


def generate_random_sum_array(length, total):
    # Create an array of zeros
    arr = np.zeros(length, dtype=int)

    # Randomly distribute 'total' across the array
    indices = np.random.choice(np.arange(length), size=total, replace=True)
    for idx in indices:
        arr[idx] += 1  # Increment element at randomly chosen index

    # Shuffle the array to randomize the distribution
    np.random.shuffle(arr)

    return arr