# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch
from matplotlib import pyplot as plt

Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class Buffer:
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
    self.max_idx = -1

  def append(self, data):
    self.data[self.index] = data  # Store data in underlying data structure

    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max_idx = min(self.max_idx + 1, self.size)

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

class RegularReplayMemory():
  def __init__(self, capacity, n, discount, device):
    self.device = device
    self.capacity = capacity
    self.history = 4
    self.discount = discount
    self.n = n
    self.t = 0  # Internal episode timestep counter
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
    self.transitions = Buffer(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
    self.avoids = np.zeros(capacity, dtype=bool)

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal, invalid=None):
    state = state[-1].to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    self.avoids[self.transitions.index] = False if invalid is None else True
    self.transitions.append((self.t, state, action, reward, not terminal))  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  # Returns the transitions with blank states where appropriate
  def _get_transitions(self, idxs):
    transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
    transitions = self.transitions.get(transition_idxs)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
      blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0
    for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
      blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
    transitions[blank_mask] = blank_trans
    return transitions

  # Returns a valid sample from each segment
  def _get_samples_from_segments(self, batch_size):
    valid = False
    while not valid:
      idxs = np.random.randint(0, self.transitions.max_idx, batch_size)
      if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and not np.any(self.avoids[idxs]):
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0
    # Retrieve all required transition data (from t - h to t + n)
    transitions = self._get_transitions(idxs)
    # Create un-discretised states and nth next states
    all_states = transitions['state']
    states = torch.tensor(all_states[:, :self.history], dtype=torch.uint8)
    next_states = torch.tensor(all_states[:, self.n:self.n + self.history], dtype=torch.uint8)
    # Discrete actions to be used as index
    actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
    R = torch.matmul(rewards, self.n_step_scaling)
    # Mask for non-terminal nth next states
    nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
    return states, actions, R, next_states, nonterminals

  def sample(self, batch_size):
    states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size)  # Get batch of valid samples
    nonterminals = nonterminals.bool()
    nonterminals = ~nonterminals
    states = states.to(torch.float32).cuda()
    next_states = next_states.to(torch.float32).cuda()
    return states, actions, returns, next_states, nonterminals

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  """def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(self.history - 1)):
      blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
    transitions[blank_mask] = blank_trans
    state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device)  # Agent will turn into batch
    self.current_idx += 1
    return state

  next = __next__  # Alias __next__ for Python 2 compatibility"""