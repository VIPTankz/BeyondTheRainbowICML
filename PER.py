import numpy as np
from collections import deque
import torch
import time
import gymnasium as gym
import numpy
from math import sqrt


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree():
  def __init__(self, size, procgen=False):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Updates nodes values from current tree
  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  # Propagates changes up tree given tree indices
  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  # Propagates single value up tree given a tree index for efficiency
  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  # Updates values given tree indices
  def update(self, indices, values):
    self.sum_tree[indices] = values  # Set new values
    self._propagate(indices)  # Propagate values
    current_max_value = np.max(values)
    self.max = max(current_max_value, self.max)

  # Updates single value given a tree index for efficiency
  def _update_index(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate_index(index)  # Propagate value
    self.max = max(value, self.max)

  def append(self, value):
    self._update_index(self.index + self.tree_start, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of values in sum tree
  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
    # If indices correspond to leaf nodes, return them
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
    successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
    successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
    return self._retrieve(successor_indices, successor_values)

  # Searches for values in sum tree and returns values, data indices and tree indices
  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

  def total(self):
    return self.sum_tree[0]

class PER:
    def __init__(self, size, device, n, envs, gamma, alpha=0.2, beta=0.4, framestack=4, imagex=84, imagey=84, rgb=False):

        self.st = SumTree(size)
        self.data = [None for _ in range(size)]
        self.index = 0
        self.size = size

        # this is the number of frames, not the number of transitions
        # the technical size to ensure there are errors with overwritten memory in theory is very high-
        # (2*framestack - overlap) * first_states + non_first_states
        # with N=3, framestack=4, size=1M, average ep length 20, we need a total frame storage of around 1.35M
        # this however is still pretty light given it uses discrete memory. Careful when using RGB though, as we don't need as much memory
        if rgb:
            self.storage_size = int(size * 4)
        else:
            self.storage_size = int(size * 1.25)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0

        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.imagex = imagex
        self.imagey = imagey

        self.max_prio = 1

        self.framestack = framestack

        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = device

        self.last_terminal = [True for i in range(envs)]
        self.tstep_counter = [0 for i in range(envs)]

        self.n_step = n
        self.state_buffer = [[] for i in range(envs)]
        self.reward_buffer = [[] for i in range(envs)]

        if rgb:
            self.state_mem = np.zeros((self.storage_size, 3, self.imagex, self.imagey), dtype=np.uint8)
        else:
            self.state_mem = np.zeros((self.storage_size, self.imagex, self.imagey), dtype=np.uint8)
        self.action_mem = np.zeros(self.storage_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size, dtype=float)
        self.done_mem = np.zeros(self.storage_size, dtype=bool)

        # everything here is stored as ints as they are just pointers to the actual memory
        # reward contains N values. The first value contains the action. The set of N contains the pointers for both
        # the reward and dones
        self.trans_dtype = np.dtype([('state', int, self.framestack), ('n_state', int, self.framestack),
                                     ('reward', int, self.n_step)])

        self.blank_trans = (np.zeros(self.framestack, dtype=int), np.zeros(self.framestack, dtype=int),
                            np.zeros(self.n_step, dtype=int))

        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

        #self.priority_min = [float('inf') for _ in range(2 * self.size)]
        #print("Prio Size: " + str(len(self.priority_min)))

    def append(self, state, action, reward, n_state, done, stream, prio=True):

        # append to memory
        self.append_memory(state, action, reward, n_state, done, stream)

        # append to pointer
        self.append_pointer(stream, prio)

        if done:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done

    # def _set_priority_min(self, idx, priority_alpha):
    #     idx += self.size
    #     self.priority_min[idx] = priority_alpha
    #     while idx >= 2:
    #         idx //= 2
    #         self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def append_pointer(self, stream, prio):

        while len(self.state_buffer[stream]) >= self.framestack + self.n_step and len(self.reward_buffer[stream]) >= self.n_step:
            # First array in the experience
            state_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (starts after N frames)
            n_state_array = self.state_buffer[stream][self.n_step:self.n_step + self.framestack]

            # Reward array (first N rewards)
            reward_array = self.reward_buffer[stream][:self.n_step]

            #print("Added Experience: (" + str(self.point_mem_idx) + ")")
            #print((np.array(state_array, dtype=int), np.array(n_state_array, dtype=int), np.array(reward_array, dtype=int)))

            # Add the experience to the list
            self.pointer_mem[self.point_mem_idx] = (np.array(state_array, dtype=int), np.array(n_state_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            #self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))
            self.st.append(self.max_prio ** self.alpha)

            self.capacity = min(self.size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.size

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)
            self.beta = 0

    def finalize_experiences(self, stream):
        # Process remaining states and rewards at the end of an episode
        while len(self.state_buffer[stream]) >= self.framestack and len(self.reward_buffer[stream]) > 0:
            # First array in the experience
            first_array = self.state_buffer[stream][:self.framestack]
            #print(first_array)

            # Second array in the experience (Final `framestack` elements)
            second_array = self.state_buffer[stream][-self.framestack:]
            #print(second_array)

            # Reward array
            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.extend([0])

            #print(reward_array)

            # print("Added Experience: (" + str(self.point_mem_idx) + ")")
            # print((np.array(first_array, dtype=int), np.array(second_array, dtype=int),
            #                                                  np.array(reward_array, dtype=int)))
            # Add the experience
            self.pointer_mem[self.point_mem_idx] = (np.array(first_array, dtype=int), np.array(second_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            #self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))
            self.st.append(self.max_prio ** self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.size
            self.capacity = min(self.size, self.capacity + 1)

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(self, state, action, reward, n_state, done, stream):

        if self.last_terminal[stream]:
            # add full transition
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # remember n_step is not applied in this memory
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0

        else:
            # just add relevant info
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    def sample(self, batch_size):

        # get total sumtree priority
        p_total = self.st.total()

        # first use sumtree prios to get the indices
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        try:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
        except Exception as e:
            print(segment_length)
            print(segment_starts)
            print(e)
            raise Exception("Stop")

        prios, idxs, tree_idxs = self.st.find(samples)

        probs = prios / p_total

        # fetch the pointers by using indices
        pointers = self.pointer_mem[idxs]
        #print("Pointers")
        #print(pointers)

        # Extract the pointers into separate arrays
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        # get state info
        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        # reward and dones just use the same pointer. actions just use the first one
        rewards = self.reward_mem[reward_pointers]
        dones = self.done_mem[reward_pointers]
        actions = self.action_mem[action_pointers]

        # apply n_step cumulation to rewards and dones
        if self.n_step > 1:
            rewards, dones = self.compute_discounted_rewards_batch(rewards, dones)

        #prob_min = self.priority_min[1] / p_total
        #max_weight = (prob_min * self.capacity) ** (-self.beta)

        # Compute importance-sampling weights w
        weights = (self.capacity * probs) ** -self.beta

        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch

        if torch.isnan(weights).any():
            print("Nan Found is sample!")
            print(f"Prios {prios}")
            print(f"Probs {probs}")
            print(f"Weights {weights}")

        # move to pytorch GPU tensors
        states = states.to(torch.float32).to(self.device)
        n_states = n_states.to(torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        # return batch
        return tree_idxs, states, actions, rewards, n_states, dones, weights

    def compute_discounted_rewards_batch(self, rewards_batch, dones_batch):
        """
        Compute discounted rewards for a batch of rewards and dones.

        Parameters:
        rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
        dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)

        Returns:
        np.ndarray: 1D array of discounted rewards for each element in the batch
        np.ndarray: 1D array of cumulative dones (True if any done is True in the sequence)
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(batch_size)
        cumulative_dones = np.zeros(batch_size, dtype=bool)

        for i in range(batch_size):
            cumulative_discount = 1
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones

    def update_priorities(self, idxs, priorities):
        priorities = priorities + self.eps

        # for idx, priority in zip(idxs, priorities):
        #     self._set_priority_min(idx - self.size + 1, sqrt(priority))

        if np.isnan(priorities).any():
            print("NaN found in priority!")
            print(f"priorities: {priorities}")

        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(idxs, priorities ** self.alpha)

def create_experience(previous_state):
    state = previous_state[:]
    action = np.random.randint(0, 18)
    reward = np.random.choice([0, 1], p=[0.5, 0.5])
    new_frame = np.random.randint(1, 255, (image_size, image_size), dtype=np.uint8)
    state_ = np.roll(previous_state, shift=-1, axis=0)
    state_[-1] = new_frame

    done = np.random.choice([False, True], p=[0.9, 0.1])

    return state, action, reward, state_, done


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    framestack = 4
    tree = PER(8, device, 3, 2, 0.99, alpha=0.2, beta=0.4, framestack=framestack, imagex=2, imagey=2)

    image_size = 2
    batch_size = 2

    """    env = gym.wrappers.FrameStack(gym.wrappers.AtariPreprocessing(gym.make("ALE/" + "Pong" + "-v5", frameskip=1)), 4, lz4_compress=False)
    state, _ = env.reset()

    for i in range(12):
        state_, reward, done, trun, _ = env.step(env.action_space.sample())

        if i != 11:
            state = state_

    from matplotlib import pyplot as plt

    plt.imshow(state[3], interpolation='nearest')
    plt.show()

    plt.imshow(state_[0], interpolation='nearest')
    plt.show()

    plt.imshow(state_[1], interpolation='nearest')
    plt.show()

    plt.imshow(state_[2], interpolation='nearest')
    plt.show()

    plt.imshow(state_[3], interpolation='nearest')
    plt.show()

    print(state_.shape)"""

    s0 = np.random.randint(1, 255, (framestack, image_size, image_size), dtype=np.uint8)
    s1 = np.random.randint(1, 255, (framestack, image_size, image_size), dtype=np.uint8)
    start = time.time()
    tot_samples = 0
    for i in range(20):

        state, action, reward, state_, done = create_experience(s0)
        if i == 5:
            done = True

        tree.append(state, action, reward, state_, done, 0)
        tot_samples += 1
        s0 = state_[:]

        print("Maintree")
        print(tree.st.sum_tree)
        print("min tree")
        print(np.array(tree.priority_min))

        state, action, reward, state_, done = create_experience(s1)

        tree.append(state, action, reward, state_, done, 1)
        tot_samples += 1

        s1 = state_[:]

        print("Maintree")
        print(tree.st.sum_tree)
        print("min tree")
        print(np.array(tree.priority_min))

        if tree.capacity >= batch_size:
            for i in range(20):
                tree_idxs, states, actions, rewards, n_states, dones, weights = tree.sample(batch_size)

                if np.isin(0, states.cpu()):
                    print(states)
                    asdafda

            # Update priorities
            new_priorities = np.array([round(np.random.random(), 3) for i in range(batch_size)])
            tree.update_priorities(tree_idxs, new_priorities)

    end = time.time()
    print("Time:")
    print(end - start)
    raise Exception("Done")
