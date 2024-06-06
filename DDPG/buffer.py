import numpy as np
import torch
import Utils.utils as utils

class ReplayBuffer(object):

	"""
	   A replay buffer for storing and sampling experience tuples for training reinforcement learning agents.

    Attributes:
        max_size (int): Maximum number of experience tuples the buffer can hold.
        ptr (int): Pointer to the current index in the buffer for inserting new experience tuples.
        size (int): Current number of experience tuples stored in the buffer.
        state (np.ndarray): Array to store states.
        action (np.ndarray): Array to store actions.
        next_state (np.ndarray): Array to store next states.
        reward (np.ndarray): Array to store rewards.
        not_done (np.ndarray): Array to store done flags (1 if not done, 0 if done).
        device (torch.device): Device to which the data is moved (CPU or CUDA).

    Methods:
        add(state, action, next_state, reward, done):
            Adds an experience tuple to the replay buffer.
        sample(batch_size):
            Samples a batch of experience tuples from the replay buffer.
        save_buffer():
            Saves the entire buffer data to a CSV file.
	"""
	
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):

		"""
		Initializes the replay buffer.

        Parameters:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_size (int): Maximum size of the replay buffer.
		"""
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1)) 

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):

		"""
		Adds an experience tuple to the replay buffer.

        Parameters:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            next_state (np.ndarray): Next state.
            reward (float): Reward received.
            done (bool): Done flag (True if episode ended, False otherwise).
		"""
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done 

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):

		"""
		Samples a batch of experience tuples from the replay buffer.

        Parameters:
            batch_size (int): Number of experience tuples to sample.

        Returns:
            tuple: Batch of sampled experience tuples (state, action, next_state, reward, not_done),
                   each as a PyTorch tensor.
		"""
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save_buffer(self):
		"""
		Saves the entire buffer data to a CSV file.

        The buffer data is concatenated into a single array and saved as a CSV file.
		"""
		buffer_data =  np.concatenate([self.state, self.action, self.reward, 
				 self.next_state, self.not_done], axis=1)   
		
		file = utils.global_dir + '/data/buffer_data.csv' 
		np.savetxt(file, buffer_data, delimiter=',') 
