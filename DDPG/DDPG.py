import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# selectiong on which GPU code should be run..
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# if GPU avilable then use the GPU otherwise use the CPU. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# device = "cpu"

# if torch.cuda.is_available():
# 	print("training on the nvedia GPU........")   

# torch.cuda.empty_cache() 
 
# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module): 

	"""
		Actor class define the structure of neural network of the actor part of actor-critic frameworks and doing the forward pass.
		Arguments : 
			state_dim (integer) : number of observation actor observed from the environment. 
			action_dim (integer) : number of actions actor need to give to the environment. 

		Return : 
			model : return the actor model. 

	"""

	def __init__(self, state_dim, action_dim):

		"""
		init function is used to initilize the critic network.
		
		Arguments : 
			state_dim (integer) : number of observation actor observed from the environment. 
			action_dim (integer) : number of actions actor need to give to the environment. 

		Return : 
			None

		"""
		super(Actor, self).__init__()  

		self.l1 = nn.Linear(state_dim, 256) 
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim) 
	
	def forward(self, state):

		"""
			forward function takes the (state, action) and predict its Q-value.  

			Arguements : 
				state (array) : observations of the actor networks. 
				action (array) : output array of the actor network.   
			Return : 
				model : return the critic model. 
		"""

		a = F.relu(self.l1(state))  
		a = F.relu(self.l2(a))  

		model = torch.tanh(self.l3(a))
		return model


class Critic(nn.Module):

	"""
		Critic class defines the structure of the neural network for the critic part of the actor-critic framework
    and performs the forward pass.
	"""
    
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		"""
		Initializes the critic network.

        Arguments:
            state_dim (int): Number of observations the actor receives from the environment.
            action_dim (int): Number of actions the actor needs to take in the environment.

        Returns:
            None

		"""
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):

		"""
			Forward function takes the state and action, and predicts the Q-value.

        Arguments:
            state (array): Observations of the actor network.
            action (array): Output action of the actor network.

        Returns:
            model: The Q-value for the given (state, action) pair.
		"""
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):

	"""
		DDPG class defines the Deep Deterministic Policy Gradient algorithm.
	"""
    
    
	def __init__(self, state_dim, action_dim, discount=0.99, tau=0.001):
		
		""" 
		Initializes the DDPG object.

        Arguments:
            state_dim (int): Number of observations the actor receives from the environment.
            action_dim (int): Number of actions the actor needs to take in the environment.
            discount (float): Discount factor (gamma) used while updating the Q-value of (state, action).
            tau (float): Used to soft update the target actor and critic.

        Returns:
            None
		""" 

		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor) 
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic) 
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount 
		self.tau = tau

		self.actor_loss_list = [] 
		self.critic_loss_list = [] 


	def select_action(self, state):

		"""
			Selects an action given the current state.

        Arguments:
            state (array): Current state of the environment.

        Returns:
            action (array): Action selected by the actor network.

		"""

		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):

		"""
			Trains the actor and critic networks using samples from the replay buffer.

        Arguments:
            replay_buffer (object): Replay buffer object containing interaction data.
            batch_size (int): Number of samples to randomly select from the replay buffer to 
			update the networks. Default value is 256.

        Returns:
            None
		"""

		### Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		### Compute the target Q value 
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()   
	
		### Get current Q estimate  
		current_Q = self.critic(state, action)

		### Compute critic loss 
		critic_loss = F.mse_loss(current_Q, target_Q)  

		### Append the critic loss to the critic_loss_list
		self.critic_loss_list.append(critic_loss.cpu().data.numpy())

		### Optimize the critic 
		self.critic_optimizer.zero_grad() 
		critic_loss.backward() 
		self.critic_optimizer.step() 

		### Compute actor loss  
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		### Append the actor loss to the actor_loss_list. 
		self.actor_loss_list.append(actor_loss.cpu().data.numpy())  

		### Optimize the actor     
		self.actor_optimizer.zero_grad() 
		actor_loss.backward() 
		self.actor_optimizer.step()

		### Update the frozen target models 
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, dir, ep):  
		
		"""
			Saves the actor, critic models and their losses.

        Arguments:
            dir (str): Directory where the models and losses should be saved.
            ep (int): Episode number at which the data is saved.

        Returns:
            None
		"""

		torch.save(self.critic.state_dict(), dir + "/model/_critic" + str(ep))
		torch.save(self.critic_optimizer.state_dict(), dir + "/model/_critic_optimizer" +  str(ep)) 
		
		torch.save(self.actor.state_dict(), dir + "/model/_actor" + str(ep))  
		torch.save(self.actor_optimizer.state_dict(), dir + "/model/_actor_optimizer"+ str(ep))  

		ac_loss = np.asarray(self.actor_loss_list) 
		c_loss = np.asarray(self.critic_loss_list)  

		np.savetxt(dir + "/mat/actor_loss "+ str(ep) + ".csv", ac_loss, delimiter=',') 
		np.savetxt(dir + "/mat/critic_loss"+ str(ep) +".csv", c_loss, delimiter=',') 


	def load(self, dir, ep): 

		"""
			Loads the actor, critic models and their optimizers.

        Arguments:
            dir (str): Directory from where the models and optimizers should be loaded.
            ep (int): Episode number from which the data should be loaded.

        Returns:
            None
		"""

		self.critic.load_state_dict(torch.load(dir + "/model/_critic" + str(ep) )) 
		self.critic_optimizer.load_state_dict(torch.load(dir + "/model/_critic_optimizer" + str(ep) )) 
		self.critic_target = copy.deepcopy(self.critic) 

		self.actor.load_state_dict(torch.load(dir + "/model/_actor" + str(ep) )) 
		self.actor_optimizer.load_state_dict(torch.load(dir + "/model/_actor_optimizer" + str(ep) )) 
		self.actor_target = copy.deepcopy(self.actor) 



