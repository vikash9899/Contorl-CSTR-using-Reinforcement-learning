import numpy as np
import Utils.utils as utils 
from gymnasium import spaces
import gymnasium as gym 
from Utils.random_sa import sample_states, sample_actions 
import Simulation.simulator as simulator 



# maximum episode lenght 
episode_length = 500 

class cstr_env(gym.Env):


    def __init__(self): 

        """
            This functions used to initialize the global variables of the class.  
        """


        # Define action & observation space   
        self.action_space = spaces.Box(low = np.array([-1.0, -1.0], 
                                                    dtype=np.float32), 
                                                    high = np.array([1.0 , 1.0], 
                                                    dtype=np.float32), 
                                                    dtype=np.float32, shape=(2, ) )   


        # self.action_space = spaces.Box(low = -8500, high = 0)   
        # self.observation_space = spaces.Box(low=np.array([-1.0, -1.0 , -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 , -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 
        #                                                 dtype=np.float32), 
        #                                                 high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
        #                                                 dtype=np.float32),  
        #                                                 dtype=np.float32, shape=(16, ) )  

        ##### self.action_space = spaces.Box(low = -8500, high = 0)   
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0 , -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ], 
                                                        dtype=np.float32), 
                                                        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  
                                                        dtype=np.float32),  
                                                        dtype=np.float32, shape=(8, ) )  

        # n_episode variable give the current episode number. 
        self.n_episode = 0 

        # file1 = utils.global_dir + "/data/open/setpoint_state_ref.csv"      
        # file2 = utils.global_dir + "/data/open/setpoint_action_ref.csv"     


        # self.st_states = np.loadtxt(file1, delimiter=',')   
        # self.st_actions = np.loadtxt(file2, delimiter=',')   

        # self.setpoint_len = self.st_states.shape[0] 


    def setpoint(self):

        """
            function setpoint is used to give the reference state / goal state to the current episode. 
            (this is one of the steady state of the process )
        """

        # This the reference state for each episode if you want to train for other reference point change the below line. 
        self.setpoint_states  =  np.array([.70, .60, 127.25, 124.39], dtype=float)     
        self.setpoint_actions =  np.array([6.01706488, -2655.80418483], dtype=float )   
 
        # normalizing the setpoint states using the min-max scaling. 
        self.scaled_setpoint_state = utils.normalize_minmax_states(self.setpoint_states)     # type: ignore


        return None 
    

    def save_episode_data(self):

        # dir = utils.global_dir + "/data/training_data"   

        # np.savetxt(dir + "/"+str(self.n_episode)+"ep_states.csv" , self.ep_states, delimiter=',')  
        # np.savetxt(dir + "/"+str(self.n_episode)+"ep_actions.csv" , self.ep_actions, delimiter=',')  

        # # np.savetxt(dir + "/"+str(self.n_episode)+"se_ep_states.csv" , self.se_ep_states, delimiter=',')  
        # # np.savetxt(dir + "/"+str(self.n_episode)+"se_ep_actions.csv" , self.se_ep_actions, delimiter=',')  

        # # np.savetxt(dir + "/"+str(self.n_episode)+"s_states.csv" , self.s_ep_states, delimiter=',')  
        # # np.savetxt(dir + "/"+str(self.n_episode)+"s_actions.csv" , self.s_ep_actions, delimiter=',')  

        # dir1 = utils.global_dir + "/data/training_plots"  

        # file1 = dir1 + "/"+str(self.n_episode)+"_xu_.png"  
        # # file2 = dir1 + "/"+str(self.n_episode)+"se_fig_.png"  
        # # file3 = dir1 + "/"+str(self.n_episode)+"s_fig_.png"  
        
        # ploting.plot_rl_comparision(self.ep_states, self.ep_actions, self.setpoint_states, 0, file1)  
        # # ploting.plot_rl_comparision(self.se_ep_states, self.se_ep_actions, 0, file2)  
        # # ploting.plot_rl_comparision(self.s_ep_states, self.s_ep_actions, 0, file3)  

        # file4 = dir1 + "/"+str(self.n_episode)+"_e.png"  
        # ploting.plot_x(self.errors_, 0, file4) 
        
        # file5 = dir1 + "/"+str(self.n_episode)+"_i_e.png"  
        # ploting.plot_x(self.integral_errors_, 0, file5)    

        pass 


    def calculate_reward(self, x_next, u_curr, u_prev):
        
        """
            calculate reward funtion is used to calculate the reward value for state :math:`S_t`  at time instant t. 
            Arguments : 
                x_next(array) : next state when take the action :math:`A_t` at the state :math:`S_t`. 
                u_curr (array) : action at the current time instant t. 
                u_prev (array) : action at the previous time instatn t-1. 
        """ 

        s_x_next = utils.normalize_minmax_states(x_next) 


        # reward function 
        r = np.sum( (s_x_next - self.scaled_setpoint_state)** 2)  

        reward = -r 
        
        return reward 


    def calculate_errors(self, state):
        """
            Calculate the error between the current 'state' vector :math:`[C_A, C_B, T_R, T_K]` and reference state :math:`[C_{A, ref}, C_{B, ref}, T_{R, ref}, T_{K, ref}]`. 

            Arguments : 
                state (array): current state of the precess. 
            
            Return : 
                errors (array): errors of the of current state
        """

        # errors = (state - self.setpoint_states)** 2   
        errors = np.abs(state - self.setpoint_states)  

        self.errors_ = np.append(self.errors_, np.copy(errors.reshape(1, 4)), axis=0) 

        return errors 


    def calculate_integral_errors(self):

        """
            Calculate the integral error between of current 'state' vector :math:`[C_A, C_B, T_R, T_K]` 
            Arguments : 
            
            Return : 
                integral errors (array) : 
        """      

        integral_errors = np.sum(self.errors_, axis=0) 

        self.integral_errors_ = np.append(self.integral_errors_, np.copy(integral_errors.reshape(1, 4)), axis=0)  

        return integral_errors 


    def is_done(self, x_next):
        """
            Check the done condition i.e. weather the last five consecutive states are within the defined epsilon or not. 
        """

        done=False

        c1 = (abs(x_next[0] - self.setpoint_states[0]) < 0.01)   # type: ignore
        c2 = (abs(x_next[1] - self.setpoint_states[1]) < 0.01)   # type: ignore
        c3 = (abs(x_next[2] - self.setpoint_states[2]) < 0.5)    # type: ignore
        c4 = (abs(x_next[3] - self.setpoint_states[3]) < 0.5)    # type: ignore


        steady_state = c1 and c2 and c3 and c4 

        self.goal_state_done[self.ep_step] = steady_state
        
        if self.ep_step > 5: 
            p5 = self.goal_state_done[self.ep_step-4] 
            p4 = self.goal_state_done[self.ep_step-3] 
            p3 = self.goal_state_done[self.ep_step-2] 
            p2 = self.goal_state_done[self.ep_step-1] 
            p1 = self.goal_state_done[self.ep_step-0] 

            if p5 and p4 and p3 and p2 and p1:
                done = True  


        return done 


    def step(self, s_action):

        """
            Step function take the action of the agent and returns the next-observation to the agent. 
        """

        # get the original scale of the actions. 
        action = utils.reverse_normalize_minmax_actions(s_action) 

        # make current_u to the action coming in step function.
        self.current_u = action 

        # print("action : ", action) 
        
        # take the one step in the cstr. 
        try :
            time_step = 0   
            x_next = simulator.cstr_dynamics(time_step, time_step + 0.005, self.current_s[0], # type: ignore
                                            self.current_s[1], self.current_s[2], self.current_s[3],  # type: ignore
                                            action[0], action[1])  # type: ignore
            
        except:
            print("Overflow error occured : ")  
            print("actions : ", action) 
            print("states : ", self.current_s ) 
            x_next = self.current_s   

        
        # print("X_next : ", x_next) 
        done = self.is_done(x_next)   

        # print("done : ", done)    

        # calculate the reward for the current state. 
        reward = self.calculate_reward(x_next, self.current_u, self.previous_u) 

        # calculate the error for the current state. 
        error = self.calculate_errors(x_next)   

        ## this will calculate the integral_error for the current state. 
        # integral_error = self.calculate_integral_errors() 

        s_error = utils.normalize_minmax_error(error)  

        # s_integral_error = utils.normalize_minmax_ierror(integral_error)   


        # Normalizing the states 
        s_x_next = utils.normalize_minmax_states(x_next)   # type: ignore
        # s_action = utils.normalize_minmax_actions(se_action) 

        # observations = np.concatenate([s_x_next, self.scaled_setpoint_state,  s_error, s_integral_error]) 

        # Observations to the current state
        observations = np.concatenate([s_x_next,  s_error]) 


        self.ep_states = np.append(self.ep_states, np.copy(x_next.reshape(1, 4)), axis=0)   # type: ignore   
        self.ep_actions = np.append(self.ep_actions, np.copy(action.reshape(1, 2)), axis=0)       

        # changing the previous state to the current state. 
        self.previous_u = self.current_u 
        # changing the current state to next state
        self.current_s = x_next 
        # increase the step by one 
        self.ep_step += 1  


        # print(observations)
        if (self.ep_step == episode_length or done) and (self.n_episode%episode_length) in [episode_length-2, episode_length-1 ]:     
            # print(self.n_episode)  
            # print(self.n_episode%20)   
            self.save_episode_data() 


        # this is the trancated condition. 
        trancated = False 
        if self.ep_step == episode_length:
            trancated = True


        if self.ep_step == episode_length-1 or done:      
            self.n_episode += 1 
        
        # if done is true i.e. terminated is equal to done. 
        terminated = done

        # returning the next observations, reward, terminated, trancated and info
        return observations, reward, terminated, trancated, {}    


    def reset(self):

        """
            Reset functions is called at start of each episode. It is used to give the initial state to agent and set the goal state. 
        """

        self.ep_step = 0 

        self.current_u= None 
        self.previous_u = None 
        self.current_s = None 

        ## this the list of true false which stores the weather the state is near to the goal state or not. 
        self.goal_state_done = [False] *  (episode_length+5)

        self.errors_ = np.empty(shape=( 0 , 4), dtype=np.float64)   
        self.integral_errors_ = np.empty(shape=( 0 , 4), dtype=np.float64)    

        self.setpoint_states = None
        self.setpoint_actions = None 

        self.ep_states = np.empty(shape=( 0 , 4), dtype=np.float64)
        self.ep_actions = np.empty(shape=( 0 , 2), dtype=np.float64) 

        ## this function is set the setpoint for the current state and actions. 
        self.setpoint()  


        # this is the fixed initial state. 
        state, action = np.array([1.04, 0.8, 140.52, 139.10 ]),  np.array([21.01, -1234.44]) 

        # self.ep_states  = np.append(self.ep_states, np.copy(state.reshape(1, 4)), axis=0)      
        # self.ep_actions = np.append(self.ep_actions, np.copy(action.reshape(1, 2)), axis=0)        


        self.current_u = action 
        self.previous_u = action  
        self.current_s = state 

        # calculating the error for the current state. 
        error = self.calculate_errors(state)  

        error_max = np.array([error[0]+0.1, error[1]+0.1, error[2]+1, error[3]+1])   

        utils.act_range_max_e = np.array(error_max, dtype=float)     


        # integral_error = self.calculate_integral_errors() 
        utils.act_range_max_ie = np.array((error_max)*300, dtype=float)  

        s_error = utils.normalize_minmax_states(error) 
        # s_integral_error = utils.normalize_minmax_ierror(integral_error) 

        s_state = utils.normalize_minmax_states(state) 
        # s_action = utils.normalize_minmax_actions(action)


        # observations contains the state and errors. 
        observations = np.concatenate([s_state, s_error])  


        info_dic = { 
            "setpoint_state": self.setpoint_states, 
            "setpoint_action": self.setpoint_actions,
            "error_max": error_max
        } 

        return observations, info_dic 


