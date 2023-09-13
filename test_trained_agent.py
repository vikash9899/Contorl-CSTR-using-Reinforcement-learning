import DDPG as dg 
import env as e 
import utils
import numpy as np 
import ploting
import time 


def closed_loop_simulation_rl():  
    
    """
        Function is used to test trained RL agent (actor network) which is saved in the folder ./data/models/..,

        It saves the states and action in the ./data/test_data/.., and plot in the ./data/test_plots/..   :::: 
    """

    cstr = e.cstr_env()  

    state_dim = cstr.observation_space.shape[0]     
    action_dim = cstr.action_space.shape[0]         
    
    ddpg = dg.DDPG(state_dim, action_dim)     
    
    dir = utils.global_dir + '/data' 
    
    ddpg.load(dir, 17)    
    
    states = [] 
    actions = [] 
    
    start = time.time() 
    for ep in range(0, 1):  
        
        print("running..", ep)  
        
        obs, _ = cstr.reset()  
        states.clear() 
        actions.clear() 

        scaled_states = []  
        scaled_action = []  

        for itr in range(500):

            scaled_states.append(obs) 

            s = obs[:4]  

            s_t = utils.reverse_normalize_minmax_states(s)  

            error = utils.reverse_normalize_minmax_error(np.array(obs[4])) 
            ierror = utils.reverse_normalize_minmax_ierror(np.array(obs[5])) 

            o_obs = np.concatenate([s_t, error, ierror])  
            # print(o_obs) 

            states.append(o_obs) 


            a = ddpg.select_action(obs)     
            scaled_action.append(a)    
            
            # next_state, reward, terminated, trancate, _ 
            obs, rewards, done, tranc,  _ = cstr.step(a)    
            
            a = utils.reverse_normalize_minmax_actions(a)    
            # s = utils.reverse_normalize_minmax_states(s)

            actions.append(a)   

            # actions.append(a)  
            # states.append(s)  
            
            # if done: 
            #     break 
        
        end = time.time() 

        time_in_second = (end - start) 
        time_in_min = time_in_second/60

        print("time in seconds : ", time_in_second ) 
        print("time in minutes : ", time_in_min ) 


        file_name = './data/time_rl'+'.txt'  
        with open(file_name, 'w') as file: 
            file.write("time in seconds :" + str(round(time_in_second, 2)))  
            file.write("\ntime in minutes :" + str(round(time_in_min, 2)))   
                
        Xk = np.array(states)       
        Uk = np.array(actions)   
        
        file1 = utils.global_dir + "/data/test_data/xk" + str(ep) + ".csv"
        file2 = utils.global_dir + "/data/test_data/uk" + str(ep)+  ".csv"  

        np.savetxt(file1, Xk, delimiter=',')    
        np.savetxt(file2, Uk, delimiter=',')  


        file3 =  utils.global_dir + "/data/test_plots/fig_" + str(ep) + '.jpg'

        Xk = Xk[:, [0, 1, 2, 3]] 
        ploting.plot_rl_comparision(Xk, Uk, ep, file3) 

        Xk = np.array(scaled_states)   
        Uk = np.array(scaled_action)    
        
        
        # file1 = utils.global_dir + "/data/test_data/s_xk" + str(ep) + ".csv"
        # file2 = utils.global_dir + "/data/test_data/s_uk" + str(ep)+  ".csv"  

        # np.savetxt(file1, Xk, delimiter=',')    
        # np.savetxt(file2, Uk, delimiter=',')      



if __name__=="__main__":
    closed_loop_simulation_rl()
