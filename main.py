import numpy as np
import DDPG.DDPG as dg 
import Env.env as e
import Utils.utils as utils
import DDPG.buffer as bf
import time 

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


# setting the episode length. 
episode_length = 500 


def actor_training(env, policy, batch_size, replay_buffer, start_timesteps, total_ep_training ):

    global episode_length 

    # list to store the episode rewards. 
    ep_reward_list = [] 

    decay_rate = 0.02 

    data_listt = [] 

    start = time.time() 
    for ep in range(1, total_ep_training):
        
        print("\n \n \n episode : ", ep ) 
        episode_reward = 0 
        # call the reset function to get the initial state....
        state, info_dic = env.reset() 

        # setpoint state and setpoint actions for the current episode 
        setpoint_state = info_dic['setpoint_state']   
        setpoint_action = info_dic['setpoint_action'] 

        # decay of the statndard diviation with the episode 
        decay_std = 1 * 1/(1 + decay_rate * ep)   

        std_dev = np.array([decay_std, decay_std], dtype=float)     
        ep_len = 0  

        actions__ = [] 
        obs__ = [] 

        us_obs__ = []
        us_actions__ = [] 


        break_episode = False  
        counter = 0
        while not(break_episode): 
            counter += 1

            ep_len += 1 

            ### selecting the actions random actions for some staring some episodes...
            if ep <= start_timesteps:
                action = env.action_space.sample()   
            else:

                noisy_action = policy.select_action(state) + \
                            utils.gaussian_noise(mean=np.array([0, 0]) , std=std_dev )  
                                           
                action = utils.clip_negative_positive_one(noisy_action)  
                
            ### Perform action
            # print("actions : ", action) 

            obs__.append(state)  
            actions__.append(action)    


            t_s  = utils.reverse_normalize_minmax_states(state[:4]) 
            t_e  = utils.reverse_normalize_minmax_error(state[4])  
            t_ie = utils.reverse_normalize_minmax_ierror(state[5]) 

            obsss = np.concatenate([t_s, t_e, t_ie]) 

            us_obs__.append(obsss) 

            t_a = utils.reverse_normalize_minmax_actions(action)   

            us_actions__.append(t_a)    
            
            #### calling the step function which is returning the next_state, reward, termination and trancate....
            next_state, reward, terminated, trancate, _ = env.step(action) 
            
            ### break episode if terminate or trancate is true. 
            break_episode = terminated or trancate 

            done = terminated 

            if done:
                d = 1
            else:
                d = 0

            ### Store data in replay buffer 
            replay_buffer.add(state, action, next_state, reward, d)  

            ### Train agent after collecting sufficient data
            if ep >= start_timesteps:
                policy.train(replay_buffer, batch_size )    


            if break_episode: 
                obs__ = np.array(obs__) 
                actions__ = np.array(actions__)  


                print(" step   : ", np.round(setpoint_state, 2))  
                print(" state  : ", np.round(t_s, 2))   
                print(" step a : ", np.round(setpoint_action, 2)) 
                print(" action : ", np.round(t_a, 2))    
                print(" done   : ", done)        

                print("   _std : ", np.round(decay_std, 3))   
                print("episode length : ", counter) 


                dict__n = {
                    "setpoint_state  ": setpoint_state, 
                    "state           ":t_s,
                    "setpoint_action ": setpoint_action,
                    "action          ":t_a,
                    "ep_reward       ":episode_reward,
                    "reward          ":reward,
                    "done            ":done,
                    "std             ":decay_std, 
                    "episode length  ":counter 
                }  

                data_listt.append(dict__n) 


            if ep % 500 == 0: 
                # dir_l = utils.global_dir + '/data'  
                # policy.save(dir_l, ep)   

                # file  = utils.global_dir + '/data/mat/reward_list'+ str(ep) +'.csv'
                # np.savetxt(file, ep_reward_list, delimiter=',')     
                pass 


            state = next_state 
            episode_reward += reward 

        ep_reward_list.append(episode_reward)    
        print("\nEpisode reward : ", round(episode_reward, 2))  
        

    end = time.time() 

    time_taken_in_seconds = round(end-start, 2)  
    time_taken_in_minutes = round(time_taken_in_seconds / 60, 2) 
    time_taken_in_hours  = round(time_taken_in_minutes / 60, 2) 


    # saving the time taken to train the Actor agent. 
    with open('time.txt', 'w') as file:
        file.write(str(start))  
        file.write('\n') 
        file.write(str(end))  
        file.write("\n"+str(time_taken_in_seconds)+" time taken during training in seconds")      
        file.write("\n"+str(time_taken_in_minutes)+" time taken during training in minutes")     
        file.write("\n"+str(time_taken_in_hours)+" time taken during training in hours")     

    print() 

    file  = utils.global_dir + '/data/mat/reward_list.csv'
    np.savetxt(file, ep_reward_list, delimiter=',')  

    dir_l = utils.global_dir + '/data'
    policy.save(dir_l, 17) 


    with open('output_list.txt', 'w') as file:
        for dictionary in data_listt:
            for key, value in dictionary.items():
                file.write('%s:%s\n' % (key, value))
            file.write('\n')




def run(): 

    ## creating the environment. 
    cstr = e.cstr_env() 

    state_dim_ = cstr.observation_space.shape[0] 
    action_dim_ = cstr.action_space.shape[0]  

    print("state dim : ", state_dim_) 
    print("action dim : ", action_dim_ )   
    
    # create the DDPG object. 
    ddpg = dg.DDPG( 
            state_dim=state_dim_, 
            action_dim=action_dim_, 
            discount=1, 
            tau=0.001
        )    

    ## buffer size 
    buffer_size = int(1e5)  

    # creating the buffer which is used to save the tranistions [s, a, s', r, done]
    buffer = bf.ReplayBuffer(state_dim_, action_dim_, buffer_size)  

    actor_training( 
        env= cstr,
        policy= ddpg,
        # batch size used to update the actor-critic.
        batch_size= 256,
        replay_buffer= buffer,
        # warm-up episodes till 500 episodes there is no update in the actor-critic network. 
        start_timesteps= 500,
        # totoal number of episodes used to train the actor-critic network.
        total_ep_training = 3000 
    )


    # buffer.save_buffer() 


if __name__=="__main__":
    run() 


