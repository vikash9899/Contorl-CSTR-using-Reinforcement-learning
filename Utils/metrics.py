import Utils.utils as utils 
import numpy as np 
import matplotlib.pyplot as plt

def plot_series(series, title, xlabel, ylabel, file): 
    """ 
    """
    x = list(range(500, len(series)+500))  
    plt.figure(figsize=(10, 6)) 
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.title(title, fontsize=24)  
    plt.xlabel(xlabel, fontsize=24)  
    plt.ylabel(ylabel, fontsize=24)  
    plt.plot(x, series) 
    plt.savefig(file)  
    plt.close()  


def plot_sliding_mean(data, window, titile, xlabel, ylabel, file):
    
    m_cl = [] 
    for i in range(0, len(data)-window):
        # print(i) 
        me = sum(data[i:i+window])/window
        m_cl.append(me) 


    plot_series(m_cl, titile, xlabel, ylabel, file)    


if __name__=="__main__": 
    
    file1 = utils.global_dir + '/data/mat/reward_list.csv' 
    file2 = utils.global_dir + '/data/mat/actor_loss 17.csv' 
    file3 = utils.global_dir + '/data/mat/critic_loss17.csv' 
    # file4 = utils.global_dir + '/data/mat/ep_len.csv'   
    
    reward = np.loadtxt(file1, delimiter=',') 
    actor_loss = np.loadtxt(file2, delimiter=',') 
    critic_loss = np.loadtxt(file3, delimiter=',') 
    # ep_len = np.loadtxt(file4, delimiter=',')  
    

    reward = reward[511:-10]   
    critic_loss = critic_loss[100:-1000000]  
    actor_loss = actor_loss[100:-1000000] 


    file1 = utils.global_dir + '/data/mat/reward_list.png' 
    file2 = utils.global_dir + '/data/mat/actor_loss.png' 
    file3 = utils.global_dir + '/data/mat/critic_loss.png' 
    # file4 = utils.global_dir + '/data/mat/ep_len.png' 
    
    plot_series(reward, 'Reward', 'episode ', 'reward', file1) 
    plot_series(actor_loss, 'Actor loss', 'number of updates ', 'actor loss', file2)  
    plot_series(critic_loss, 'critic loss', 'number of updates ', 'critic loss', file3)    
    # # plot_series(ep_len, 'ep length', 'episode ', 'ep length', file4)    

    # window = 10 
    # plot_sliding_mean(reward, window, 'Reward', 'episode ', 'reward', file1) 
    # plot_sliding_mean(actor_loss, window, 'Actor loss', 'episode ', 'actor loss', file2)  

    # window = 500
    # plot_sliding_mean(critic_loss, window, 'critic loss', 'episode ', 'critic loss', file3)      

    
     
     
     