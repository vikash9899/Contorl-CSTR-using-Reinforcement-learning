import numpy as np 
import matplotlib.pyplot as plt 

"""
    This module is used to ploting the figures 
"""

def plot_xu(S, I, file):

    """
        Function plot_xu is used to plots the four states :math:`[C_A, C_B, T_R, T_K]` and two action :math:`[F, \dot Q]`.  

        Arguments : 
            **S (np.ndarray)** : S is the numpy array contents the four states :math:`[C_A, C_B, T_R, T_K]`.  \n
            **I (np.ndarray)** : I is the numpy array contents the two actions :math:`[F, \dot Q]`.  \n
            **file (str, path)** : path where figure need to be saved.  \n 
            
        Return :  
            None
    """



    S = np.array(S) 
    I = np.array(I)  
    length = S.shape[0] 
    x = list(range(length))    
    
    S = np.transpose(S) 
    I = np.transpose(I) 
    
    data = np.concatenate([S, I], axis=0 )  

    title = [  "control $(C_A)$",  "control  $(C_B)$",  "control  $(T_R)$",
              "control  $(T_K)$", " control $F$ ", "control  $Q$"]

    ylabel = ['$C_A$', '$C_B$', '$T_R$', '$T_K$', '$F$', '$Q$']
    
    
    ymin = [0, 0, 0, 0, 0, -9000] 
    ymax = [3, 3, 200, 200, 150, 0]  
    
    args = {
        'color' : 'green',
        'linestyle' : 'solid',
        'marker' : 'o',
        'markerfacecolor' : 'red',
        'markersize' : 3 
    } 
    
    states = [0.71, 0.60, 127.25, 124.39 ]  
    ca = [0.71] * len(x) 
    cb = [0.6] * len(x) 
    tr = [127.25] * len(x) 
    tk = [124.39] * len(x)   
    
    f = [6.01] * len(x) 
    q = [-2765] * len(x) 
    
    
    state_lists = [ca, cb, tr, tk, f, q]  
	# Figure used for the plot the x, and u of the process. 
    plt.figure(figsize=(20, 10)) 
    for i in range(0, 6):
        plt.subplot(3, 2, i+1)  
        plt.title(title[i], fontsize=20 ) 
        plt.ylabel(ylabel[i], fontsize=16 )
        plt.xlabel('time', fontsize=16) 
        # plt.ylim(ymin[i], ymax[i]) 
        plt.plot(x, data[i])  
        # plt.plot(x, state_lists[i]) 
        
    
    plt.tight_layout() 
    # plt.show() 
    plt.savefig(file)   
    plt.close()   


def plot_rl_comparision(rl_xk, rl_uk, itr, file):


    """
        Function plot_xu is used to plots the four states :math:`[C_A, C_B, T_R, T_K]` and two action :math:`[F, \dot Q]`.  

        Arguments : 
            **rl_xk (np.ndarray)** : S is the numpy array contents the four states :math:`[C_A, C_B, T_R, T_K]`.  \n
            **rl_uk (np.ndarray)** : I is the numpy array contents the two actions :math:`[F, \dot Q]`.  \n 
            **itr (integer)** : itr is unique key given to the figure.  \n 
            **file (str, path)** : path where figure need to be saved.  \n
            
        Return :  
            None
    """

    rl_xk = np.array(rl_xk)   
    rl_uk = np.array(rl_uk)    

    length = rl_xk.shape[0]  
    rl_time = list(range(length))  
    
    rl_data = np.concatenate([rl_xk, rl_uk], axis=1 )  

	# reference state of the each output variables 
    goal_state = np.array([.70, .60, 127.25, 124.39], dtype=float)           
    
    ca = [goal_state[0]] * len(rl_time) 
    cb = [goal_state[1]] * len(rl_time) 
    tr = [goal_state[2]] * len(rl_time) 
    tk = [goal_state[3]] * len(rl_time)   

    list_of_goal_state = [ca, cb, tr, tk] 

    o_ca = round(abs(rl_xk[-1][0] - goal_state[0]), 4) 
    o_cb = round(abs(rl_xk[-1][1] - goal_state[1]), 4) 
    o_tr = round(abs(rl_xk[-1][2] - goal_state[2]), 4) 
    o_tk = round(abs(rl_xk[-1][3] - goal_state[3]), 4) 

    offset = [o_ca, o_cb, o_tr, o_tk]  

    rl_ca_f = round(rl_xk[-1][0], 3) 
    rl_cb_f = round(rl_xk[-1][1], 3) 
    rl_tr_f = round(rl_xk[-1][2], 3) 
    rl_tk_f = round(rl_xk[-1][3], 3)  
    rl_f_f = round(rl_uk[-1][0], 3) 
    rl_q_f = round(rl_uk[-1][1], 3)    

    rl_final_value = [ rl_ca_f,rl_cb_f,rl_tr_f,rl_tk_f, rl_f_f, rl_q_f ]  

	
    title = ["concentration of reactant A $(C_A)- test :$" + str(itr), "concentration of reactant B $(C_B)- test :$" + str(itr), "temperature inside reactor $(T_R)- test :$" + str(itr),
             "temperature of the cooling jacket $(T_K)- test :$" + str(itr), " feed $F$ ", " heat flow $Q- test :$" + str(itr)] 


    ylabel = ['$C_A$', '$C_B$', '$T_R$', '$T_K$', '$F$', '$Q$']
    
    
    fig, axs = plt.subplots(3, 2, figsize=(20, 10), tight_layout=True) 

    # print(type(axs))  
    # print(len(axs)) 

    ## ploting the Ca and the reference state. 
    axs[0][0].set_title(title[0],  fontsize=16)
    axs[0][0].set_xlabel('Time ---', fontsize=12)
    axs[0][0].set_ylabel(ylabel[0],  fontsize=12)

    axs[0][0].plot(rl_time, list_of_goal_state[0], label='setpoint')     
    axs[0][0].plot(rl_time, rl_data[:, 0], label='rl')  
    axs[0][0].legend(["setpoint", "rl"] )   
    

    axs[0][0].text(.5, .85, 'set point: ' + str(goal_state[0]), 
            transform=axs[0][0].transAxes, fontsize=16, color='red')  
    
    axs[0][0].text(.5, .7, 'rl : ' + str(rl_final_value[0]), 
            transform=axs[0][0].transAxes, fontsize=16, color='red') 

    axs[0][0].text(.5, .5, 'offset: ' + str(offset[0]), 
            transform=axs[0][0].transAxes, fontsize=16, color='red')  


    ## ploting the Cb and the reference state. 
    axs[0][1].set_title(title[1],  fontsize=16)
    axs[0][1].set_xlabel('Time ---', fontsize=12)
    axs[0][1].set_ylabel(ylabel[1],  fontsize=12)

    axs[0][1].plot(rl_time, list_of_goal_state[1], label='setpoint')     
    axs[0][1].plot(rl_time, rl_data[:, 1], label='rl')  
    axs[0][1].legend(["setpoint", "rl"] )   
    

    axs[0][1].text(.5, .85, 'set point: ' + str(goal_state[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red')  
    
    axs[0][1].text(.5, .7, 'rl : ' + str(rl_final_value[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red') 

    axs[0][1].text(.5, .5, 'offset: ' + str(offset[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red')  


    ## ploting the Tr and the reference state. 
    axs[1][0].set_title(title[2],  fontsize=16)
    axs[1][0].set_xlabel('Time ---', fontsize=12)
    axs[1][0].set_ylabel(ylabel[2],  fontsize=12)

    axs[1][0].plot(rl_time, list_of_goal_state[2], label='setpoint')     
    axs[1][0].plot(rl_time, rl_data[:, 2], label='rl')  
    axs[1][0].legend(["setpoint", "rl"] )   
    

    axs[1][0].text(.5, .85, 'set point: ' + str(goal_state[2]), 
            transform=axs[1][0].transAxes, fontsize=16, color='red')  
    
    axs[1][0].text(.5, .7, 'rl : ' + str(rl_final_value[2]), 
            transform=axs[1][0].transAxes, fontsize=16, color='red') 

    axs[1][0].text(.5, .5, 'offset: ' + str(offset[2]), 
            transform=axs[1][0].transAxes, fontsize=16, color='red')  

    ## ploting the Tk and the reference state. 
    axs[1][1].set_title(title[3],  fontsize=16)
    axs[1][1].set_xlabel('Time ---', fontsize=12)
    axs[1][1].set_ylabel(ylabel[3],  fontsize=12)

    axs[1][1].plot(rl_time, list_of_goal_state[3], label='setpoint')     
    axs[1][1].plot(rl_time, rl_data[:, 3], label='rl')  
    axs[1][1].legend(["setpoint", "rl"] )   
    

    axs[1][1].text(.5, .85, 'set point: ' + str(goal_state[3]), 
            transform=axs[1][1].transAxes, fontsize=16, color='red')  
    
    axs[1][1].text(.5, .7, 'rl : ' + str(rl_final_value[3]), 
            transform=axs[1][1].transAxes, fontsize=16, color='red') 

    axs[1][1].text(.5, .5, 'offset: ' + str(offset[3]), 
            transform=axs[1][1].transAxes, fontsize=16, color='red')  


    ## ploting the F and the reference state. 
    axs[2][0].set_title(title[4],  fontsize=16)
    axs[2][0].set_xlabel('Time ---', fontsize=12)
    axs[2][0].set_ylabel(ylabel[4],  fontsize=12)

    # axs[2][0].plot(rl_time, list_of_goal_state[4], label='setpoint')     
    axs[2][0].plot(rl_time, rl_data[:, 4], label='rl')  
    axs[2][0].legend(["rl", "rl"] )   
    


    ## ploting the Q and the reference state. 
    axs[2][1].set_title(title[5],  fontsize=16)
    axs[2][1].set_xlabel('Time ---', fontsize=12)
    axs[2][1].set_ylabel(ylabel[5],  fontsize=12)

    # axs[2][1].plot(rl_time, list_of_goal_state[5], label='setpoint')     
    axs[2][1].plot(rl_time, rl_data[:, 5], label='rl')  
    axs[2][1].legend(["rl", "rl"] )   
    

    plt.savefig(file) 
    plt.close()       


def plot_mpc_rl_comparision(rl_xk, rl_uk, mpc_xk, mpc_uk, itr, file):

    """
        Function plot_xu is used to plots the four states :math:`[C_A, C_B, T_R, T_K]` and two action :math:`[F, \dot Q]`.  

        Arguments : 
            **rl_xk (np.ndarray)** : rl_xk is the numpy array contents the four states (closed loop simulation by the reinforcement learning) :math:`[C_A, C_B, T_R, T_K]`.\n 
            **rl_uk (np.ndarray)** : rl_uk is the numpy array contents the two actions :math:`[F, \dot Q]`. \n
            **mpc_xk (np.ndarray)** : mpc_xk is the numpy array contents the four states of closed loop simulation by the MPC controller :math:`[C_A, C_B, T_R, T_K]`. \n
            **mpc_uk (np.ndarray)** : mpc_xk is the numpy array contents the four actions of closed loop simulation by the MPC controller. :math:`[F, \dot Q]`. \n 
            **itr (integer)** : itr is unique key given to the figure. \n
            **file (str, path)** : path where figure need to be saved. \n
            
        Return :  
            None
    """

    """
        Function plot_xu is used to plots the four states :math:`[C_A, C_B, T_R, T_K]` and two action :math:`[F, \dot Q]`.  

        Arguments : 
            **rl_xk (np.ndarray)** : S is the numpy array contents the four states :math:`[C_A, C_B, T_R, T_K]`.  \n
            **rl_uk (np.ndarray)** : I is the numpy array contents the two actions :math:`[F, \dot Q]`.  \n 
            **file (str, path)** : path where figure need to be saved.  \n
            
        Return :  
            None
    """
    rl_xk = np.array(rl_xk)   
    rl_uk = np.array(rl_uk)   

    length = rl_xk.shape[0]  
    rl_time = list(range(length))    
    
    rl_data = np.concatenate([rl_xk, rl_uk], axis=1 )  
    
    mpc_xk = np.array(mpc_xk)   
    mpc_uk = np.array(mpc_uk)   

    length = mpc_xk.shape[0]  
    mpc_time = list(range(length))  
    
    mpc_data = np.concatenate([mpc_xk, mpc_uk], axis=1 )  

    goal_state = np.array([.70, .55, 127.25, 124.39], dtype=float)     
    
    ca = [goal_state[0]] * len(rl_time) 
    cb = [goal_state[1]] * len(rl_time) 
    tr = [goal_state[2]] * len(rl_time) 
    tk = [goal_state[3]] * len(rl_time)   

    list_of_goal_state = [ca, cb, tr, tk] 

    o_ca = round(abs(rl_xk[-1][0] - goal_state[0]), 4) 
    o_cb = round(abs(rl_xk[-1][1] - goal_state[1]), 4) 
    o_tr = round(abs(rl_xk[-1][2] - goal_state[2]), 4) 
    o_tk = round(abs(rl_xk[-1][3] - goal_state[3]), 4) 

    offset = [o_ca, o_cb, o_tr, o_tk]  

    rl_ca_f = round(rl_xk[-1][0], 3) 
    rl_cb_f = round(rl_xk[-1][1], 3) 
    rl_tr_f = round(rl_xk[-1][2], 3) 
    rl_tk_f = round(rl_xk[-1][3], 3)  
    rl_f_f = round(rl_uk[-1][0], 3) 
    rl_q_f = round(rl_uk[-1][1], 3)    

    rl_final_value = [ rl_ca_f,rl_cb_f,rl_tr_f,rl_tk_f, rl_f_f, rl_q_f ]  


    title = ["concentration of reactant A $(C_A)- test :$" + str(itr), "concentration of reactant B $(C_B)- test :$" + str(itr), "temperature inside reactor $(T_R)- test :$" + str(itr),
             "temperature of the cooling jacket $(T_K)- test :$" + str(itr), " feed $F$ ", " heat flow $Q- test :$" + str(itr)] 


    ylabel = ['$C_A$', '$C_B$', '$T_R$', '$T_K$', '$F$', '$Q$']
    

    fig, axs = plt.subplots(3, 2, figsize=(20, 10), tight_layout=True) 

    # print(type(axs))  
    # print(len(axs)) 

    axs[0][0].set_title(title[0],  fontsize=16)
    axs[0][0].set_xlabel('Time ---', fontsize=12)
    axs[0][0].set_ylabel(ylabel[0],  fontsize=12)

    # axs[0][0].plot(rl_time, list_of_goal_state[0], label='setpoint')     
    axs[0][0].plot(rl_time, rl_data[:, 0], label='rl')   
    axs[0][0].plot(mpc_time, mpc_data[:, 0], label='rl')   
    axs[0][0].legend(["rl", "mpc"] )    
    
    

    axs[0][1].set_title(title[1],  fontsize=16)
    axs[0][1].set_xlabel('Time ---', fontsize=12)
    axs[0][1].set_ylabel(ylabel[1],  fontsize=12)

    axs[0][1].plot(rl_time, list_of_goal_state[1], label='setpoint', c='red')     
    axs[0][1].plot(rl_time, rl_data[:, 1], label='rl')  
    axs[0][1].plot(mpc_time, mpc_data[:, 1], label='rl')  
    axs[0][1].legend(["setpoint","rl", "mpc"] )   
    

    axs[0][1].text(.5, .85, 'set point: ' + str(goal_state[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red')  
    
    axs[0][1].text(.5, .7, 'rl : ' + str(rl_final_value[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red') 

    axs[0][1].text(.5, .5, 'offset: ' + str(offset[1]), 
            transform=axs[0][1].transAxes, fontsize=16, color='red')  


    axs[1][0].set_title(title[2],  fontsize=16)
    axs[1][0].set_xlabel('Time ---', fontsize=12)
    axs[1][0].set_ylabel(ylabel[2],  fontsize=12)

    axs[1][0].plot(rl_time, rl_data[:, 2], label='rl')  
    axs[1][0].plot(mpc_time, mpc_data[:, 2], label='rl')  
    axs[1][0].legend(["rl", "mpc"] )   
    
	
    axs[1][1].set_title(title[3],  fontsize=16)
    axs[1][1].set_xlabel('Time ---', fontsize=12)
    axs[1][1].set_ylabel(ylabel[3],  fontsize=12)

    axs[1][1].plot(rl_time, rl_data[:, 3], label='rl')  
    axs[1][1].plot(mpc_time, mpc_data[:, 3], label='rl')  
    axs[1][1].legend(["rl", "mpc"] )   
    


    axs[2][0].set_title(title[4],  fontsize=16)
    axs[2][0].set_xlabel('Time ---', fontsize=12)
    axs[2][0].set_ylabel(ylabel[4],  fontsize=12)

    axs[2][0].plot(rl_time, rl_data[:, 4], label='rl')  
    axs[2][0].plot(mpc_time, mpc_data[:, 4], label='rl')  
    axs[2][0].legend(["rl", "mpc"] )   
    

    axs[2][1].set_title(title[5],  fontsize=16)
    axs[2][1].set_xlabel('Time ---', fontsize=12)
    axs[2][1].set_ylabel(ylabel[5],  fontsize=12)

    axs[2][1].plot(rl_time, rl_data[:, 5], label='rl')  
    axs[2][1].plot(mpc_time, mpc_data[:, 5], label='rl')  
    axs[2][1].legend(["rl", "mpc"] )   
    

    plt.savefig(file, dpi=150)  
    plt.close()      


