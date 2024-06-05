

"""
    .. module:: argparse_actions

    This module is used to scale and rescale the observations and actions of the RL agent ---
"""


import numpy as np
import os

global_dir = os.getcwd() 


act_range_min = np.array([0.2, 0.4, 50., 50.], dtype=float)   
act_range_max = np.array([3, 1.2, 150., 150.], dtype=float)  


transform_range_min = np.array([-1., -1., -1., -1.], dtype=float)  
transform_range_max = np.array([1., 1., 1., 1.], dtype=float)   


def normalize_minmax_states(X:np.ndarray):

    """
        This function takes four states :math:`[C_A, C_B, T_R, T_K]`  as the input and transform each of them to the range [-1 to 1] using minmax-scaling.  

        Arguments:
            X (list, np.ndarray):  Array of four values :math:`[C_A, C_B, T_R, T_K]`. 

        Returns:
            X_scaled (list, np.ndarray):  Scaled values of the array X.  

    """

    global act_range_min, act_range_max, transform_range_min, transform_range_max

    X_std = (X-act_range_min) / (act_range_max - act_range_min) 

    X_scaled = X_std * (transform_range_max - transform_range_min) +  transform_range_min 

    return X_scaled 


def reverse_normalize_minmax_states(X_scaled:np.ndarray):

    """
        This function four states :math:`[C_A, C_B, T_R, T_K]` as the input and reverse-transform each of them from range [-1 to 1] to their actual range. 

        Arguments:
            X_scaled (list, np.ndarray):  Array of four values :math:`[C_A, C_B, T_R, T_K]`. 

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """

    
    global act_range_min, act_range_max, transform_range_min, transform_range_max 

    X = ( (X_scaled - transform_range_min) / (transform_range_max - transform_range_min) ) * (act_range_max - act_range_min) + act_range_min

    return X 


action_min_r = np.array([5., -8500.], dtype=float) 
action_max_r = np.array([100., 0.], dtype=float)     


transform_action_min_r = np.array([-1, -1], dtype=float) 
transform_action_max_r = np.array([1, 1], dtype=float) 



def normalize_minmax_actions(X:np.ndarray):
    
    """
        This function four states :math:`[F, \dot Q]` as the input and reverse-transform each of them from range [-1 to 1] to their actual range. 

        Arguments:
            X_scaled (list, np.ndarray):  Array of two values :math:`[F, \dot Q]`. 

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """


    global action_min_r, action_max_r, transform_action_min_r, transform_action_max_r

    X_std = (X-action_min_r) / (action_max_r - action_min_r) 

    X_scaled = X_std * (transform_action_max_r - transform_action_min_r) +  transform_action_min_r 

    return X_scaled 


def reverse_normalize_minmax_actions(X_scaled:np.ndarray):

    """

        function takes two actions :math:`[F, \dot Q]` as the input and reverse-transform each of them from range [-1 to 1] to their actual range. 

        Arguments:
            X_scaled (list, np.ndarray):  Array of two values :math:`[F, \dot Q]`.  

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """
    
    global action_min_r, action_max_r, transform_action_min_r, transform_action_max_r 

    X = ( (X_scaled - transform_action_min_r) / (transform_action_max_r - transform_action_min_r) ) * (action_max_r - action_min_r) + action_min_r

    return X 


act_range_min_e = np.array([0., 0., 0., 0.], dtype=float)      
act_range_max_e = None    


transform_range_min_e = np.array([-1., -1., -1., -1.],  dtype=float)
transform_range_max_e = np.array([ 1., 1., 1., 1.],  dtype=float) 


def normalize_minmax_error(X:np.ndarray): 

    """
        This function takes error of four states :math:`[C_A, C_B, T_R, T_K]`  as the input and transform each of them to the range [-1 to 1] using minmax-scaling.  

        Arguments:
            X (list, np.ndarray):  Array of four values. 

        Returns:
            X_scaled (list, np.ndarray):  Scaled values of the array X.  

    """


    global act_range_min_e, act_range_max_e, transform_range_min_e, transform_range_max_e

    X_std = (X-act_range_min_e) / ((act_range_max_e - act_range_min_e) )

    X_scaled = X_std * (transform_range_max_e - transform_range_min_e) +  transform_range_min_e 

    return X_scaled 


def reverse_normalize_minmax_error(X_scaled:np.ndarray):

    """
        This function takes error of four states :math:`[C_A, C_B, T_R, T_K]` as the input and reverse-transform each of them from range [-1 to 1] to their actual range. 

        Arguments:
            X_scaled (list, np.ndarray):  Array of four values`. 

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """

    
    global act_range_min_e, act_range_max_e, transform_range_min_e, transform_range_max_e 

    X = ( (X_scaled - transform_range_min_e) / (transform_range_max_e - transform_range_min_e) ) * (act_range_max_e - act_range_min_e)  + act_range_min_e

    return X 


act_range_min_ie = np.array([0., 0., 0., 0.], dtype=float) 
act_range_max_ie = None  


transform_range_min_ie = np.array([-1., -1., -1., -1.],  dtype=float)
transform_range_max_ie = np.array([1., 1., 1., 1.],  dtype=float) 


def normalize_minmax_ierror(X:np.ndarray):

    """
        This function takes integral error of four states :math:`[C_A, C_B, T_R, T_K]`  as the input and transform each of them to the range [-1 to 1] using minmax-scaling.  

        Arguments:
            X (list, np.ndarray):  Array of four values. 

        Returns:
            X_scaled (list, np.ndarray):  Scaled values of the array X.  

    """

    global act_range_min_ie, act_range_max_ie, transform_range_min_ie, transform_range_max_ie

    X_std = (X-act_range_min_ie) / (act_range_max_ie - act_range_min_ie)

    X_scaled = X_std * (transform_range_max_ie - transform_range_min_ie) +  transform_range_min_ie 

    return X_scaled 


def reverse_normalize_minmax_ierror(X_scaled:np.ndarray):


    """
        This function takes integral error of four states :math:`[C_A, C_B, T_R, T_K]` as the input and reverse-transform each of them from range [-1 to 1] to their actual range. 

        Arguments:
            X_scaled (list, np.ndarray):  Array of four values`. 

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """

    
    global act_range_min_ie, act_range_max_ie, transform_range_min_ie, transform_range_max_ie 

    X = ( (X_scaled - transform_range_min_ie) / (transform_range_max_ie - transform_range_min_ie) ) * (act_range_max_ie - act_range_min_ie)  + act_range_min_ie

    return X 




def clip_negative_positive_one(x):
    max1 = np.array([1, 1]) 
    min1 = np.array([-1, -1])  
    x = np.maximum(x, min1) 
    x = np.minimum(x, max1) 
    return x  


def clip_actions(X): 

    """
        Function takes the array of actions :math:`[F, \dot Q]` and convert to the given range.  

        Arguments:
            X_scaled (list, np.ndarray):  Array of four values`. 

        Returns:
            X (list, np.ndarray):  returns the original values of the array X_scaled.   

    """

    X = np.clip(X, np.array([5., -8500.]), np.array([100., 0.]))  

    return X 


def gaussian_noise(mean:np.ndarray, std:np.ndarray):
    noise = np.random.normal(mean, std) 
    return noise



