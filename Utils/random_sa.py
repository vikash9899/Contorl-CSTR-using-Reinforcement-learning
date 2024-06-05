import random 
import numpy as np 

def sample_actions():
    F = 5 + random.random() * 95
    Q = -1 * random.random() * 8500 

    return np.array([F, Q]).reshape(2, )



def sample_states():
    c_a = 0.1 + random.random() * 1.9
    c_b = 0.1 + random.random() * 1.9
    t_r = 50 + random.random() * 100
    t_k = 50 + random.random() * 90

    return np.array([c_a, c_b, t_r, t_k]).reshape(4, )


