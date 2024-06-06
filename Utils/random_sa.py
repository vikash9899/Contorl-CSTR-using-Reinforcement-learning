import random 
import numpy as np 

def sample_actions():
    """
    Samples random actions for a given environment.

    This function generates two random action values:
        1. A flow rate (F) in the range [5, 100).
        2. A heat duty (Q) in the range [-8500, 0).

    Returns:
        numpy.ndarray: An array containing the sampled action values [F, Q].
    """

    F = 5 + random.random() * 95
    Q = -1 * random.random() * 8500 

    return np.array([F, Q]).reshape(2, )



def sample_states():
    """
    Samples random states for a given environment.

    This function generates four random state values:
        1. Concentration of component A (c_a) in the range [0.1, 2).
        2. Concentration of component B (c_b) in the range [0.1, 2).
        3. Reactor temperature (t_r) in the range [50, 150).
        4. Jacket temperature (t_k) in the range [50, 140).

    Returns:
        numpy.ndarray: An array containing the sampled state values [c_a, c_b, t_r, t_k].
    """
    c_a = 0.1 + random.random() * 1.9
    c_b = 0.1 + random.random() * 1.9
    t_r = 50 + random.random() * 100
    t_k = 50 + random.random() * 90

    return np.array([c_a, c_b, t_r, t_k]).reshape(4, )


