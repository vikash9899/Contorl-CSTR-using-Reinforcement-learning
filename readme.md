# Information about the different files. 

## main.py 
main.py will train the actor-critic network and save the metrics [mean episode reward, actor loss, critic loss] in the "./data/mat" folder and save the trained actor critic network in the "./data/models" key 17 is given to he final model.


## metrics.py 
metrics.py file is used to plot the [mean episode reward, actor loss, critic loss] plots will be saved in the "./data/mat" folder. 


## test_trained_agent.py 
test_trained_agent.py file is used to test the tarined agent. test episode data will be saved in the "./data/test_data" folder and test plots will be saved in the "./data/test_plots" folder.


## env.py 
This file contains the CSTR eviroment where reset, step functions are implemented. 


## DDPG.py  
This file contains the ddpg algorithm.


## buffer.py 
This file use to stores the buffer data used in the ddpg algorithm. 

## simulator.py 
simulator file is used to simulate the CSTR equations/to solve the CSTR equations.

## ploting.py 
ploting file contains the code to plot the closed loop simulation and different figures. 

## random_sa 
This file is used to randomly select the states and action within the given range of the CSTR. 


