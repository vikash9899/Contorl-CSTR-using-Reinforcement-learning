import numpy as np
from math import exp 
from scipy.integrate import odeint 


def cstr_dynamics(ti,tf,y01, y02, y03, y04, F, Q_dot):

    """
    Simulates the dynamics of a Continuous Stirred Tank Reactor (CSTR) using 
    ordinary differential equations (ODEs).

    Parameters:
        ti (float): Initial time.
        tf (float): Final time.
        y01 (float): Initial concentration of component A (mol/l).
        y02 (float): Initial concentration of component B (mol/l).
        y03 (float): Initial reactor temperature (Celsius).
        y04 (float): Initial jacket temperature (Celsius).
        F (float): Flow rate (l/h).
        Q_dot (float): Heat duty (kj/h).

    Returns:
        numpy.ndarray: Array of the final state values [C_A, C_B, T_R, T_K], where:
            C_A (float): Concentration of component A (mol/l).
            C_B (float): Concentration of component B (mol/l).
            T_R (float): Reactor temperature (Celsius).
            T_K (float): Jacket temperature (Celsius).
    """

    def func(y,t):
        """
        Defines the system of ODEs representing the CSTR dynamics.

        Parameters:
            y (list): List of state variables [C_A, C_B, T_R, T_K].
            t (float): Time.

        Returns:
            list: Derivatives of the state variables [dC_A/dt, dC_B/dt, dT_R/dt, dT_K/dt].
        """
        # parameters
        K0_ab = 1.287e12   # K0 [h^-1] 
        K0_bc = 1.287e12   # K0 [h^-1]
        K0_ad = 9.043e9    # K0 [l/mol.h]
        R_gas = 8.3144621e-3   # Universal gas constant
        E_A_ab = 9758.3*1.00   #* R_gas# [kj/mol]
        E_A_bc = 9758.3*1.00   #* R_gas# [kj/mol]
        E_A_ad = 8560.0*1.0    #* R_gas# [kj/mol]
        H_R_ab = 4.2   # [kj/mol A]
        H_R_bc = -11.0     # [kj/mol B] Exothermic
        H_R_ad = -41.85    # [kj/mol A] Exothermic
        Rou = 0.9342   # Density [kg/l]
        Cp = 3.01  # Specific Heat capacity [kj/Kg.K]
        Cp_k = 2.0     # Coolant heat capacity [kj/kg.k]
        A_R = 0.215    # Area of reactor wall [m^2]
        V_R = 10.01    #0.01 # Volume of reactor [l]
        m_k = 5.0  # Coolant mass[kg]
        T_in = 130.0   # Temp of inflow [Celsius]
        K_w = 4032.0   # [kj/h.m^2.K]
        C_A0 = (5.7+4.5)/2.0*1.0   # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
        #beta = 0.9
        #alpha = 0.95
        beta = 1
        alpha = 1
        dydt1 = F*(C_A0 - y[0]) -(beta * K0_ab * exp((-E_A_ab)/((y[2]+273.15))))*y[0] - (K0_ad * exp((-alpha*E_A_ad)/((y[2]+273.15))) )*(y[0]**2)
        dydt2 = -F*y[1] + (beta * K0_ab * exp((-E_A_ab)/((y[2]+273.15))))*y[0] - (K0_bc * exp((-E_A_bc)/((y[2]+273.15))))*y[1]
        dydt3 = (((beta * K0_ab * exp((-E_A_ab)/((y[2]+273.15))))*y[0]*H_R_ab + (K0_bc * exp((-E_A_bc)/((y[2]+273.15))))*y[1]*H_R_bc + (K0_ad * exp((-alpha*E_A_ad)/((y[2]+273.15))))*(y[0]**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-y[2]) +((K_w*A_R*(y[3]-y[2]))/(Rou*Cp*V_R))
        dydt4 = (Q_dot + K_w*A_R*(y[2]-y[3]))/(m_k*Cp_k)
        return [dydt1,dydt2,dydt3,dydt4] 


    #Initial Conditions
    # tf = ti + 0.005
    y0 = np.zeros((4,1)) 
    y0[0] = y01
    y0[1] = y02
    y0[2] = y03
    y0[3] = y04
    y0 = [y01, y02, y03, y04] 
    t = np.linspace(ti,tf)  # tspan  

    y = odeint(func, y0, t)     # solver

    y = y[-1] 
    
    return y
  
