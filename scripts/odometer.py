# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:17:28 2023

@author: KNebiolo

Script Intent: We need to develop a method that accounts for the amount of kCal
consumed during a simulation.  An odometer if you will. 

The intent of this method is to keep a running counter of the amount of kCal 
consumed by converting the amount of oxygen resprired into calories with standard
metabolic equations.  

Brett (1964) provides active metabolic rates or oxygen consumption (O2/kg/hr) 
as a function of water temperature and swimming speed (body lengths/second), 
while Brett and Glass (1973) provide standard metabolic rate as a function of 
water temperature and weight.   
 
"""
# import dependencies
import numpy as np

# declare workspaces
inputWS = r"J:\2819\005\Calcs\ABM\Data"
outputWS = r"J:\2819\005\Calcs\ABM\Output"

# unknowns - what are the length units? assuming cm for now and bl/s

# create a function for total metabolic costs - many of these variables will 
# be class properties
def odometer(u,ucrit,mass, wave_drag,water_temp,swim_mode):
    # calculate active standard metabolic rate using Table 2 from Brett and Glass (1973)
    # 02_rate in units of mg O2/hr
    if water_temp <= 5.3:
        sr_o2_rate = np.exp(0.0565 * np.power(np.log(mass),0.9141)) 
        ar_o2_rate = np.exp(0.4667 * np.power(np.log(mass),0.9989))
        
    elif 5.3 < water_temp <= 15:
        sr_o2_rate = np.exp(0.1498 * np.power(np.log(mass),0.8465))
        ar_o2_rate = np.exp(0.9513 * np.power(np.log(mass),0.9632))

    elif water_temp > 15:
        sr_o2_rate = np.exp(0.1987 * np.power(np.log(mass),0.8844))
        ar_o2_rate = np.exp(0.8237 * np.power(np.log(mass),0.9947))

    # calculate total metabolic rate
    swim_cost = sr_o2_rate + wave_drag * (np.exp(np.log(sr_o2_rate) + u * ((np.log(ar_o2_rate) - np.log(sr_o2_rate))/ucrit))-sr_o2_rate)
    
    return swim_cost, ar_o2_rate, sr_o2_rate

# define variables and try out a calculation
swim_speed = 5.0  # bl/s
mass = 1000       # g
ucrit = 7.        # bl/s
wave_drag = 3.4
water_temp = 18.  # C
swim_mode = 'prolonged'

test = odometer(swim_speed,ucrit,mass,wave_drag,water_temp,swim_mode)


    
    