# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:36:35 2023

@author: KNebiolo

Script Intent: Create a battery based on models of Castro-Santos and Brett.

The fatigue method provides a means to calculate the fatigue level of a fish.   

The work of Castro-Santos (2005) gives us fatigue time as a function of swimming 
speed and fish length.  

ln(T) = a + b Us

where T = time in seconds, a and b are coefficients, and Us is swim speed in 
body lengths per second. 

In sustained swimming mode, fish will not fatigue. However, once a threshold is 
passed and a fish is in prolonged swimming mode, it is swimming anaerobically 
and utilizing energy reserves.  As fish swim even faster still, they will enter 
burst mode, which comes at an even higher metabolic cost.   

When fish start swimming anaerobically, their energy reserves start to decrease.  
At the end of a timestep, we reduce energy reserves proportional to the length 
of the timestep compared to the time until fatigue. For example, say a fish has 
45 seconds until it fatigues at its current swim speed, and the timestep is 0.25 
seconds.  At the end of the first timestep (0.25 seconds later) the fish has 
44.75 seconds until it fatigues and 99.44% of its original energy reserves.  
At the end of the next time step, if it does not change swimming speed, the fish 
has 44.5 seconds until it fatigues.  However, the fish may need to accelerate 
to maintain speed over ground.  If it accelerates and increases swim speed we 
may find that there is now 40 seconds until the fish fatigues.  But, the fish 
does not have 100% of the necessary reserves, it has 99.44%.  Therefore, the 
actual time to fatigue is 39.776 seconds (40 * 0.9944). 

When a fish reduces swim speed below anaerobic levels, it starts to recover.  
See Figure 18 (Brett 1964).  The fish does not reach 100% recovered until 11
hours later.  However, studies show that the fish can perform at high levels 
again with as little as 45 minutes between tests.  Brett even considers fish 
recovered after 3.2 hours because they return to spontaneous activity.  
Maybe a fish must wait at least 45 minutes or up until 3.2 hours?  At 0 minutes 
0 % recovered, 45 minutes 100% recovered, can we fit exponential decay to that? 
"""
# import dependencies
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

# identify workspaces
inputWS = r"J:\2819\005\Calcs\ABM\Data"
outputWS = r"J:\2819\005\Calcs\ABM\Output"


# import recovery data and fit function 
recover = pd.read_csv(os.path.join(inputWS,'recovery.csv'))
recover['Seconds'] = recover.Minutes * 60.
recovery = interpolate.CubicSpline(recover.Seconds,recover.Recovery,extrapolate = True,)
secs = np.arange(0,2760,60) 
int_recover = recovery(secs)
# plt.plot(secs,int_recover)
# plt.plot(recover.Seconds,recover.Recovery,'ro')
# plt.show


# Identify Variables Used in Battery Function 
'''variables are exogenous and will be properties of an agent class object,
therefore instead of passing all variables we will just pass self'''

# use American Shad as surrogate for now
max_U = 7.2
a_p = 10.7
b_p = -1.0
a_s = 6.16
b_s = -0.33

# swimming mode
mode = 'sustained' # sustained, prolonged, sprint, or holding 

# current and previous swim speed - if swim speed changes so does rate of fatigue
swim_speed = 2. 

# current battery level
batt = 1.

# recharge state - initial battery drain has recharge state of 1.
recharge = 1.

# running time to fatigue in seconds
ttfr = 0.0

# change in time per time step in seconds
dt = 1.

# total recovery time
stopwatch = 0.0

# prolonged time
prolong_time = dt

# create some arrays to hold data for QC
ttf_arr = np.zeros(3600)
stopwatch_arr = np.zeros(3600)
batt_arr = np.zeros(3600)
mode_arr = np.repeat('',3600)

for t in np.arange(0,3600,1):
    # fish is moving and behavior may or may not be aneorbic
    if batt > 0.1 and mode != 'holding':
        # # make things interesting, throw a random swim speed
        swim_speed = np.random.randint(1,10,1)[0]
        
        # calculate ttf
        if 4 < swim_speed <= max_U:
            # reset stopwatch
            stopwatch = 0.0
            
            # set swimming mode
            mode = 'prolonged'
            
            # calculate time to fatigue at current swim speed
            ttf = np.exp(a_p + swim_speed * b_p)
            fuck
            
        elif swim_speed > max_U:
            # reset stopwatch
            stopwatch = 0.0
            
            # set swimming mode
            mode = 'sprint'
            
            # calculate time to fatigue at current swim speed
            ttf = np.exp(a_s + swim_speed * b_s)
            
        else:
            mode = 'sustained'
            
            # calculate recovery % at beginning of time step
            rec0 = recovery(stopwatch) / 100.
            
            # make sure realistic value - also can't divide by 0
            if rec0 < 0.0:
                rec0 = 0.0
                
            # calculate recovery % at end of time step
            rec1 = recovery(stopwatch + dt) / 100. 
            
            if rec1 > 1.0:
                rec1 = 1.0
            
            # calculate percent increase
            per_rec = rec1 - rec0
            
            # stopwatch
            stopwatch = stopwatch + dt
            
            ttf = 1
            
        # take into account the time that has already past - that's now how long a fish has
        ttf0 = ttf - ttfr
        
        # calculate time to fatigue at end of time step
        ttf1 = ttf0 - dt
        
        # add to running timer
        ttfr = ttfr + dt
        
        if mode != 'sustained':
            # calculate battery level
            batt = recharge * ttf1/ttf
            
            # make sure battery drain is reasonable
            if batt < 0.0:
                batt = 0.0
        else:
            # calculate battery level
            batt = batt + per_rec    
            
            # make sure battery recharge is reasonable
            if batt > 1.0:
                batt = 1.0
        
        # # increase swim speed
        # swim_speed = swim_speed * 1.01
        
        if batt <= 0.1:
            mode = 'holding'
            ttfr = 0.0
            
    # fish is station holding and recovering    
    else:
        ''' recovery is based on how long a fish has been in a recovery state'''
        # calculate recovery % at beginning of time step
        rec0 = recovery(stopwatch) / 100.
        
        # make sure realistic value - also can't divide by 0
        if rec0 < 0.0:
            rec0 = 0.0
            
        # calculate recovery % at end of time step
        rec1 = recovery(stopwatch + dt) / 100. 
        
        if rec1 > 1.0:
            rec1 = 1.0
        
        # calculate percent increase
        per_rec = rec1 - rec0
        
        # calculate battery level
        batt = batt + per_rec
        
        # stopwatch
        stopwatch = stopwatch + dt
        
        # battery is charged enough to start moving again
        if batt >= 0.9:
            # reset recharge stopwatch
            stopwatch = 0.0
            
            # change swimming mode to sustained
            mode = 'sustained'
            
            # set the recharge rate to the current battery 
            recharge = batt
            
            # set swim speed to something reasonable
            swim_speed = 3.
 
    # insert into array
    batt_arr[t] = batt
    ttf_arr[t] = ttfr
    
plt.plot(np.arange(0,3600,1),batt_arr)
plt.show




    
    
    
    
        
    
