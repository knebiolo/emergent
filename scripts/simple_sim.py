# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:21:42 2021

@author: KNebiolo
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib import colors
from matplotlib.collections import PatchCollection
import os
import numpy as np
import time
from emergent import emergent
import pandas as pd

# identify project workspace and parameters
inputWS = r"D:\ABM Simulations\Data"
outputWS = r"D:\ABM Simulations\Output"
proj_dir = r"D:\ABM Simulations"
simName = 'fack'
n_frames = 3600                                                                # Define length of simulation, each frame is 1 second of model time
n_agents = 20
resultsDB = emergent.dBase(outputWS,simName)
dbDir = os.path.join(outputWS,simName + '.db')

# Create obstacle dataframes
obsWS = os.path.join(inputWS,'obstacles')
obs = emergent.obstacles(obsWS)
landWS = os.path.join(inputWS,'land')
land = emergent.obstacles(landWS)
obs = obs.append(land)

# Create Origins Dataframe
origins = pd.read_csv(os.path.join(inputWS,'origin.csv'))
nodes = origins.OBJECTID.values

# Create Destinations Dataframe
destinations = pd.read_csv(os.path.join(inputWS,'destination.csv'))

# start up simulation initialization function
start_up = emergent.simulation(proj_dir,simName, n_frames,n_agents,obs, land, origins, destinations)

fuck

# create simulation environment
fig = plt.figure()
fig.set_size_inches(14,7,True)
ax = fig.add_subplot(111)
patches = []
for i in agents:
    rotPos = i.shapePos()
    polygon = Polygon(rotPos[:,:2], color = 'b', fill = True, closed = True)
    patches.append(polygon)
p = PatchCollection(patches)
destL = []
for i in agents:
    destL.append(np.ndarray.tolist(i.dest))
destL = np.array(destL)
del i

ax.plot(destL[:,0],destL[:,1], 'gv')
ax.add_collection(p)
ax.add_collection(f)
ax.set_xlim([726678,769259])
ax.set_ylim([4556396,4586487])
text99 = ax.text(727000,4557000,'time: ', fontsize = 8)

# run a simulation!
t0 = time.time()
ani = animation.FuncAnimation(fig, shippy.simulate, init_func = shippy.sim_init, frames = n_frames, interval = 100, fargs = (agents, 1000000000, n_frames)).save(os.path.join(outputWS,simName + '.mp4'))#, writer = writer, dpi = dpi)
plt.show()
t1 = time.time()
print ("animation complete in %s seconds, output saved"%(round(t1 - t0,3)))

