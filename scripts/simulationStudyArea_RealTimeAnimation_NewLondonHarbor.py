# Script Intent: Use Ge and Xue's function for attraction and repulsion to let two agents migrate to their destination
#                Agent collision avoidance will not employ COLREGs
# Script Author: KPN

# Import Modules
import numpy as np
from shapely.geometry import Polygon as poly
from shapely.geometry import Point as point
from shapely.geometry import LineString as line
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from shapely.geometry import LineString
import sys
sys.path.append(r"C:\Users\Kevin Nebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\rational_emergence\rational_emergence")
import emergent
import scipy.constants
import pandas as pd
import os
import sqlite3
import time
import pickle
import copy
from scipy import interpolate
import networkx as nx
from osgeo import ogr
import fiona

## Identify Workspaces
inputWS = r"D:\ABM Simulations\Data"
outputWS = r"D:\ABM Simulations\Output"
simName = 'sim_001'
n_frames = 3600                                                                # Define length of simulation, each frame is 1 second of model time
n_agents = 20
resultsDB = emergent.dBase(outputWS,simName)
dbDir = os.path.join(outputWS,simName + '.db')

# Create obstacle dataframe
obsWS = os.path.join(inputWS,'obstacles')
obs = emergent.obstacles(obsWS)
#obs = obs[obs.type != 0]
landWS = os.path.join(inputWS,'land')
land = emergent.obstacles(landWS)


# Create Origins Dataframe
origins = pd.read_csv(os.path.join(inputWS,'origin.csv'))
nodes = origins.OBJECTID.values
#origins = origins[origins.OBJECTID == 7]


# Create Destinations Dataframe
destinations = pd.read_csv(os.path.join(inputWS,'destination.csv'))
#destinations = destinations[destinations.OBJECTID == 1]

travel_network = nx.DiGraph()
travel_network.add_nodes_from(nodes)
travel_network.add_edges_from([(2,1),
                               (3,1),
                               (4,1),
                               (5,1),
                               (6,1),
                               (7,1)])

#nx.draw_circular(travel_network, arrows = True, with_labels = True)
#plt.show()



# create agents
profiles = emergent.initialStates(n_agents,obs,origins,destinations,ships,travel_network,n_frames)
# make a random vessel very slow - 10 knots
slowVessel = np.random.choice(np.linspace(0,n_agents-1))
profiles = profiles.set_value(slowVessel,'V_des',10)

fack
agents = []
rows = np.arange(0,len(profiles) - (len(profiles)-n_agents),1)
for i in rows:
    windFarms = []
    buff = []
    agn = shipABM.Ship(i,profiles,resultsDB, obs,land)
    agn.Route()
    agents.append(agn)


# Set up our analysis environment, square 100 x 100 grid
#x = np.arange(0,20000,0.1)
#y = np.arange(0,10000,0.1)
#x, y = np.meshgrid(x, y)

# Parameters
G = scipy.constants.G

# Time step
dt = 1

farmRows = []
farmID = 1
for i in obs.iterrows():
    if i[1]['type'] != 'land':
        farm = i[1]['shape']
        centroid = farm.centroid.wkt
        perimeter = farm.length
        area = farm.area
        row = ((farmID,centroid,perimeter,area))
        farmRows.append(row)
        farmID = farmID + 1
resultsDB[0].executemany('INSERT INTO windFarms VALUES (?,?,?,?)',farmRows)
resultsDB[1].commit()
 
# create figure of trajectories
#fig = plt.figure()
#fig.set_size_inches(6,4)
#
#ax = fig.add_subplot(111) 

farms = []
paths = []
#
#for row in obs.iterrows():
#    shape = row[1]['shape']
#    typ = row[1]['type']
#    if typ == 'Land':
#        color = 'green'
#    elif typ == 'Traffic Separation':
#        color = 'blue'
#    else:
#        color = 'red'
#    coords = list(shape.exterior.coords)
#    polygon = Polygon(coords,color = color, fill = False, closed = True)
#    farms.append(polygon)
#f = PatchCollection(farms)

for row in land.iterrows():
    shape = row[1]['shape']
    typ = row[1]['type']
    if typ == 'Land':
        color = 'green'
    elif typ == 'Traffic Separation':
        color = 'blue'
    else:
        color = 'red'
    coords = list(shape.exterior.coords)
    polygon = Polygon(coords,color = color, fill = True, closed = True)
    farms.append(polygon)

for i in obs.iterrows():
    if i[1]['type'] != 'land':
        farm = i[1]['shape']
        coords = list(farm.exterior.coords)
        polygon = Polygon(coords,color = 'r', fill = True, closed = True)
        farms.append(polygon)
            
f = PatchCollection(farms)
##
#for i in agents:
#    short = i.short_route
#    shortLine = []
#    idx = 0
#    for i in short:
#        vert = list(i)
#        shortLine.append(vert)
#    paths.append(shortLine)
#sols = LineCollection(paths, colors = 'r', linestyles = '-', zorder = 4)                                                 # create a line collection of bad lines
#ax.add_collection(sols)                                               # add axes ax                                                  # create a line collection of bad lines
#ax.add_collection(f)
#ax.set_xlim([726678,769259])
#ax.set_ylim([4556396,4586487])
##plt.savefig(os.path.join(outputWS,'Figures','Obstructrions.png'))
#plt.show()
#
## create shapely polyline from solution
#
#
## define a polygon feature geometry with a single attribute
#schema = {
#        'geometry':'LineString',
#        'properties':{'origin':'int','destination':'int'},
#        }
#
#with fiona.open(os.path.join(outputWS,'solution_18.shp'),'w','ESRI Shapefile', schema) as c:
#    for i in shortLine:
#        route = LineString(shortLine)
#        c.write({'geometry':mapping(route),
#                 'properties': {'origin':7,'destination':1},
#                 })


# set obstacle weight for simulation
obsL = []
obsW = 100000000

# create empty figure for simulation
fig = plt.figure()
fig.set_size_inches(14,7,True)
ax = fig.add_subplot(111)
patches = []
for i in agents:
    rotPos = i.shapePos()
    #color = i.color
    polygon = Polygon(rotPos[:,:2], color = 'b', fill = True, closed = True)
    patches.append(polygon)
p = PatchCollection(patches)
destL = []
for i in agents:
    destL.append(np.ndarray.tolist(i.dest))
destL = np.array(destL)

ax.plot(destL[:,0],destL[:,1], 'gv')

if len(obsL) > 0:
    ax.plot(obsL[:,0],obsL[:,1], 'ro')

ax.add_collection(p)
ax.add_collection(f)
ax.set_xlim([726678,769259])
ax.set_ylim([4556396,4586487])
text99 = ax.text(727000,4557000,'time: ', fontsize = 8)
del i

# initialize simulation
def init():
    patches = []
    p = PatchCollection(patches)
    ax.add_collection(p)
    text99.set_text('time: %s'%(0))
    return patches,




def simulate(i, agents, obsL, obsW,n_frames):
    patches = []
    agentsL = copy.copy(agents)
    for j in agents:
        if i < j.t_start:
            agentsL.remove(j)
        if j.goal == True:
            agentsL.remove(j)
    steps = []
    #for i in np.arange(0,100,1):   
    for l in agentsL:
        l.nextWpt()
        
        #l.velRestrict()
        l.M(agents,obsL,obsW)                                                   # calculate destination mass at every time step, agents are moving, sometimes we don't care about them and our agent list decreases
        att = l.F_att(agents, obsL, obsW)                               # calculate attractive force for each agent
        rep = l.F_rep_agn(agentsL, i)                                               # agent repulsive force
        obs = l.F_rep_obs(windFarms,obsW)                                            # obstacle repulsive force
        totForce = att + rep + obs                                              # sum total force felt by agent
        tot1Dir = totForce/np.linalg.norm(totForce)                             # calculate unit vector of total force vector
        l.delta_c = np.arctan2(tot1Dir[1],tot1Dir[0]) - l.psi

        RPS = l.RPScommand()
        #print (RPS)
        l.RPS = RPS
        l.surge()
        #print (l.u)
        l.move()   
#        if l.delta_c != 0.0:
#            fuck
            
        print ("time step: %s, Agent: %s, vel: %s, RPS: %s, dr: %s, r:%s, psi:%s, att: %s, rep: %s, obs: %s, crash:%s"%(i,l.ID,np.round(l.u,2),np.round(l.RPS,2),np.round(l.delta_c,2),np.round(l.r[0],2),np.round(l.psi[0],2),np.round(att,3),np.round(rep,3),np.round(obs,3),l.crash))              
        text99.set_text('time: %s'%(i))
        
        # second step, update and set patches       
        rotPos = l.shapePos()
        hsv1 = 0.4 * (l.u/l.desVel)
        hsv2 = 0.9
        hsv3 = hsv2
        hsv = (hsv1,hsv2,hsv3)
        colorArr = colors.hsv_to_rgb(hsv)
        color = colorArr
        polygon = Polygon(rotPos[:,:2],color = color, fill = True, closed = False)
        patches.append(polygon)  
        
        
        # update event log
        #steps.append((l.ID,i,str(att),str(rep),str(obs),str(tot1Dir),l.delta_c[0],RPS,l.u,str(l.prevPos[:2]),str(l.currentPos[:2]),l.voyageCounter))

        # write to time step log within agent
        l.time_step[i] = (l.ID,i,str(att),str(rep),str(obs),str(tot1Dir),l.delta_c[0],RPS,l.u,str(l.prevPos[:2]),str(l.currentPos[:2]),l.voyageCounter)

        #l.time_step_log(i,att,rep,obs,tot1Dir,RPS)
        if l.goal == True:
            agentsL.remove(l)
    #resultsDB[0].executemany('INSERT INTO timeStep VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',steps)
    #resultsDB[1].commit()
    for p in patches:
        ax.add_patch(p)

t0 = time.time()
ani = animation.FuncAnimation(fig, shippy.simulate, init_func = shippy.sim_init, frames = n_frames, interval = 100, fargs = (agents, obsL, obsW, n_frames)).save(os.path.join(outputWS,simName + '.mp4'))#, writer = writer, dpi = dpi)
plt.show()
t1 = time.time()
print ("animation complete in %s seconds, output saved"%(round(t1 - t0,3)))


