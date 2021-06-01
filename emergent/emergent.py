# -*- coding: utf-8 -*-
# Library for self ABM
# Author: KPN
#------------------------------------------------------------------------------#

# Serve as repository for classes and modules

# Library will depend upon following modules:
'''
Please make sure your module libraries are up to date, this module depends upon:
numpy 1.8.1, pandas, shapely and scipy

'''
import pandas as pd
import numpy as np
import scipy.constants
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import LinearRing
import networkx as nx
from scipy.integrate import odeint
from scipy import interpolate
import sqlite3
import os
from osgeo import ogr
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.collections import PatchCollection
import scipy.constants
import time
import pickle
import copy
from scipy import interpolate
import networkx as nx
from osgeo import ogr
import fiona

G = scipy.constants.G

def dBase(outputWS,dbName):
    '''function creates an event log database for later analysis'''
    path = os.path.join(outputWS,dbName + '.db')
    # Create and connect to results database
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS route''')
    c.execute('''DROP TABLE IF EXISTS agent''')
    c.execute('''DROP TABLE IF EXISTS interaction''')    
    c.execute('''DROP TABLE IF EXISTS timeStep''')    
    c.execute('''DROP TABLE IF EXISTS surge''') 
    c.execute('''DROP TABLE IF EXISTS windFarms''')   
    c.execute('''CREATE TABLE route(agent_ID INTEGER, route TEXT)''')
    c.execute('''CREATE TABLE agent(agent_ID INTEGER, m REAL, dest TEXT, start TEXT, type TEXT, L REAL, B REAL, T REAL, Tprime REAL, Kprime REAL, desVel REAL)''')
    c.execute('''CREATE TABLE interaction(timeStamp INTEGER, own INTEGER, target INTEGER, ownPsi REAL, targetPsi REAL, qDir REAL, repLogic INTEGER, inertialDisp REAL, agentDist REAL, collisionDist Real, RPS_scen INTEGER, rep TEXT, voyage INTEGER, crash TEXT)''')
    c.execute('''CREATE TABLE timeStep(agent_ID INTEGER, timeStamp INTEGER, att TEXT, rep TEXT, obs TEXT, totDir TEXT, delta_c REAL, RPS REAL, u REAL, prev TEXT, curr TEXT, voyage INTEGER)''')
    c.execute('''CREATE TABLE surge(agent_ID INTEGER, u REAL, maxAgnScen INTEGER, maxObsScen INTEGER)''')
    c.execute('''CREATE TABLE windFarms(farmID INTEGER, centroid TEXT, perimeter REAL, area REAL)''')
    return c, conn

def rotMatrix(attitude):
    ''' rotation matrix function makes use of the self's attitude matrix describing
    rotation that takes {n} into {b}
    Perez, T., & Fossen, T. I. (2007). Kinematic models for manoeuvring and seakeeping of marine vessels. Modeling, Identification and Control, 28(1), 19-30.

    the inut, an attitude vector can be accessed with self.attitude
    '''
    return np.array([[np.cos(attitude[2])*np.cos(attitude[0]),-np.sin(attitude[2])*np.cos(attitude[1])+np.cos(attitude[2])*np.sin(attitude[0])*np.sin(attitude[1]),np.sin(attitude[2])*np.sin(attitude[1])+np.cos(attitude[2])*np.sin(attitude[1])+np.cos(attitude[2])*np.cos(attitude[1])*np.sin(attitude[0])],
                 [np.sin(attitude[2])*np.cos(attitude[0]),np.cos(attitude[2])*np.cos(attitude[1])+np.sin(attitude[1])*np.sin(attitude[0])*np.sin(attitude[2]),-np.cos(attitude[2])*np.sin(attitude[1])+np.sin(attitude[2])*np.cos(attitude[1])*np.sin(attitude[0])],
                  [-np.sin(attitude[0]),np.cos(attitude[0])*np.sin(attitude[1]),np.cos(attitude[0])*np.cos(attitude[1])]])

def obstacles(obsWS):
    '''Function create a dataframe of vertices for all obstacle shapefiles found wihtin a 
    workspace.  
    
    In order for the ABM to properly function, the shapefiles must be preprocessed. 
    
    The obstacle polygon must have a convex boundary and contain at least 3 fields 
    as written: 'direction', 'buffer' and 'type'.  
    
    For navigational channels, the 'direction' field describes the direction of traffic.  
    If an agent is an 'incoming' vessel, the 'outgoing' marked channels will act as
    an obstacle, while 'outgoing' vessels will not view the channel as an obstacle and 
    can travel through it.  When direction is 'both', the agent will always notice 
    the obstacle and avoid it.
    
    The buffer field is required for route.  The MCA recommends vessels stay at least 
    800 m from ocean renewable energy infrastructure.  Safety buffers are also 
    applied to 'land' features.  Route uses the buffer field to plan around obstructions
    while maintaining minimum safety distances.  
    
    The type field indicates tye type of obstruction, and can either take 'land',
    'WEA' or 'channel'. 
     
    The output of the function is a pandas dataframe, which will serve as an input for an agent class, 
    therefore each agent knows where all obstacles are regardless of their direction.'''
    
    # list files in input directory
    files = os.listdir(obsWS)    
    obstacles = []
    shapes = []
    for f in files: # we need to find the shapefiles first...
        if f.endswith('.shp'):
            shapes.append(f)    
    for s in shapes: # loop through shapefiles in directory and add to list
        fileName = os.path.join(obsWS,s)
        shp = ogr.Open(fileName,0)
        obstacles.append(shp)       
    columns = ['shape','direction','buff','type']
    obstacleDF = pd.DataFrame(columns = columns)
    for obs in obstacles:
        lyr = obs.GetLayer(0)
        for feat in lyr:
            pts = feat.GetGeometryRef()                                         # get points
            ring = pts.GetGeometryRef(0)                                        # why is the ring necessary
            points = ring.GetPointCount()                                       # seems redundant
            arr = []
            for i in range(points):                                           # are you kidding me, I have to loop through the ponts
                p = ring.GetPoint(i)
                lon = p[0]
                lat = p[1]
                arr.append([lon,lat])                                           # so many steps...
            poly = Polygon(arr)                                                 # shapely polygon
            direction = feat.GetField('Direction')                              # get fields
            buff = feat.GetField('Buffer')
            typ = feat.GetField('Type')
            row = pd.DataFrame([[poly,direction,buff,typ]], columns = columns)  # create row
            obstacleDF = obstacleDF.append(row)                                 # append to result
    obstacleDF.fillna(0,inplace = True)                                         # convert any nan to 0
    return obstacleDF

def initialStates (n,obstacles,origins,destinations,ships,travel_network,frames,outputWS = None):
    '''Function creates the initial state of each agent for a given model run.  
    By implementing this function, the modeler is assured that the initial states
    of each agent are random. 
    
    If the the modeler requires sequential model runs with the same initial 
    states, the dataframe produced by this fucntion can be saved to an output 
    workspace with the optional 'output' argument set to True, and supplied with
    an output workspace directory.  If no outputWS is returned, an error message 
    will appear.  
    
    Required Arguments:
        n = number of agents
        obstacles = pandas dataframe of obstacles, output of obstacles function
        origins = pandas dataframe of origins
        destinations = pandas dataframe of destinations
        ship = python dictionary with ship types (key) and their relative 
               proportions within th modeled system (value) 
    '''
        
    # create profiles ID's, start at zero to make iterating easy
    profiles = np.arange(0,n,1)
    
    # create array of ship types, can either be cargo or tanker
    shipTypes = np.random.choice(list(ships.keys()), size = n)#, p = ships.values())
    
    # create arrays for K-T indices, L,B,T and DWT - based on whether or not ship was a cargo or tanker
    Tprime = np.zeros(len(profiles),np.dtype(float))
    Kprime = np.zeros(len(profiles),np.dtype(float))
    L = np.zeros(len(profiles),np.dtype(float))
    B = np.zeros(len(profiles),np.dtype(float))
    T = np.zeros(len(profiles),np.dtype(float))
    change = np.zeros(len(profiles),np.dtype(float))
    DWT = np.zeros(len(profiles),np.dtype(float))
    V_des = np.random.uniform(17,20,n)

    for i in profiles:
        change[i] = np.random.choice([-1,1])
        if shipTypes[i] == 'Cargo':
            Tprime[i] = np.random.uniform(1.2,1.5,size = 1) #1.5, 2.5
            Kprime[i] = np.random.uniform(2.2,3.0,size = 1) #1.5, 2.0
            L[i] = np.random.uniform(226,330)
            '''we need a comprehensive database of ship parameters, these values suck and I feel like we can't properly calibrate'''
            if L[i] > 253:
                B[i] = np.random.uniform(44,60)
                T[i] = np.random.uniform(13,20)
                DWT[i] = np.random.uniform(800000,2000000)                          
            else:
                B[i] = np.random.uniform(24,40)
                T[i] = np.random.uniform(7,12)
                DWT[i] = np.random.uniform(200000,800000)   
        else:
            Tprime[i] = np.random.uniform(1.5,1.7,size = 1) # 3.0, 6.0
            Kprime[i] = np.random.uniform(2.7,3.2,size = 1) # 1.7, 3.0
            L[i] = np.random.uniform(226,330)
            if L[i] > 253:
                B[i] = np.random.uniform(44,60)
                T[i] = np.random.uniform(13,20)
                DWT[i] = np.random.uniform(1600000,2400000) 
            else:
                B[i] = np.random.uniform(24,40)
                T[i] = np.random.uniform(7,12)
                DWT[i] = np.random.uniform(200000,1600000)   
    
    # give 'em all random starting velocities                
    v0 = np.random.uniform(2,4,n)
    
    # alter start times - make sure agents don't get bunched up    
    t_start = np.round(np.linspace(0,frames-200,n),0)
       
    origin = np.random.choice(origins.OBJECTID.values,n)                       # create an origin for each agent, it get's a random choice of origin
    destination = []
    for i in origin:                                                           # however, it's destination is limited to one of the possible destinations we give it in the graph G
        if len(destinations) > 1:
            dests = list(nx.neighbors(travel_network,i))
            destination.append(np.random.choice(dests))
        else:
            destination.append(destinations.OBJECTID.values[0])
    # build data frame    
    dataframe = {'profileNo':profiles,'shipTypes':shipTypes,'Tprime':Tprime,'Kprime':Kprime,'L':L,'B':B,'T':T,'DWT':DWT,'V_des':V_des,'v0':v0,'change':change,'t-start':t_start,'origin':origin,'destination':destination}        
    df = pd.DataFrame.from_dict(dataframe,orient = 'columns')
    # join dataframe to XY coordinates of the origin and destination:
    df = df.merge(origins,how = 'left', left_on = 'origin', right_on = 'OBJECTID')
    df.drop(columns = ['Channel','OBJECTID','Direction'], inplace = True)
    df.rename(columns = {'X':'X0','Y':'Y0'}, inplace = True)
    df = df.merge(destinations,how = 'left', left_on = 'destination', right_on = 'OBJECTID')
    df.drop(columns = ['Channel','OBJECTID','Direction'], inplace = True)
    df.rename(columns = {'X':'XD1','Y':'YD1'}, inplace = True)   
    df['X0'] = df.X0 + np.random.normal(200,100,len(df))
    df['Y0'] = df.Y0 + np.random.normal(200,100,len(df))
    df['XD1'] = df.XD1 + np.random.normal(200,100,len(df))
    df['YD1'] = df.YD1 + np.random.normal(200,100,len(df))
    
    #print (df.head)
   
    if outputWS != None:
        df.to_csv(os.path.join(outputWS,'initialStates.csv'),index = False)
    return df  

class Ship():  # create a class object to describe a self agent
    '''A class object for a ship agent
    The class object describes and holds all of the attributes of a ship agent.
    During a time step, the simulation will update affected attributes, while some 
    remain stable throughout the simulation.
    
    The class object also contains all of the functions an agent needs to explore 
    its world and interact with other agents.  The functions also include write 
    methods to log files.  

    '''
    def __init__(self, profileNumber,profileData,dBase,route_obstacles = None, nav_obstacles = None):
        '''when class is intialized, feed length, weight, velocity (m/s),
        heading (radians), current position (pos1), and the agent's destination (goal)
        '''
        self.ID = str(profileNumber)                                            # agent identifier
        self.dest = np.array([profileData.loc[profileNumber]['XD1'],profileData.loc[profileNumber]['YD1']]) # agent destination
        if 'XD' in profileData:
            self.intDest = np.array([profileData.loc[profileNumber]['XD'],profileData.loc[profileNumber]['YD']]) # intermediate destination 
        else:
            self.intDest = 'none'
        self.xpos = profileData.loc[profileNumber]['X0']                        # set starting X position for movement algorithm
        self.ypos = profileData.loc[profileNumber]['Y0']                        # set starting Y position for movement algorithm
        #self.color = color                                                      # these should be in matplotlib color labels                            
        self.type = profileData.loc[profileNumber]['shipTypes']                # what's it gonna be?                                                # if you're a cargo vessel...
        self.Tprime = profileData.loc[profileNumber]['Tprime']                  # K-T indices
        self.Kprime = profileData.loc[profileNumber]['Kprime']
        self.L = profileData.loc[profileNumber]['L']                            # values taken from: https://en.wikipedia.org/wiki/Tanker_(ship)
        self.B = profileData.loc[profileNumber]['B']
        self.T = profileData.loc[profileNumber]['T']
        self.DWT = profileData.loc[profileNumber]['DWT']          
        self.t_start = profileData.loc[profileNumber]['t-start']       
        self.C_B = 0.80                                                         # block coefficient
        ''' need decent values for drag coefficient'''
        self.C_D = 0.10                                                         # drag coefficient
        self.d = 0.65 * self.T                                                  # propeller diameter                                              # propeller diameter
        #self.d = 0.35 * self.T                                                  # propeller diameter                                              # propeller diameter

        self.m = self.DWT * 907.185                                             # vessel mass converted from DWT measurement        
        self.K_t = 1.2                                                          # propeller thrust coefficient NEEDS TO BE CALIBRATED???
        self.A = ((self.L * self.B) + (2 * (self.B * self.T)) + (2 * (self.L * self.T))) * self.C_B       # wetted area (m^2) useful for resistance
        self.delta_c = np.array([0.0])                                          # initial rudder heading
        self.r = np.array([0.0])                                                # initial rotational velocity
        self.currentPos = np.array([self.xpos,self.ypos,0])                     # N, E, D coordinates in array format
        self.u = profileData.loc[profileNumber]['v0'] * 0.514444                # get the initial velocity in m/s, convert from knots
        self.startVel = profileData.loc[profileNumber]['v0'] * 0.514444
        self.desVel = profileData.loc[profileNumber]['V_des']
        self.openWaterDesVel = profileData.loc[profileNumber]['V_des']
        if route_obstacles is None: # obstacles for route planning - buffered and simplified in a GIS
            self.route_obstacles = []
        else:
            self.route_obstacles = route_obstacles
        if nav_obstacles is None: # obstacles as mapped into a GIS - not buffered and not simplified - actual GIS data
            self.nav_obstacles = []
        else:
            self.nav_obstacles = nav_obstacles
        self.origDesVel = self.desVel
        self.RPS = 5                                                            # current RPS setting (currently set at 300 RPM)
        self.rho = 1029                                                           # density of seawater kg/m^3
        self.collide = 0
        self.c = dBase[0]
        self.conn = dBase[1]
        self.voyageCounter = 1
        # agent background
        self.c.execute("INSERT INTO agent VALUES(%s,%s,'%s','%s','%s',%s,%s,%s,%s,%s,%s)"%(self.ID,self.m,str(self.dest),str(self.currentPos),str(self.type),self.L,self.B,self.T,self.Tprime,self.Kprime,self.desVel))
        self.conn.commit()
        self.crash = False

        self.goal = False
        self.delta_max = np.radians(35)                                                  # maximum rudder differential 
        self.inertialStopFunc = self.inertialStop(1)
        self.time_step = {}
    
    def velRestrict(self):
        pos = self.currentPos[:2]
        # if the agent is within the mixer... this should be fixed with a polygon and shapely at a later date
        if 597171 < pos[0] < 605117 and 4475972 < pos[1] < 4480975:
            self.desVel = 10
        elif pos[0] < 605117 and pos[1] > 4475026:
            self.desVel = 10            
        #elif pos[0] < 612749 and pos[1] > 4467321:
        #    self.desVel = 10
        else:
            self.desVel = self.openWaterDesVel
    
    size = 50                                                                   # this will make my triangles bigger
    nullShape = np.array([[0,size,0],[- 0.5 * size,-1 * size,0],[0.5 * size,-1 * size,0]])  # null is the outline of the vessel, this can get way more complicated, for now we have a triangle                                                                     # surge velocity (m/s)
                    
    maxRPS = 30                                                                 # maximum propeller shaft RPS

    def nextWpt(self):
        '''Function identifies the next waypoint for each agent depending upon its current location.
        Because the likelihood of an agent actually hitting a waypoint (single point in space),
        the agent only has to get within 100 m of the point, before the next waypoint becomes the new destination'''
        c = Point(np.array(self.currentPos[:2]))
        wpt = Point(self.wpt)
        pt = (self.wpt[0],self.wpt[1])
        if c.distance(wpt) <= 50:
            idx = self.wpts.index(pt)
            if wpt != Point(self.dest):
                self.wpt = self.wpts[idx + 1]    
            
    def Route(self):
        '''Function identifies shortest route around obstacles, using a Euclidean Shortest Path
        algorithm modified from Hong & Murray (2013)
        '''
        if len(self.route_obstacles) > 0:
            wpts = nx.Graph()
            A = Point(self.currentPos[:2])
            B = Point(self.dest)
            farms = []
            for row in self.route_obstacles.iterrows():
                farm = row[1]['shape']
                
                #buff = row[1]['buff']
                #farm = farm.buffer(buff).simplify(0.4, preserve_topology=False)
                farms.append(farm)                                                 # append simplified farm to list 
            
            wpts.add_node(list(A.coords)[0])                                       # add origin and destination to G*
            wpts.add_node(list(B.coords)[0])
    
            AB = LineString([A,B])                                                 # create the line segment AB
            # Step 2, if AB crosses obstacle polygon find G* if not add AB to G* we are done
            testList = []                                                          # list of nodes to test
            tested = []                                                            # list of nodes already tested 
            tested_edge = []
            crossList = []                                                         # list of polygons crossed by the current tested edge
            
            # if original segment crosses a polygon
            for poly in farms:                                                     # for each polygon in our list
                if AB.crosses(poly):                                                # if AB crosses it
                    crossList.append(poly)                                          # append it to a cross list

                    
                
            
            # if AB does not cross any polygon, the shortest path to the origin is a straight line
            if len(crossList) == 0:                                                
                wpts.add_edge(list(A.coords)[0],list(B.coords)[0],dist = AB.length)
                print ("AB does not cross any obstacle, G* consists of: {0}".format(wpts))
            
            # if the edge crosses at least 1 polygon, we need to build G*
            else:                                                                  # if it doesn't we're fucked
                print ("AB crosses at least one obstacle, test all obstacle vertices")
                
                # Part 1, build and test a set of edges from the origin to every obstacle vertex
                i = A                                                              # part 1, A becomes i
                jList = []
                for p in crossList:                                                # for every polygon in the cross list
                    verts = list(p.exterior.coords)                                # extract exterior coordinates, these are nodes to test
                    for v in verts:
                        jList.append(v)                                            # append their vertices to j list
                jList = list(set(jList))
                del p
                testList = []
                for j in jList:                                                    # for every j in jlist:
                    j = Point(j)
                    ij = LineString([i,j])                                         # create a line segment from i to j
                    test = True                                                    # if true, line segment does not intersect with any other polygon, we add this edge to the graph
                    for p in crossList:
                        if ij.crosses(p) or ij.within(p):                                          # if ij does not cross this polygon, or if a polygon does not contain ij
                            test = False
                            print ("Ai crosses an obstacle")
                            break
                    if test == True:
                        wpts.add_nodes_from([list(j.coords)[0]])
                        wpts.add_edge(i.coords[0],j.coords[0],dist = ij.length)
                        testList.append(j)                                          # let's add j to the test list, becomes i in next round
                        print ("Ai does not cross any obstacle, waypoint {0} added to G*".format(j.coords[:][0]))
                del i, j, jList, p, v, verts, crossList
                while testList:                                                    # while test list is not empty
                    for i in testList:
                        crossList = []
                        iB = LineString([i,B])                                      # create line segment to B
                        for poly in farms:                                         # for every polygon, if iB happens to cross it, add it to the cross list
                            if iB.crosses(poly) or iB.within(poly):
                                crossList.append(poly)
                        del poly
                        if len(crossList) == 0:                                     # if it turns out iB doesn't cross a polygon we have a winner, add it to G*
                            wpts.add_edge(i.coords[0],B.coords[0],dist = iB.length)
                            print ("iB does not cross any obstacle, an edge to the destination has been found and added to G*")
                        else:
                            jList = []
                            for p in crossList:                                    # for every polygon in the cross list
                                verts = list(p.exterior.coords)
                                for v in verts:
                                    iv = LineString([i,v])
                                    if iv not in tested_edge:
                                        jList.append(v)                                # append its vertices to the jlist
                            jList = list(set(jList))
                            del p, verts, v
                            for j in jList:                                        # for every j in jlist:
                                j = Point(j)
                                ij = LineString([i,j])                              # create a line segment from i to j
                                tested_edge.append(ij)
                                test = True
                                for p in farms:
                                    if ij.crosses(p) or ij.within(p):              # if ij does not cross a polygon, or if a polygon does not contain ij
                                        test = False
                                        print ("ij crosses an obstacle")
                                if test == True:
                                    wpts.add_nodes_from([list(j.coords)[0]])
                                    wpts.add_edge(i.coords[0],j.coords[0],dist = ij.length)
                                    print ("ij does not cross an obstacle, edge added to G*")
                                if j not in tested:
                                    testList.append(j)                             # let's add j to the test list, becomes i in next round
                        tested.append(i)
                        testList.remove(i)
                        del crossList                                                

        else:
            wpts = nx.Graph()
            A = Point(self.currentPos[:2])
            B = Point(self.dest)
            AB = LineString([A,B])                                                 # create the line segment AB
            wpts.add_node(list(A.coords)[0])                                       # add origin and destination to G*
            wpts.add_node(list(B.coords)[0])
            wpts.add_edge(list(A.coords)[0],list(B.coords)[0],dist = AB.length)
            
        print ("G* built, consisting of waypoints:{0}".format(list(wpts.nodes)))
        short = nx.dijkstra_path(wpts,list(A.coords)[0],list(B.coords)[0],weight = "dist")
        self.short_route = short
        self.c.execute("INSERT INTO route VALUES(%s,'%s')"%(self.ID,str(short)))
        self.conn.commit()
        print ("Shortest route for agent {0} found".format(self.ID))
        print ("Agents route is along the following waypoints:{0}".format(np.round(short,0)))
        self.wpts = short   
        if len(self.wpts) < 2:
            self.wpt = self.wpts[0]
        else:
            self.wpt = self.wpts[1]                                                 # identify the first waypoint
        posVec = (self.wpt - self.currentPos[:2])                               # what is the position vector to the way point from the vessel's current position
        posDir = posVec/np.linalg.norm(posVec)                                  # unit vector describing direction of target vessel relative to own
        heading = np.arctan2(posDir[1],posDir[0])                               # heading in radians
        self.prevPos = np.array([self.currentPos[0] - np.cos(heading) * self.u,self.currentPos[1] - np.sin(heading) * self.u,0.0]) # create a previous position for the sake of the simulation
        self.psi = np.array([heading])                                      # what is the heading in radians


    def M(self, agents, obs, obsW):
        '''destination must be super massive so that it always has an attractive pull on
        the own agent unless it is extremely close to another agent.  Simply 100
        times the sum of all agent masses within an affected region
        '''
        mList = []
        for i in agents:
            mList.append(i.m)
        for i in obs:
            mList.append(obsW)
        exp = len(agents) + len(obs)
        return np.sum(mList)

    def attitude(self):
        '''attitude method is called and is a function of the class itself.  After
        movement model concludes, psi is updated (model limited heading) and should
        not equal delta_c (collision avoidance command heading) unless it is within
        a physically realistic turning radius.
        '''
        return np.array([0,0,self.psi + np.pi/2 *-1])

    def shapePos(self):
        '''  When a ship agent is initialized, the original orientation is given due north,
        The simulation will update shapePos whenever this method is called,
        the vessel shpae will rotate according to model limited heading (psi).
        The resultant shape position is rotated according to attitude.
        '''
        # columns: X, Y , Z
        # rows: Bow, Port Stern, Starboard Stern
        current = np.array([[self.currentPos[0], self.currentPos[1] + self.size, self.currentPos[2]],
                         [self.currentPos[0] - self.size * 0.5, self.currentPos[1] - self.size,self.currentPos[2]],
                         [self.currentPos[0] + self.size * 0.5, self.currentPos[1] - self.size,self.currentPos[2]]])
        delta = current - self.nullShape

        rotPos = np.zeros(current.shape)

        rot = rotMatrix(self.attitude())
        # for now this is an ineficient method for rotation, it would be better to use GPU processing
        # for this application
        for j in np.arange(len(rotPos)):
            rotPos[j] = rot.dot(current[j] - delta[j]) + delta[j]               # need to translate to null position location (around origin)
            #rotPos[j] = ref.dot(rotPos[j])
        return rotPos       
        
    def inertialStop(self,delta):
        ''' Function wraps the intertial stop velocity function into a numpy vectorized
        function that applies intertialStopVel over an array of delta t's'''
        
        def inertialStopVel(t,m,v0,A,C_D):
            '''Function for the velocity of a vessel during inertial stop at time (t).
            During inertial stop there is no negative thrust from a reversal in gear, 
            therefore the only thing slowing the vessel down is drag.
            
            During a meeting on January 5, 2016, Dr. Meyer derived this formula with 
            Mathematica.  Kevin Nebiolo implemented it in Python on October 13, 2016 for use
            in an agent based model of the commercial shipping industry
            
            t = model time step
            m = mass of vessel
            v0 = initial velocity
            A = cross sectional area
            delta = change in seconds'''
            
            return (m * v0)/(m + A * C_D * t * v0)
        
        vfunc = np.vectorize(inertialStopVel)                                   # vectorize inertialStopVel  
        
        maxVel = self.openWaterDesVel
        velArray = np.arange(0,maxVel+2.0,0.5)
        dispArray = np.array([])  
        for i in velArray:            
        # apply vectorized function over a vector of dt's
            t = np.linspace(0,7200,7201)                                            # create a vector of model time steps
            vel_t = vfunc(t,self.m,i,self.A,self.C_D)                          # calculate velocity at time (t)
            disp_t = vel_t * delta                                                  # calculate displacement at time (t)
            try:
                t_index = np.where(vel_t < 2.0)[0][0]                               # find the index where the agent is pretty much stopped
            except:
                t_index = len(t) -1                                                 # if the agent never fully stopped, what was the last time index?
            t_at_0V_A = int(t[t_index])                                                 # what time step was that at?
            inertialDisp = sum(disp_t[:t_at_0V_A+1])
            dispArray = np.append(dispArray,inertialDisp)
        return interpolate.interp1d(velArray,dispArray,kind = 'cubic',bounds_error = False)   



    def F_att(self,agents,obs,obsW):
        '''
        Gravitational attraction between the agent and the destination.
        Note, M is supermassive and is equal to the double the sum of all object
        masses within the simulation

        Newton's law of gravitation specifies that the gravitational force exerted
        by a point mass M on an object is equal to (Meyer 2002):
            F = - (G*M*r_hat)/(magnitude r)**2

        Function Inputs:
            the method incporates class variables,
            agent list may change depending upon those within close proximity
        Works Cited:
        Meyer, T. H. (2002). Introduction to Geometrical and Physical Geodesy:
            Foundation of Geomatics. Redlands, CA: ESRI Press.
        '''
        r = (self.currentPos[0:2] - self.wpt)
        rhat = r/np.linalg.norm(r)
        #att = np.negative((G * self.M(agents,obs,obsW) * rhat)/(np.linalg.norm(r)**2))
        att = np.negative((G * self.M(agents,obs,obsW) * rhat))
        self.attForce = att
        return att

    def F_rep_agn(self, agents, time):
        '''
        Modified gravitational attraction function for repulsion from other self agents.
        Rather than having a negative, attractive force, repulsion is expressed as a
        positive force by multipying by -1
    
        Function uses case logic to identify if a target vessel is a collision threat.
    
        The function also scales the force by distance.  Without scaling the distance,
        repulsive force is only strong enough to repell a self when distances are close.
        This behavior is unsafe, therefore we scale this distance by sum of the length
        overall of the own and target vessel.
    
        Function Inputs:
            G = gravitational constant
            agents = list of agents within range
        
        Function also classifies each interaction, decides on the level of evasive 
        maneuvering required and writes this information to the event log.  
        
        Depending upon the interaction with the agent, they may need to slow down.  
        Some interactions come with high risk (RPS = 3) and the agent will apply full 
        reverse, while other interactions will only warrant inertial stop (RPS = 2).  
        When the agent does not need to slow down, RPS = 1
        
        RPS_scen:
            3: high risk, crash trajectory polygons overlap, RPS = full astern
            2: medium risk, inertial trajectory polygons overlap, RPS = 0
            1: low to no risk, vessel aims to achieve desired RPS
            
        Repulsive force logic is a coded value classifying the type of agent-agent
        interaction.  There can be multiple interaction types, each with their own 
        repulsive force logic. 
        
        repLogic:
            1: target-agent (q_curr) within the trajectory polygon of the own-agent, apply repulsive force
            2: agents are head on, apply repulsive force
            3: own-agent is in line with and behind target-agent, no repulsive force
            4: own-agent approaching the port side of the target-agent, apply repulsive force
            5: own-agent approaching the starboard side of the target-agent, no repulsive force 
            6: trajectory polygons of the own and target-agent do not over lap, no repulsive force
            7: target-agent is not within the 270 degree swath around own-agent, no repulsive force
            8: target-agent is greater than 5 km away, no repulsive force applied
            
        '''

        self.interactions = []                                                  # create list of interactions to add at end 
        self.RPS_rep_scen = []                                                  # RPS scenario dictionary that own agent will compile for interactions with every other agent
                                                                                # numerical value indicates risk level
                                                                                # 3 = high risk, crash trajectory polygons overlap, RPS = full astern
                                                                                # 2 = medium risk, inertial trajectory polygons overlap, RPS = 0
                                                                                # 1 = low to no risk, vessel aims to achieve desired RPS

        self.matchVelLogic = []
        self.matchVel = []
        repArr = []                                                             # create array to store repulsive forces generate by all agents in simulation
        agentsUnder2k = []
        if self.u > 0.0:
            for i in agents:
                if i.ID == self.ID:
                    rep = np.array([0.0,0.0])
                    repArr.append(rep)
                    self.RPS_rep_scen.append(1)
                else:              
                    p_curr = self.currentPos[:2]
                    p_prev = self.prevPos[:2] 
                    q_curr = i.currentPos[:2]                                       # current position of target-agent 
                    q_prev = i.prevPos[:2]
                    v = (q_curr - p_curr)                                           # vector of the target vessel relative to the own
#                    v = (p_curr - q_curr)                                           # vector of the target vessel relative to the own
                    v_prime = (p_curr - q_curr)                                     # vector of the own relative to the target vessel
                    v_hat = v/np.linalg.norm(v)                                     # unit vector desribing direction of own vessel relative to target
                    v_prime_hat = v_prime/np.linalg.norm(v_prime)                   # unit vector desribing direction of target vessel relative to own
                    v_hat_ang = np.arctan2(v_hat[1],v_hat[0])                       # direction in radians between ships
                    dist = np.linalg.norm(v)                                        # distance between ships
                    if dist <= 5 and self.goal == False:
                        actionLogic = 187                                           # you are dead
                        self.crash = True                                           # you have crashed, you are now dead
                        rep = (G * i.m * v_hat)/(dist**2)
                        #rep = np.array([0,0])                                       # there is no repulsive force to feel because you are dead, you can't feel anything
                        repLogic = 187                                              # you dead homie, screamin 187 on a motha fuckin agent
                        repArr.append(rep)                                          # append the repulsive force to an array of repulsive forces
                        s_o = np.array([0])
                        c = dist
                        RPS_scen = 1
                    # we don't care about most interaction
                    elif 20 < dist <= 5000:                                         # if the target agent is less than or equal to 3 km away
                        psi_o = p_curr - p_prev
                        psi_t = q_curr - q_prev
                        psi_o_prime = psi_o/np.linalg.norm(psi_o)             # unit vector direction own agent                                                                               # rotational velocity agent A - construct
                        psi_t_prime = psi_t/np.linalg.norm(psi_t)             #    unit vector direction target agent 
                        psi_o_ang = self.psi                                            # psi of own agent, heading in radians
                        tau_o = self.delta_max/5.0                                          # maximum rudder deflection angle own-agent
                        tau_t = i.delta_max/5.0                                             # maximum rudder deflection angle target-agent
                        #tau_o = 2                                          # maximum rudder deflection angle own-agent
                        #tau_t = 2 
                        s_o = self.inertialStopFunc(self.u).tolist()
                        s_t = i.inertialStopFunc(i.u).tolist()
                        agentsUnder2k.append(i.ID)
                        # We have three opporunities to say if action of wasting computational resources on assessing collision is worthwhile...
                        if psi_o_ang - np.radians(135) <= v_hat_ang <= psi_o_ang + np.radians(135):
                            actionLogic = 1
                            ########################################
                            # Step 1: Develop trajectory polygons #
                            #######################################
                            # calculate the inertial displaced positions of the own (p_s) and target agent (p_t)
                            p_s = p_curr + s_o * psi_o_prime                        # inertial displaced position of the own agent, total displacement s_o
                            q_s = q_curr + s_t * psi_t_prime                        # inertial displaced position of the target agent, total displacement s_o
                            
                            # calculate R_tA and R_tB
                            rot_p_tau_o = np.array([[np.cos(tau_o/2.0), -np.sin(tau_o/2.0)],[np.sin(tau_o/2.0),np.cos(tau_o/2.0)]])       # rotation matrix for agent A at current rate of rotation
                            rot_p_tau_t = np.array([[np.cos(tau_t/2.0), -np.sin(tau_t/2.0)],[np.sin(tau_t/2.0),np.cos(tau_t/2.0)]])       # rotation matrix for agent B at current rate of ratation
                            rot_s_tau_o = np.array([[np.cos(-1 * tau_o), -np.sin(-1 * tau_o)],[np.sin(-1 * tau_o),np.cos(-1 * tau_o)]])       # rotation matrix for agent A at current rate of rotation
                            rot_s_tau_t = np.array([[np.cos(-1 * tau_t), -np.sin(-1 * tau_t)],[np.sin(-1 * tau_t),np.cos(-1 * tau_t)]])       # rotation matrix for agent B at current rate of ratation
                            
                            p_s_p = p_curr + s_o * rot_p_tau_o.dot(psi_o_prime)     # inertial displaced, port-rotated position of the own agent
                            q_s_p = q_curr + s_t * rot_p_tau_t.dot(psi_t_prime)     # inertial displaced, port-rotated position of the target agent
                            p_s_s = p_curr + s_o * rot_s_tau_o.dot(psi_o_prime)     # inertial displaced, starboard-rotated position of the own agent 
                            q_s_s = q_curr + s_t * rot_s_tau_t.dot(psi_t_prime)     # inertial displaced, starboard-rotated position of the target agent
                            
                            # create trajectory polygons
                            if self.u == 0.0:
                                Psi_o = Point(p_curr).buffer(1000)
                            else:    
                                Psi_o = Polygon([p_curr,p_s_p,p_s,p_s_s])              # create a polygon of all A positions
                            if i.u == 0.0:
                                Psi_t = Point(p_curr).buffer(1000)
                            else:
                                Psi_t = Polygon([q_curr,q_s_p,q_s,q_s_s])              # create a polygon of all B positions
                            
                            ########################################################################
                            # Step 3 apply case logic to determine if repulsive force is required #
                            ####################################################################### 
                            if s_o > 0.0 and s_t > 0.0:
                                if Psi_o.intersects(Psi_t):
                                    # test for intersection
                                    Psi_c = Psi_o.intersection(Psi_t)
                                    p_0 = Point(p_curr)                                 # create shapely point for current position of own-agent
                                    c = p_0.distance(Psi_c)                             # calculate the distance from the own-agent to the polygon where trajectory polygons overlap
                                    # Type II interaction: the vessel's are head on
                                    if np.radians(170) < (i.psi - self.psi) <  np.radians(190) or np.radians(-170) < (i.psi - self.psi) <  np.radians(-190):                             
                                        repLogic = 2                                    # repulsive force logic - vessels are head on
                                        #rep = (G * i.m * v_hat)/((dist/(self.L + i.L))**2)      # apply repulsive force
                                        if c < 300:
                                            RPS_scen = 1                       # full astern thrust 
                                            self.matchVel.append(5.0)
                                            rep = (G * i.m * v_hat)/((dist/(self.L + i.L))**2)      # apply repulsive force

                                            #rep = (G * i.m * v_hat)/(dist**2)      # apply repulsive force
                                        elif 300 <= c < 1000:
                                            self.matchVel.append(5.0)
                                            RPS_scen = 1
                                            rep = np.array([0.0,0.0])      # apply repulsive force
                                        else:
                                            RPS_scen = 1
                                            self.matchVel.append(10.0)
                                            rep = np.array([0.0,0.0])      # apply repulsive force
                                        repArr.append(rep)                              # append repulsive force to array
                                    # Type I interaction, target agent is within the own agent's trajectory polygon 
                                    elif Point(q_curr).within(Psi_o):
                                        repLogic = 1                                    # repulsive force logic - target agent within 
                                        #rep = (G * i.m * v_hat)/((dist/(2 * (self.L + i.L)))**2)      # apply repulsive force
                                        if  c < 500:
                                            RPS_scen = 3                                    # full astern thrust 
                                            rep = (G * i.m * v_hat)/((dist/(self.L + i.L))**2)       # apply repulsive force
                                        elif 500 <= c < 1000:
                                            RPS_scen = 2                                    # full astern thrust
                                            rep = (G * i.m * v_hat)/(dist**2)       # apply repulsive force
                                        else:
                                            self.matchVel.append(5.0)
                                            RPS_scen = 1
                                            rep = np.array([0.0,0.0])      # apply repulsive force
                                        repArr.append(rep)                              # append repulsive force to array                                
                                        
                                    # Type III interaction: own agent is in line and behind the target agent 
                                    elif np.all(np.sign(psi_o_prime) == np.sign(v)) and psi_o_ang - np.radians(22.5) <= v_hat_ang <= psi_o_ang + np.radians(22.5):
                                        repLogic = 3                                    # own agent is in line and behind the target agent
                                        #rep = (G * i.m * v_hat)/((dist/(self.L + i.L))**2)      # apply repulsive force
                                        #rep = (G * i.m * v_hat)/((dist/s_o)**2)      # apply repulsive force

                                        if  c < 500:
                                            RPS_scen = 3                                    # full astern thrust   
                                        elif 500 <= c < 1000:
                                            RPS_scen = 2                                    # full astern thrus                                                                   
                                        else:
                                            self.matchVel.append(i.u)
                                            RPS_scen = 1
                                        rep= np.array([0.0,0.0])
                                        repArr.append(rep)                              # append repulsive force to array                                
                                                                            
                                    # Type IV interaction: own agent is approaching the port side of the target agent
                                    elif np.sign(psi_o_prime[0] * v_hat[1] - v_hat[0] * psi_o_prime[1]) < 0:  
                                        repLogic = 4                                    # own agent is approaching the port side of the target agent
                                        #rep = (G * i.m * v_hat)/((dist/(self.L + i.L)*2)**2)      # apply repulsive force
                                        rep = (G * i.m * v_hat)/((dist/(self.L + i.L))**2)      # apply repulsive force
                                        if  c < 500:
                                            RPS_scen = 3                                    # full astern thrust   
                                        elif 500 <= c < 1000:
                                            RPS_scen = 2                                    # full astern thrus                                                                   
                                        else:
                                            self.matchVel.append(10.0)
                                            RPS_scen = 1
                                        repArr.append(rep)                              # append repulsive force to array                                

                                    # Type V interaction: own agent is approaching the starboard side of the target agent                            
                                    else:           
                                        repLogic = 5                                    # own agent is approaching the starboard side of the target agent
                                        rep = np.array([0.0,0.0])                           # stand on vessel does not feel repulsive force
                                        RPS_scen  = 1                                   # the agent solves for RPS to maintain desired velocity
                                        repArr.append(rep)                                        
                                else:
                                    repLogic = 6                                        # trajectory polygons of the own and target-agent do not overlap, no repulsive force applied               
                                    RPS_scen = 1
                                    rep = np.array([0.0,0.0])
                                    repArr.append(rep)
                                    c = s_o
                            elif s_o > 0.0 and s_t == 0.0: 
                                if Point(q_curr).within(Psi_o):
                                    p_0 = Point(p_curr)                                 # create shapely point for current position of own-agent
                                    c = p_0.distance(Point(q_curr))                             # calculate the distance from the own-agent to the polygon where trajectory polygons overlap
                                    repLogic = 1                                    # repulsive force logic - target agent within 
                                    rep = (G * i.m * v_hat)/(dist**2)      # apply repulsive force
                                    #rep = (G * i.m * v_hat)/((dist/s_o)**2)      # apply repulsive force
                                    if c < 0.05 * s_o:
                                        RPS_scen = 3                                    # full astern thrust                                                                      
                                    elif 0.05 * s_o >= c < 0.75 * s_o:
                                        RPS_scen = 2
                                    else:
                                        self.matchVel.append(i.u)
                                        RPS_scen = 1                                                                  
                                    repArr.append(rep)                              # append repulsive force to array
                                else:
                                    repLogic = 6                                        # trajectory polygons of the own and target-agent do not overlap, no repulsive force applied               
                                    RPS_scen = 1
                                    rep = np.array([0.0,0.0])
                                    repArr.append(rep)
                                    c = s_o
                            else:
                                repLogic = 99                                        # trajectory polygons of the own and target-agent do not overlap, no repulsive force applied               
                                RPS_scen = 1
                                rep = np.array([0.0,0.0])
                                repArr.append(rep)
                                c = s_o                            
                        else:                                                      
                            actionLogic = 0
                            repLogic = 7                                            # target-agent is not within the 270o swath around the own-agent, no repulsive force applied
                            rep = np.array([0.0,0.0])                                   
                            RPS_scen  = 1 
                            repArr.append(rep)
                                            
                    else:                                                                             
                        actionLogic = 0
                        repLogic = 8                                                # target-agent is greater than 5 km away, no repulsive force applied
                        rep = np.array([0.0,0.0])                                       # there is no repulsive force from this agent
                        RPS_scen  = 1                                               # if there are no 2's or 3's the vessel maintains course and desired velocity
                        repArr.append(rep)                                          # append the repulsive force for this agent to the final array
                    
                    self.RPS_rep_scen.append(RPS_scen) 
                    # Write interaction data to event log
                    if actionLogic == 0 or actionLogic == 2:
                        self.interactions.append((time,self.ID,i.ID,self.psi[0],i.psi[0],v_hat_ang,repLogic,0,dist,0,RPS_scen,np.array2string(rep),self.voyageCounter,self.crash))            
                    else:
                        self.interactions.append((time,self.ID,i.ID,self.psi[0],i.psi[0],v_hat_ang,repLogic,float(s_o),dist,c,RPS_scen,np.array2string(rep),self.voyageCounter,self.crash))
                                                #0    1       2      3         4           5     6           7       8   9   10      11           12              13                                   self.RPS_rep_scen.append(RPS_scen)
            self.c.executemany('INSERT INTO interaction VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',self.interactions)
            self.conn.commit()
            
            repArr = np.nan_to_num(np.array(repArr))                                # convert nans to numbers
            self.agnRepArr = np.sum(repArr,axis = 0)
            return np.sum(repArr, axis = 0)                                        # returnt he sum of the repulsive force to the simulation
            
        else:
            self.agnRepArr = [0.0,0.0]
            return [0.0, 0.0]

    def F_rep_obs(self, obs, obsW):
        '''
        Modified gravitational attraction function for repulsion from other obstructions.
        Rather than having a negative, attractive force, repulsion is expressed as a
        positive force by multipying by -1
    
        Function uses case logic to identify if an obstruction is a collision threat.
    
        The function also scales the force by distance.  Without scaling the distance,
        repulsive force is only strong enough to repell an agent when distances are close.
        This behavior is unsafe, therefore we scale this distance by dividing the 
        collision distance by the inertial stop distance.
    
        Function Inputs:
            G = gravitational constant
            agents = list of agents within range
        
        Function also classifies each reaction, decides on the level of evasive 
        maneuvering required and writes this information to the event log.  
        
        Depending upon the reaction to the obstacle, they may need to slow down.  
        Some interactions come with high risk (RPS = 3) and the agent will apply full 
        reverse, while other interactions will only warrant inertial stop (RPS = 2).  
        When the agent does not need to slow down, RPS = 1
        
        RPS_scen:
            3: high risk, crash trajectory polygons overlap, RPS = full astern
            2: medium risk, inertial trajectory polygons overlap, RPS = 0
            1: low to no risk, vessel aims to achieve desired RPS
                  
        '''

        self.RPS_obs_scen = []                                                  # RPS scenario dictionary that own agent will compile for interactions with every other agent
                                                                                # numerical value indicates risk level
                                                                                # 3 = high risk, crash trajectory polygons overlap, RPS = full astern
                                                                                # 2 = medium risk, inertial trajectory polygons overlap, RPS = 0
                                                                                # 1 = low to no risk, vessel aims to achieve desired RPS
        
        repArr = []
        #if len(obs) > 0:
        for row in self.nav_obstacles.iterrows():
            if row[1]['type'] == 'land' or row[1]['type'] == 'WEA':
                if self.u > 1.0:
                    p_curr = self.currentPos[:2]                         # create a point of the agent's current position
                    p_prev = self.prevPos[:2] 
                    psi_o = p_curr - p_prev               
                    p_currPoint = Point(p_curr)                         # create a point of the agent's current position
#                    omega = row[1]['shape'].centroid
                    obsLine = LinearRing(row[1]['shape'].exterior.coords)       # boundary of polygon as line                   
                    omega = obsLine.interpolate(obsLine.project(p_currPoint))        # identify the nearest point, omega       
#                    sigma = np.array(list(omega.coords))[0] - np.array(list(p_currPoint.coords))[0]  # vector describing the position of the own agent from omega 
                    sigma = np.array(list(p_currPoint.coords))[0] - np.array(list(omega.coords))[0]  # vector describing the position of the own agent from omega 
                    obsDist = p_currPoint.distance(row[1]['shape'])
                    
                    if obsDist <= 500: 
                        self.matchVel.append(3.0)
                    elif 500 < obsDist <= 1000:
                        self.matchVel.append(5.0)
                    elif 1000 < obsDist <= 2000:
                        self.matchVel.append(8.0)
                    elif 2000 < obsDist <= 3000:
                        self.matchVel.append(10.0)                    
                    
                    sigmahat = sigma/np.linalg.norm(sigma)                      # unit vector desribing direction of own vessel relative to the obstacle
                    sigmaNorm = np.linalg.norm(sigma)
                    Omega = row[1]['shape']                                     # obstacle
                    coll_dist = p_currPoint.distance(Omega)                          # distance to obstacle
                    s_o = self.inertialStopFunc(self.u).tolist()
                    #s_o = s_o / 2.0
                    if (psi_o == np.array([0.0,0.0])).all():
                        Vx = np.cos(self.psi)
                        Vy = np.sin(self.psi)
                        psi_o_prime = np.arctan2(Vy, Vx)
                    else:
                        psi_o_prime = psi_o/np.linalg.norm(psi_o)             # unit vector direction own agent 
                    
                    tau_o = 0.05
                    # calculate the inertial displaced positions of the own (p_s) and target agent (p_t)
                    p_s = p_curr + s_o * psi_o_prime                        # inertial displaced position of the own agent, total displacement s_o
                    
                    # calculate R_tA and R_tB
                    rot_p_tau_o = np.array([[np.cos(tau_o), -np.sin(tau_o)],[np.sin(tau_o),np.cos(tau_o)]])       # rotation matrix for agent A at current rate of rotation
                    rot_s_tau_o = np.array([[np.cos(-1 * tau_o), -np.sin(-1 * tau_o)],[np.sin(-1 * tau_o),np.cos(-1 * tau_o)]])       # rotation matrix for agent A at current rate of rotation
                    
                    p_s_p = p_curr + s_o * rot_p_tau_o.dot(psi_o_prime)     # inertial displaced, port-rotated position of the own agent
                    p_s_s = p_curr + s_o * rot_s_tau_o.dot(psi_o_prime)     # inertial displaced, starboard-rotated position of the own agent 
                    
                    # create trajectory polygons
                    Psi_o = Polygon([p_curr,p_s_p,p_s,p_s_s])              # create a polygon of all A positions
                    if Psi_o.area > 0.0:
                        if Psi_o.intersects(Polygon(row[1]['shape'])):
                            if coll_dist <= 5:
                                self.crash = True 
                                rep = np.array([0.0,0.0])
                                repArr.append(rep)
                            elif 5 < coll_dist <= 1000:
                                rep = (G * obsW * sigmahat)/((coll_dist/(5 * self.L))**2)
                                self.RPS_obs_scen.append(3)
                                repArr.append(rep) 
                            elif 1000 < coll_dist <= 2000:
                                #rep = (G * obsW * sigmahat)/(coll_dist**2)
                                rep = np.array([0.0,0.0])
                                self.matchVel.append(2.0)
                                self.RPS_obs_scen.append(1)
                                repArr.append(rep)
                            elif 2000 < coll_dist < 5000:
                                #rep = (G * obsW * sigmahat)/(coll_dist**2)
                                rep = np.array([0.0,0.0])
                                self.matchVel.append(5.0)
                                self.RPS_obs_scen.append(1)
                                repArr.append(rep)
                            else:
                                repArr.append(np.array([0.0,0.0]))
                        else:
                            repArr.append(np.array([0.0,0.0]))
                    else:
                        repArr.append(np.array([0.0,0.0]))
                else:
                    repArr.append(np.array([0.0,0.0]))

                
        repArr = np.nan_to_num(np.array(repArr))
        self.obsRepArr = np.sum(repArr,axis = 0)
        return np.sum(repArr,axis = 0)

    def RPScommand(self):
        '''This function returns the RPS command based upon the interactions with
        other agents and obstacles during a time step.  The agent collects information
        during interactions with all agents and each obstacle that is less than 1 km
        away.

        If any interaction is high risk then the return is full astern
        If any interaction is medium risk then the return is RPS = 0 for inertial stopping
        Otherwise the agent solves for RPS after determining the desired acceleration.
        '''
        def resistance(rho,u,C_D,A):
            '''formula for vessel resistance where:
            C_D = drag coefficient
            A = vessel wetted area
            u = agent's current velocity
            rho = density of seawater
            '''
            return 0.5 * rho * u**2 * C_D * A

        risk = self.RPS_rep_scen + self.RPS_obs_scen
        if np.any(np.equal(risk,np.repeat(3,len(risk)))):
            n = -5
        elif np.any(np.equal(risk,np.repeat(2,len(risk)))):
            n = 0
        else:
            R = resistance(self.rho, self.u,self.C_D, self.A)
            u_0 = np.nan_to_num(self.u)                                         # get agent's velocity
            if len(self.matchVel) > 0:
                u_1 = np.min(self.matchVel)
                if u_1 > self.openWaterDesVel:
                    u_1 = self.openWaterDesVel
            else:
                u_1 = self.desVel                                               # get desired velocity
            a_d = (u_1 - u_0)/120                                                # calculate desired acceration as the change in velocity over change in time
            '''I fail here, why?'''
            if np.sign(self.m * a_d + R) == -1:
                n = np.sqrt(((self.m * a_d + R)*-1)/(self.K_t * self.rho * self.d**4))   # solve for RPS
            else:
                n = np.sqrt((self.m * a_d + R)/(self.K_t * self.rho * self.d**4))   # solve for RPS
            if n > self.maxRPS:
                n = self.maxRPS
#            elif n < 0:
#                n = 0.0
        self.matchVel = []
        return n

    def surge(self):
        '''Surge function adopted from Ueng (2008), allows an agent to increase
        or decrease surge velocity as a function of vessel density

        '''
        def thrust(K_t,rho,n,d):
            '''formula for thrust where:
            K_t = propeller thrust coefficient
            RPS = agent's current RPS setting
            D = propeller diameter
            '''
            if np.sign(n) == 1:
                return K_t * rho * n**2 * d**4
            else:
                return (K_t * rho * n**2 * d**4) * -1


        def resistance(rho,u,C_D,A):
            '''formula for vessel resistance where:
            C_D = drag coefficient
            A = vessel wetted area
            u = agent's current velocity
            '''
            return 0.5 * rho * u**2 * C_D * A
        
        def V(dt,acc):
            return acc * dt
            
        risk = self.RPS_rep_scen + self.RPS_obs_scen      
        R = resistance(self.rho,self.u,self.C_D,self.A)                # drag force drag coefficient * wetted area * velocity squared
#        if np.round(self.delta_c,1) != 0.0:
#            if np.abs(self.delta_c[0]) > np.radians(35):
#                rudder_percent = 1.0
#            else:
#                rudder_percent = self.delta_c[0] / np.radians(35)
#            R = R + R ** rudder_percent
        T = thrust(self.K_t,self.rho,self.RPS,self.d)                      # thrust = thrust coefficient * RPS squared * propeller diameter to the 4th power
        if self.crash == True:
            self.u = 0.0
        else: 
#            if np.any(np.equal(risk,np.repeat(3,len(risk)))):
#                acc = ((-1 * thrust) - resistance)/self.m    
#            else:
#                acc = (thrust - resistance)/self.m                                      # acceleration is equal to the sum of surgeforces divided by the vessel's mass
            acc = (T - R)/self.m                                      # acceleration is equal to the sum of surgeforces divided by the vessel's mass

            dV = V(1,acc)
                
            if np.linalg.norm((self.currentPos[:2] - self.dest)) < 500:        # if the agent is within 500 m of their destination - it cycles back to the origin and starts over again
                self.goal = True
                self.voyageCounter = self.voyageCounter + 1
                self.currentPos = np.array([0,0,0]) 
                self.wpt = self.wpts[1]                                                 # identify the first waypoint
                posVec = (self.wpt - self.currentPos[:2])                               # what is the position vector to the way point from the vessel's current position
                posDir = posVec/np.linalg.norm(posVec)                                  # unit vector describing direction of target vessel relative to own
                heading = np.arctan2(posDir[1],posDir[0])                               # heading in radians
                self.prevPos = np.array([self.currentPos[0] - np.cos(heading) * self.u,self.currentPos[1] - np.sin(heading) * self.u,0.0]) # create a previous position for the sake of the simulation
                self.psi = np.array([heading])                                      # what is the heading in radians
                self.u = self.startVel
            else:
                self.u = self.u + dV                                               # the new scalar velocity is equal to the current scalar velocity plus the current scalar acceleration
            
            if self.u < 0.0:
                self.u = 0.0     
            
        self.RPS_rep_scen = []
        self.RPS_obs_scen = []
        
    def move(self):
        '''
        Movement functions based on Nomoto
        displacement function of:
        u = surge velocity/forward motion at t0 - will be the result of agent input next...
        psi = current heading at t0
        theta = command heading at t0
        K, T = Nomoto maneuverability indices

        delta t is intended to be 1 second, therefore there is no need to multiply by dt
        '''
        def dPsi(dt,r):
            return r * dt
        if self.u == 0.0:
            self.prevPos = self.currentPos                                          # previous position is now equal to the current position
            newX = np.array([self.currentPos[0]])
            newY = np.array([self.currentPos[1]])
            self.currentPos = np.zeros(3)
            self.currentPos = np.array([newX[0],newY[0],0])                         # set current position
            self.psi = self.psi                                    # set new vessel heading
            self.r = self.r                        # set new vessel rotational veloci                    
        else:    
            K = (self.Kprime * self.u)/self.L
            T = (self.Tprime * self.L)/self.u
    
            # command heading may be larger than maximum allowable rudder range
            if np.abs(self.delta_c) > self.delta_max:
                self.delta_c = np.sign(self.delta_c) * self.delta_max
            # start movement
            self.prevPos = self.currentPos                                          # previous position is now equal to the current position
            newX = np.array([self.currentPos[0] + self.u * np.cos(self.psi[0])])       # calculate New X
            newY = np.array([self.currentPos[1] + self.u * np.sin(self.psi[0])])       # calculate New Y
            self.currentPos = np.zeros(3)
            self.currentPos = np.array([newX[0],newY[0],0])                         # set current position
            self.psi = self.psi + dPsi(1,self.r)                                    # set new vessel heading            
            self.r = self.r + (K * (self.delta_c - self.r))/T                         # set new vessel rotational velocity
            # calculate a dampening force acting against the direction of motion r
            self.damp = self.r * -0.02
            #self.damp = np.sqrt(self.r**2 - (self.r/(2*self.m))**2)
            self.r = self.r + self.damp
            
            if self.goal == True:
                self.currentPos = np.array([0,0,0])
                
    def time_step_log(self,time_step,att,rep,obs,direction,RPS):
        '''function that logs results of a time step - because writing to sqlite 
        sucks for some reason'''
        try:
            self.time_log[time_step] = [self.ID,att[0],att[1],rep[0],rep[1],obs[0],obs[1],direction[0],direction[1],self.delta_c[0],RPS,self.u,self.prevPos[0],self.prevPos[1],self.currentPos[0],self.currentPos[1],self.voyageCounter]
        except AttributeError:
            self.time_log = dict()
            self.time_log[time_step] = [self.ID,att[0],att[1],rep[0],rep[1],obs[0],obs[1],direction[0],direction[1],self.delta_c[0],RPS,self.u,self.prevPos[0],self.prevPos[1],self.currentPos[0],self.currentPos[1],self.voyageCounter]
            
def selfRotation(rotMatrix, coord):
    return rotMatrix.dot(coord)



class simulation():
    '''Python object class that initializes and then runs a shippy simulation'''
    
    def __init__(self,proj_dir,sim_name, n_frames, n_agents, obs, land, origins, destinations, random_profiles = True, profile_dir = None):
        '''function that initializes a simulation and writes patches consisting
        of land and obstacle polygons for plotting, and a list of agents to iterate
        over to the object class'''
        # identify workspaces and results database
        inputWS = os.path.join(proj_dir,'Data')
        outputWS = os.path.join(proj_dir,'Output')
        self.resultsDB = dBase(outputWS,sim_name)               
    
        # Create polygons for obstacles and land
        nodes = origins.OBJECTID.values.tolist()    
        
        # Create Destinations Dataframe
        destinations = pd.read_csv(os.path.join(inputWS,'destination.csv'))
        nodes.append(destinations.OBJECTID.values.tolist()[0])
        
        # Create Ships dictionary of ship class and their relative proportion
        ships = {'Cargo':0.750,'Tanker':0.250}

        # create a travel network
        travel_network = nx.DiGraph()
        travel_network.add_nodes_from(nodes)
        
        for index, row in origins.iterrows():
            _from = row['OBJECTID']
            for index2, row2 in destinations.iterrows():
                _to = row2['OBJECTID']
                travel_network.add_edges_from([(_from,_to)])
        del index, row, _from, _to
                
        # if this simulation uses random profiles create them, if not, import them
        if random_profiles == True:
            self.profiles = initialStates(n_agents,obs,origins,destinations,ships,travel_network,n_frames)

        # create agent list
        agents = []
        rows = np.arange(0,len(self.profiles) - (len(self.profiles)-n_agents),1)
        for i in rows:
            agn = Ship(i,self.profiles,self.resultsDB, obs,land)
            agn.Route()
            agents.append(agn)
        del agn
        
        # write wind farms to project database
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

        self.resultsDB[0].executemany('INSERT INTO windFarms VALUES (?,?,?,?)',farmRows)
        self.resultsDB[1].commit()    
        del i, farmRows, farmID
        
        # create patches for land and obstacles
        self.farms = []
        paths = []
        
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
            self.farms.append(polygon)
        del row, shape, typ, coords, polygon, color
        
        for i in obs.iterrows():
            if i[1]['type'] != 'land':
                farm = i[1]['shape']
                coords = list(farm.exterior.coords)
                polygon = Polygon(coords,color = 'r', fill = True, closed = True)
                self.farms.append(polygon)
        del i, farm, coords, polygon  
          
        self.f = PatchCollection(self.farms)

        # set obstacle weight for simulation
        obsL = []
        self.obsW = 100000000
        
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
        self.p = PatchCollection(patches)
        del i
        
        if len(obsL) > 0:
            ax.plot(obsL[:,0],obsL[:,1], 'ro')
        
        ax.add_collection(self.p)
        ax.add_collection(self.f)
        ax.set_xlim([726678,769259])
        ax.set_ylim([4556396,4586487])
        text99 = ax.text(727000,4557000,'time: ', fontsize = 8)
        
        self.fig = fig
        del i

def simulate(i, agents, obsW,n_frames):
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

