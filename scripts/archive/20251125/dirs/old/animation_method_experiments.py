# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 20:47:33 2023

@author: KNebiolo

Script Intent: Can you create a method within a class that starts and carries 
out an  animation?

animation script came from:
    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter12.04-Animations-and-Movies.html

"""

# import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# declare workspaces
outputWS = r"J:\2819\005\Calcs\ABM\Output"

class animate_test():
    ''' Python class object to test if we can have a animator within a method
    within a class'''
    def __init__(self,outputWS):
        self.outputWS = outputWS
        
    def run(self):
        # define metadata for movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='a red circle following a blue sine wave')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        
        # initialize background figure
        n = 1000
        x = np.linspace(0, 6*np.pi, n)
        y = np.sin(x)
        
        fig = plt.figure()
        
        # plot the sine wave line
        sine_line, = plt.plot(x, y, 'b')
        red_circle, = plt.plot([], [], 'ro', markersize = 10)
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        
        # Update the frames for the movie
        with writer.saving(fig, os.path.join(self.outputWS,"writer_test.mp4"), 100):
            for i in range(n):
                x0 = x[i]
                y0 = y[i]
                red_circle.set_data(x0, y0)
                writer.grab_frame()
                
test = animate_test(outputWS)
test.run()