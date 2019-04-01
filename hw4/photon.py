"""
Photon class ... handles propogation, pathing, absorption, escape, etc of a photon

Unless otherwise stated:
 distances are in cm
 wavelengths are in microns
"""

import utilities as util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

RADIUS_CLOUD_CM =  util.pc2cm(100.0) #cm

class Position:
    # cartesian coords in cm 
    
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def radius(self):
        return np.sqrt(np.power(self.x,2.) + np.power(self.y,2.) + np.power(self.z,2.))

    def radius_pc(self):
        return util.cm2pc(self.radius())


class Photon:
    def __init__(self,wavelength,seed=None):
        self.w = wavelength #microns
        self.seed = seed
        self.status = 0 # -1 = absorbed, 0 = propogating, 1 = free (escaped)
        self.position = Position(0.,0.,0.)
        self.path = [] #list of positions, [-1] is the current position


        self.path.append(self.position)

    def is_free(self):
        """
        Check position to see if the photon has escaped.
        Non-destructive ... does not update position and always returns the same result for same
        position.
        :return: boolean
        """

        if self.position.radius() > RADIUS_CLOUD_CM:
            self.status = 1
            return True
        else:
            return False

    def is_absorbed(self):
        """
        Perform a random draw to see if this photon is absorbed at this event.
        Destructive ... can update the photon's status irreversibly and is non-deterministic.
        :return:boolean
        """

        #dummy ... simulate turning off dust
        #return False

        p = util.get_uniform_prob(seed=self.seed)

        if p < util.prob_absorb(self.w):
            self.status = -1
            return True
        else:
            return False


    def propagate(self):
        """
        Execute one propagation step. Results in one of:
            absorption, escape, or continue (scatter)

        :return:
        """

        #get a new direction, distance ... update position
        t,p = util.get_direction()
        r = util.get_interaction_dist()

        #this is relative to the current position
        x,y,z = util.sphere2cart(r,t,p)

        #add to current position to get absolute position
        x += self.position.x
        y += self.position.y
        z += self.position.z

        #update to the new position
        self.position = Position(x,y,z)
        self.path.append(self.position)

        #check first if escaped:
        if self.is_free():
            #we are done, photon is free
            return
        elif self.is_absorbed():
            #we are done, photon is destroyed
            return
        else:
            pass #still in the cloud


    def plot_path(self,fn=None):
        """

        :return:
        """

        ax = plt.axes(projection="3d")
        ax.set_xlabel("[pc]")
        ax.set_ylabel("[pc]")
        ax.set_zlabel("[pc]")

        title = "Photon Path (radius = %0.2f pc)" %(self.position.radius_pc())



        #make a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 100. * np.outer(np.cos(u), np.sin(v)) #100. pc
        y = 100. * np.outer(np.sin(u), np.sin(v))
        z = 100. * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color='k',alpha=0.1)

        #make the path
        x = [p.x for p in self.path]
        y = [p.y for p in self.path]
        z = [p.z for p in self.path]

        x = np.array(x)/  3.086e18
        y = np.array(y) / 3.086e18
        z = np.array(z) / 3.086e18

        ax.plot3D(x,y,z)

        side = max(100.,self.position.radius_pc())

        ax.set_xlim(-side,side)
        ax.set_ylim(-side, side)
        ax.set_zlim(-side, side)

        ax.scatter3D(0,0,0,c='green',marker='o',s=20) #? plot the central star?
        if self.status == -1:
            ax.scatter3D(util.cm2pc(self.position.x),
                         util.cm2pc(self.position.y),
                         util.cm2pc(self.position.z),c='red',marker='x',s=20)
            title += " -- Destroyed"
        elif self.status == 1:
            ax.scatter3D(util.cm2pc(self.position.x),
                         util.cm2pc(self.position.y),
                         util.cm2pc(self.position.z), c='orange', marker='o',s=20)
            title += " -- Escaped"

        ax.set_title(title)

        if fn is None:
            plt.show()
        else:
            plt.savefig(fn)