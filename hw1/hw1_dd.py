#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #1
#Februrary 14, 2019

__author__ = "Dustin Davis"


"""
I will ignore orbital angular momentum (noting that it is a conserved quantity) and will use Cartesian coordinates
since they are in a simpler form and the basis vectors remain constant (though polar coordinates are often a natural
choice for central potentials) (and the usual GR stuff about being sufficiently far and in a weak field and not worry
about curved metrics, etc).
"""

import numpy as np
import matplotlib.pyplot as plt

PERIEHLION = 8766107800000. #cm
APHELION = 524823895700000. #cm
APHELION_VELOCITY = 90000.0 #cm/s
SEMI_MAJOR_AXIS = 266795001700000. #cm
PERIOD = 27509.1291 * 86400. #days * seconds
ECCENTRICITY = 0.96714291
G = 6.674e-8 #cm3 g-1 s-2
M_SUN = 1.989e33 #g
M_COMET = 2.2e17 #g
#don't care about inclination, etc for this exercise


def cart_distance(x1,y1,x2=0,y2=0):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def potential_energy(x,y):
    """
    Gravitational only
    :return:
    """
    return -G * M_SUN * M_COMET/cart_distance(x,y)


def kinetic_energy(vx,vy):
    """
    linear only (ignoring angular)
    :return:
    """
    return 0.5 * M_COMET * (vx**2 + vy**2)


def explicit_euler(x=[APHELION],y=[0],vx=[0],vy=[APHELION_VELOCITY],dt=[PERIOD/1000.],orbits=10):
    """

    :param x: vector of x positions, length 1 s|t x[0] == initial x position
    :param y: vector of y positions, length 1 s|t x[0] == initial y position
    :param vx: vector of x velocities, length 1 s|t x[0] == initial x velocity
    :param vy: vector of y velocities, length 1 s|t x[0] == initial y velocity
    :param vy: vector of time steps, length 1 s|t x[0] == initial time step
    :param orbits: integer number of orbits to simulate
    :return: position vector (x,y,t) as 2D matrix
    """
    pass
    #todo: define sub steps here


def main():
    pass

if __name__ == '__main__':
    main()









