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

Also, using numerical pi, so there are small precision errors tha will creep in, but I am ignoring those as well.

Coordinate axis places the Sun at 0,0
"""

import numpy as np
import matplotlib.pyplot as plt

PERIEHLION = 8766107800000. #cm
APHELION = 524823895700000. #cm

#note: we are sensitive to the initial velocity (coupled with the time step size) as to whether the orbit will be
# bound or not as the error in energy increases
PERIEHLION_VELOCITY = 5500000. #cm/s
APHELION_VELOCITY = 100000.#90000.0 #cm/s

SEMI_MAJOR_AXIS = 266795001700000. #cm
PERIOD = 27509.1291 * 86400. #days * seconds
ECCENTRICITY = 0.96714291
G = 6.674e-8 #cm3 g-1 s-2
M_SUN = 1.989e33 #g
M_COMET = 2.2e17 #g
#don't care about inclination, etc for this exercise
AU = 1.496e13



def make_plots(x,y,k,u,t,title="",fn=None):
    plt.figure(figsize=(12,4))

    plt.suptitle(title)
    plt.subplot(121)
    plt.title("Position")
    plt.ylabel("[AU]")
    plt.xlabel("[AU]")
    plt.plot(x/AU,y/AU)
    plt.scatter(x[0]/AU,y[0]/AU,marker="x",s=20,c="green",zorder=9,label="Start")
    plt.scatter(x[-1]/AU, y[-1]/AU, marker="x", s=20, c="red",zorder=9,label="Stop")
    plt.scatter(0,0,marker='o',s=20,c='orange',zorder=9,label="Sun")
    plt.legend()


    plt.subplot(122)
    plt.title("Total Energy")
    plt.ylabel(r"$erg\ \times10^{29}$")
    plt.xlabel("Time [PERIOD]")
    plt.plot(t/PERIOD,(k+u)/1e29)

    if fn is not None:
        plt.savefig(fn)
    else:
        plt.show()
    plt.close('all')

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


def accel(x,y):
    """
    From Gravitational force only, always pointing toward the origin

    Just to be explicit, F/m = a ... so this is the same calculation as force, but w/o the comet mass

    :param x:
    :param y:
    :return:
    """

    theta = np.arctan2(x, y) - np.pi / 2 #-pi/2 so the 0 angle is along the +x axis
    f = -G * M_SUN  / (cart_distance(x, y) ** 2)
    ax = f * np.cos(theta)
    ay = -1. * f * np.sin(theta)

    return ax,ay


def time_ff(x,y):
    """
    Using kepler (1/2 orbit at 1/2 distance) (technically should be an integral, but this is close enough
    for a time step basis)
    :param x,y:
    :return:
    """

    return np.pi * np.sqrt(cart_distance(x,y)**3./(G*(M_SUN+M_COMET)))

def time_step(x, y, adaptive=False):
    """

    Note: not going to worry about a softening length (since only 1 pair and no collission)
    Using 0.1 as scaling factor on the time
    :param x:
    :param y:
    :param adaptive:
    :return:
    """
    if adaptive:
        #todo: make an adaptive time step
        return min(PERIOD/1000.,0.1 * time_ff(x,y))
    else:
        return PERIOD/1000.


def explicit_euler(x=[APHELION],y=[0],vx=[0],vy=[APHELION_VELOCITY],orbits=10, adaptive_time = False):
    """

    :param x: vector of x positions, length 1 s|t x[0] == initial x position
    :param y: vector of y positions, length 1 s|t x[0] == initial y position
    :param vx: vector of x velocities, length 1 s|t x[0] == initial x velocity
    :param vy: vector of y velocities, length 1 s|t x[0] == initial y velocity
    :param vy: vector of time steps, length 1 s|t x[0] == initial time step
    :param orbits: integer number of orbits to simulate
    :return: position vector (x,y,t) as 2D matrix
    """
    def next_velocity(x,y,vx,vy,dt):
        """

        :param x,y: current (nth) position
        :param vx,xy: current (nth) velocity
        :param dt:timestep
        :return: n+1 velocity in x and y
        """

        ax,ay = accel(x,y)
        next_vx = vx + dt*ax
        next_vy = vy + dt*ay

        return next_vx, next_vy

    def next_position(x,y,vx,vy,dt):
        """

        :param x:
        :param y:
        :param vx:
        :param vy:
        :param dt:
        :return:
        """

        next_vx, next_vy = next_velocity(x,y,vx,vy,dt)
        next_x = x + dt*next_vx
        next_y = y + dt*next_vy

        return next_x, next_y, next_vx, next_vy

    n = 0 #index
    time = [0.]
    while time[n] < orbits * PERIOD:
        dt = time_step(x[n],y[n],adaptive_time)
        time.append(time[n]+dt)
        next_x, next_y, next_vx, next_vy = next_position(x[n],y[n],vx[n],vy[n],dt)
        x.append(next_x)
        y.append(next_y)
        vx.append(next_vx)
        vy.append(next_vy)
        n+=1

    return np.array(x), np.array(y), np.array(vx), np.array(vy), np.array(time)
    #end explicit_euler



def main():


    #first explicit Euler
    x,y,vx,vy,t = explicit_euler(orbits=10.) #use all default values
    u = potential_energy(x,y)
    k = kinetic_energy(vx,vy)

    make_plots(x,y,k,u,t,"Explicit Euler")

    x = [APHELION]; y = [0]; vx = [0]; vy = [APHELION_VELOCITY];
    x, y, vx, vy, t = explicit_euler(x, y, vx, vy,orbits=10.,adaptive_time=True)  # use all default values
    u = potential_energy(x, y)
    k = kinetic_energy(vx, vy)

    make_plots(x, y, k, u, t, "Explicit Euler (Dynamic Time)")



if __name__ == '__main__':
    main()









