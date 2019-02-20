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

Time over which the orbit is simulated is 10x the analytical orbital period, not 10 orbits in each simulation (as the 
orbit may become unbound)
"""

import numpy as np
import matplotlib.pyplot as plt

PERIEHLION = 8766107800000. #cm
APHELION = 524823895700000. #cm

#note: we are sensitive to the initial velocity (coupled with the time step size) as to whether the orbit will be
# bound or not as the error in energy increases
PERIEHLION_VELOCITY = 5500000. #cm/s
APHELION_VELOCITY = 100000. #cm/s
SEMI_MAJOR_AXIS = 266795001700000. #cm
PERIOD = 27509.1291 * 86400. #days * seconds
ECCENTRICITY = 0.96714291
G = 6.674e-8 #cm3 g-1 s-2
M_SUN = 1.989e33 #g
M_COMET = 2.2e17 #g
#don't care about inclination, etc for this exercise
AU = 1.496e13

NUM_ORBITS = 10


def analytic_orbit():
    """
    Ellipse of comet orbit
    :return: x coords and the positive y-coords (quadrants 1 & 2), this is a symmetric closed orbit
    """

    x = np.linspace(-SEMI_MAJOR_AXIS,SEMI_MAJOR_AXIS,1000)
    a = SEMI_MAJOR_AXIS
    b = a * np.sqrt(1-ECCENTRICITY**2)
    py = b/a * np.sqrt((a**2) - (x**2))
    x = x + (SEMI_MAJOR_AXIS - PERIEHLION) #shift to deal with SUN at (0,0)
    return x,py


def make_plots(x,y,k,u,t,title="",fn=None):
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    #numerical orbit plot
    plt.suptitle(title)
    plt.subplot(221)
    plt.title("Position")
    plt.ylabel("[AU]")
    plt.xlabel("[AU]")
    plt.plot(x/AU,y/AU)
    plt.scatter(x[0]/AU,y[0]/AU,marker="x",s=20,c="green",zorder=9,label="Start")
    plt.scatter(x[-1]/AU, y[-1]/AU, marker="x", s=20, c="red",zorder=9,label="Stop")
    plt.scatter(0,0,marker='o',s=20,c='orange',zorder=9,label="Sun")
    #plt.ylim(ymin=-22.5, ymax=22.5)
    plt.legend()

    #overplot analytic solution (plot the ellipse with focus at 0,0)
    #as a "zoom in" centered on the orbit
    plt.suptitle(title)
    plt.subplot(222)
    plt.title("Position (zoom/centered)")
    plt.ylabel("[AU]")
    plt.xlabel("[AU]")
    plt.plot(x/AU,y/AU)

    orbit_x, orbit_y = analytic_orbit()
    plt.plot(orbit_x / AU, orbit_y / AU, color="red", ls=":")
    plt.plot(orbit_x / AU, -1 * orbit_y / AU, color="red", ls=":",label="Analytic Orbit")

    plt.scatter(x[0]/AU,y[0]/AU,marker="x",s=20,c="green",zorder=9,label="Start")
    plt.scatter(x[-1]/AU, y[-1]/AU, marker="x", s=20, c="red",zorder=9,label="Stop")
    plt.scatter(0,0,marker='o',s=20,c='orange',zorder=9,label="Sun")

    plt.xlim(xmin=-5,xmax=40)
    plt.ylim(ymin=-22.5, ymax=22.5)
    plt.legend()


    #total energy plot
    plt.subplot(223)
    plt.title("Energy")
    plt.ylabel(r"$erg\ \times10^{29}$")
    plt.xlabel("Time [PERIOD]")
    plt.plot(t/PERIOD,(k+u)/1e29,color='k',zorder=9,label="Sum")
    plt.plot(t / PERIOD, u / 1e29, color='blue',ls="solid",lw=2,alpha=0.5,label="Potential (U)")
    plt.plot(t / PERIOD, k / 1e29, color='red',ls="solid",lw=2,alpha=0.5,label="Kinetic (K)")
    plt.legend()

    # total energy plot
    plt.subplot(224)
    plt.title("Total (K+U) Energy")
    plt.ylabel(r"$erg\ \times10^{29}$")
    plt.xlabel("Time [PERIOD]")
    plt.plot(t / PERIOD, (k + u) / 1e29, color='k', zorder=9, label="Sum")


    if fn is not None:
        plt.savefig(fn)
    else:
        plt.show()
    plt.close('all')

def cart_distance(x1,y1,x2=0,y2=0):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def potential_energy(x,y):
    return -G * M_SUN * M_COMET/cart_distance(x,y)


def kinetic_energy(vx,vy):
    return 0.5 * M_COMET * (vx**2. + vy**2.)


def accel(x,y):
    """
    From Gravitational force only, always pointing toward the origin
    Just to be explicit, F/m = a ... so this is the same calculation as force, but w/o the comet mass
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
    """
    return np.pi * np.sqrt(cart_distance(x,y)**3./(G*(M_SUN+M_COMET)))


def time_step(x, y, adaptive=False):
    """
    Note: not going to worry about a softening length (since only 1 pair and no collission)
    Using 0.1 as scaling factor on the time
    """
    if adaptive:
        #make an adaptive time step based on free fall time
        return min(PERIOD/1000.,0.01 * time_ff(x,y))
    else:
        return PERIOD/1000.


def explicit_euler(x=[APHELION],y=[0],vx=[0],vy=[APHELION_VELOCITY],orbits=NUM_ORBITS, adaptive_time = False):
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
        ax,ay = accel(x,y)
        next_vx = vx + dt*ax
        next_vy = vy + dt*ay

        return next_vx, next_vy

    def next_position(x,y,vx,vy,dt):
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


def rk2b(x=[APHELION],y=[0],vx=[0],vy=[APHELION_VELOCITY],orbits=NUM_ORBITS, adaptive_time = False):
    """
    Slightly different organization, but the same results. A bit clearer to me in terms of coding

    basically
    w_n+1 = w_n + dt*w'_n + (dt)**2 * w''_n
    w'_n+1 = w'_n + dt*w''_n

    """

    def next_velocity(x,y,vx,vy,dt):
        ax,ay = accel(x,y)
        next_vx = vx + dt*ax
        next_vy = vy + dt*ay

        return next_vx, next_vy


    def next_position(x,y,vx,vy,dt):
        ax,ay = accel(x,y)

        next_x = x + dt*vx + 0.5*(dt**2)*ax
        next_y = y + dt*vy + 0.5*(dt**2)*ay

        return next_x, next_y

    n = 0 #index
    time = [0.]
    while time[n] < orbits * PERIOD:
        dt = time_step(x[n],y[n],adaptive_time)
        time.append(time[n]+dt)

        next_x, next_y = next_position(x[n],y[n],vx[n],vy[n],dt)
        next_vx, next_vy = next_velocity(x[n],y[n],vx[n],vy[n],dt)

        x.append(next_x)
        y.append(next_y)
        vx.append(next_vx)
        vy.append(next_vy)

        n+=1

    return np.array(x), np.array(y), np.array(vx), np.array(vy), np.array(time)
    #end rk2b


def leapfrog(x=[APHELION],y=[0],vx=[0],vy=[APHELION_VELOCITY],orbits=NUM_ORBITS, adaptive_time = False):
    # keeping with position and velocity instead of momentum, but that should not matter here
    # positions are on integer steps and velocities are shifted by 1/2 step

    def half_step_velocity(x,y,vx,vy,dt): #kick
        #kick step ... return the velocity of the n+1/2 step
        #based on 1/2 dt interval * acceleration at the current (passed in) step
        #note: 1st call will be the n position, 2nd call will be at the n+1 position
        ax,ay = accel(x, y)
        next_vx = vx + 0.5* dt * ax
        next_vy = vy + 0.5 *dt * ay

        return next_vx, next_vy


    def next_position(x,y,vx,vy,dt): #drift

        n_half_vx, n_half_vy = half_step_velocity(x,y,vx,vy,dt)  #kick

        next_x = x + dt*n_half_vx #drift
        next_y = y + dt*n_half_vy

        next_vx, next_vy = half_step_velocity(next_x,next_y,n_half_vx,n_half_vy,dt) #kick

        return next_x, next_y, next_vx, next_vy

    n = 0  # index
    time = [0.]
    while time[n] < orbits * PERIOD:
        dt = time_step(x[n], y[n], adaptive_time)
        time.append(time[n] + dt)

        next_x, next_y, next_vx, next_vy = next_position(x[n], y[n], vx[n], vy[n], dt)

        x.append(next_x)
        y.append(next_y)
        vx.append(next_vx)
        vy.append(next_vy)

        n += 1

    return np.array(x), np.array(y), np.array(vx), np.array(vy), np.array(time)
    #end leapfrog


def main():


    #######################
    #explicit Euler
    ######################
    x = [APHELION];    y = [0];    vx = [0];    vy = [APHELION_VELOCITY];
    x,y,vx,vy,t = explicit_euler(x, y, vx, vy,orbits=NUM_ORBITS) #use all default values
    u = potential_energy(x,y)
    k = kinetic_energy(vx,vy)

    make_plots(x,y,k,u,t,"Explicit Euler (Fixed Time Step)",fn="p1_fix_euler.png")

    #reset vectors (deal with context)
    x = [APHELION]; y = [0]; vx = [0]; vy = [APHELION_VELOCITY];
    x, y, vx, vy, t = explicit_euler(x, y, vx, vy,orbits=NUM_ORBITS,adaptive_time=True)  # use all default values
    u = potential_energy(x, y)
    k = kinetic_energy(vx, vy)

    make_plots(x, y, k, u, t, "Explicit Euler (Dynamic Time Step)", fn="p2_dyn_euler.png")


    ###################
    #RK2
    ###################
    x = [APHELION];    y = [0];    vx = [0];    vy = [APHELION_VELOCITY];
    x,y,vx,vy,t = rk2b(x, y, vx, vy,orbits=NUM_ORBITS) #use all default values
    u = potential_energy(x,y)
    k = kinetic_energy(vx,vy)

    make_plots(x,y,k,u,t,"RK2 (Fixed Time Step)",fn="p3_fix_fk2.png")

    # RK2
    x = [APHELION];    y = [0];    vx = [0];    vy = [APHELION_VELOCITY];
    x, y, vx, vy, t = rk2b(x, y, vx, vy, orbits=NUM_ORBITS,adaptive_time=True)  # use all default values
    u = potential_energy(x, y)
    k = kinetic_energy(vx, vy)

    make_plots(x, y, k, u, t, "RK2 (Dynamic Time Step)",fn="p4_dyn_fk2.png")


    ###################
    #Leapfrog
    ###################
    x = [APHELION];    y = [0];    vx = [0];    vy = [APHELION_VELOCITY];
    x,y,vx,vy,t = leapfrog(x, y, vx, vy,orbits=NUM_ORBITS) #use all default values
    u = potential_energy(x,y)
    k = kinetic_energy(vx,vy)

    make_plots(x,y,k,u,t,"Leapfrog (Fixed Time Step)",fn="p5_fix_leapfrog.png")

    # RK2
    x = [APHELION];    y = [0];    vx = [0];    vy = [APHELION_VELOCITY];
    x, y, vx, vy, t = leapfrog(x, y, vx, vy, orbits=NUM_ORBITS,adaptive_time=True)  # use all default values
    u = potential_energy(x, y)
    k = kinetic_energy(vx, vy)

    make_plots(x, y, k, u, t, "Leapfrog (Dynamic Time Step)",fn="p6_dyn_leapfrog.png")


if __name__ == '__main__':
    main()


