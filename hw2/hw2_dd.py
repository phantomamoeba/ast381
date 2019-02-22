#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #2
#Februrary 26, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""

import numpy as np
import matplotlib.pyplot as plt

#need to think about space (x,y) and time (all x,y at a particular time step)
#maybe a 3 DIM ndarry x,y,t ?
#maybe not ... seems all are at t -> t+1 and need only space n-1 to n+1


L = 5. * 2. * np.pi #width of the graph (5 cycle lengths)
cs = 1.0 #sound speed
show_animation = False

def initial_wave(lam,dx):
    """
    :param lam: wavelength resolution
    :param dx: step size (in space)
    :return: initial value vectors centered at L/2

    """

    x = np.array(np.arange(0.0,L+dx,dx))
    k = 2*np.pi/lam

    x0_idx = (np.abs(x - L/2.0)).argmin()
    x0 = x[x0_idx]

    y = np.cos(k*(x-x0))*np.exp((-1.*(x-x0)**2.)/2.)

    return x,y



def animate(x,y,title=None,step=None,pause=0.001):
    """
    Simple animation of the wave advecting ...
    """
    plt.clf()

    if step is not None and title is not None:
        plt.title("%s (Step %d)" %(title,step))

    plt.plot(x,y)
    plt.draw()
    plt.pause(pause)
    plt.show(block=False)


def plot(x,y1,y2,y3,y4,title):

    #just zoom in on the middle part
    xl_idx = (np.abs(x - L / 3.0)).argmin()
    xr_idx = (np.abs(x - (L - L / 3.0))).argmin()


    plt.close('all')
    plt.figure(figsize=(10, 10))
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # numerical orbit plot

    plt.subplot(221)
    plt.title("Reference (Analytic)")
    plt.plot(x[xl_idx:xr_idx], y1[xl_idx:xr_idx])

    plt.subplot(222)
    plt.title("FTCS")
    plt.plot(x[xl_idx:xr_idx], y2[xl_idx:xr_idx])

    plt.subplot(223)
    plt.title("Lax")
    plt.plot(x[xl_idx:xr_idx], y3[xl_idx:xr_idx])

    plt.subplot(224)
    plt.title("Lax-Wendroff")
    plt.plot(x[xl_idx:xr_idx], y4[xl_idx:xr_idx])

    plt.show()




def perfect_shift(y):
    """
    Sift perfectly, no calculations of y; just y(n) -> y((n+1)%L)
    (basically a rotate-right) ... just for reference
    """
    return np.append([y[-1]],y[0:-1])


def ftcs(_x,_y,_cs,_dx,_dt):
    """
    Advance 1 step using Forward-time, Centered-space
    :return:

    """
    #todo: in a loop ... think about a more pythonic way to do this

    # s = _cs * _dt / (2. * _dx)
    # next_y = np.zeros(np.shape(_y)) #next time step
    # nLen = len(_y)
    # for n in range(nLen):
    #     n_next = (n + 1) if n < (nLen-1) else 0
    #     n_prev = (n - 1) if n > 0 else nLen-1
    #
    #     next_y[n] = _y[n] - s*(_y[n_next] - _y[n_prev])
    #
    #     print(n, s, next_y[n], _y[n], _y[n_next], _y[n_prev])
    #
    # next_y = _y[:] - s * (np.append(_y[1:], _y[0]) - np.append(_y[-1], _y[:-1]))
    #
    #
    # return next_y

    #this can get out of hand fast (overflow), so will limit the max value
    if np.max(_y) > 1e30:
        _y /= 1e30
    s = _cs * _dt / (2. * _dx)
    next_y = _y[:] - s * (np.append(_y[1:], _y[0]) - np.append(_y[-1], _y[:-1]))


    return next_y



def lax(_x,_y,_cs,_dx,_dt):
    """
    Advance 1 step using Lax
    :return:

    Works pretty well ... keeps the shape, but the amplitude decays and the wave spreads (? numerical viscosity?)
    """
    s = _cs * _dt / (2. * _dx)
    next_y = 0.5 *(np.append(_y[1:], _y[0]) + np.append(_y[-1], _y[:-1]))  \
             - s * (np.append(_y[1:], _y[0]) - np.append(_y[-1], _y[:-1]))
    return next_y


def lax_wen(_x,_y,_cs,_dx,_dt):
    """
    Advance 1 step using Lax-Wendroff
    :return:

    """

    s = _cs * _dt / (2. * _dx)
    s2 = (_cs**2.) * (_dt**2.) / (2. * (_dx**2.))

    next_y = _y[:] - s*(np.append(_y[1:], _y[0]) - np.append(_y[-1], _y[:-1])) \
             + s2*(np.append(_y[1:], _y[0]) + np.append(_y[-1], _y[:-1]) - 2*_y[:])

    return next_y


def main():

    dx = 2. * np.pi / 1000.
    dt = dx/cs * 0.5 #better than minimum CFL criterion
    steps = 10000 #10x cycles in steps

    lam = 100.*dx


    # ######################
    # #Lax-Wen
    # ######################
    #
    # x, y = initial_wave(lam,dx)
    # for i in range(steps):
    #     y = lax_wen(x,y,cs,dx,dt)
    #
    #     if show_animation: #animation
    #         if i %100 == 0:
    #             animate(x,y,title="Lax-Wen", step=i)
    #             print(i)
    #
    # #plot at end
    # plt.plot(x, y)
    # plt.show()
    # exit()
    #







    ######################
    # FTCS
    ######################

    x, y = initial_wave(lam, dx)

    y_init = y

    for i in range(steps):
        y = ftcs(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % 100 == 0:
                animate(x, y, title="FTCS", step=i)

    y_ftcs = y

    ######################
    # Lax
    ######################

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % 100 == 0:
                animate(x, y, title="Lax", step=i)

    y_lax = y

    ######################
    #Lax-Wen
    ######################

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax_wen(x,y,cs,dx,dt)

        if show_animation: #animation
            if i %100 == 0:
                animate(x,y,title="Lax-Wen", step=i)

    y_lax_wen = y

    plot(x,y_init,y_ftcs,y_lax,y_lax_wen,title="High Resolution")

    #
    # #Dummy
    # x,y = initial_wave(100*dx)
    #
    # plt.plot(x,y)
    #
    # for i in range(1000):
    #     y = perfect_shift(y)
    #
    # plt.plot(x,y)
    # plt.show()
    # #end Dummy
    #
    # #FTCS
    # x, y = initial_wave(100 * dx)
    # #for i in range(1000):

if __name__ == '__main__':
    main()


