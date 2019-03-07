#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #2
#Februrary 26, 2019

__author__ = "Dustin Davis"

"""

Interesting observations:

1) selecting dt s|t the |Xi| is exactly 1, Lax-Wendroff reduces to Lax. In all other cases (< 1) Lax-Wendroff is superior
2) With exception of Lax-Wendroff at high resolution, the choice of dt can have a significant impact on the final shape
and scale of the advected wave-packet. e.g. Lax looks pretty good (in shape, though not amplitude) at 0.5, but really
starts to break down with dt_scale less than 0.2
  

"""

import numpy as np
import matplotlib.pyplot as plt


L = 5. * 2. * np.pi #width of the graph (5 cycle lengths)
cs = 1.0 #propogation speed
dx_scale = 1000.  # how many bins per period

# choice of scale can greatly impact FTCS and Lax (regardless of resoultion) and Lax-Wendroff at low-resolution
# interestingly ... if I set this to exactly 1.0, Lax-Wendroff reduces (confirmed analytically) to Lax
# for all other scale values, Lax-Wed is superior)
dt_scale = 1.  # can be up to exactly 1 (for this CFL criterion ... Lax or Lax-Wendroff )

show_animation = False  #if true, animate the solutions as they progress (shows the periodic boundaries)

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

    y = np.cos(k*(x-x0))*np.exp(-0.5*(x-x0)**2.)

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


def plot(x,y2,y3,y4,title, fn=None):

    #just zoom in on the middle part
    xl_idx = (np.abs(x - L / 3.0)).argmin()
    xr_idx = (np.abs(x - (L - L / 3.0))).argmin()

    #xl_idx = 0
    #xr_idx = -1

    ref_dx = 2. * np.pi / 1000.
    ref_lam = 100. * ref_dx

    x_ref, y_ref = initial_wave(ref_lam, ref_dx)
    xl_ref_idx = (np.abs(x_ref - L / 3.0)).argmin()
    xr_ref_idx = (np.abs(x_ref - (L - L / 3.0))).argmin()

    plt.close('all')
    plt.figure(figsize=(10, 10))
    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.subplot(221)
    plt.title("Reference (Analytic)")
    plt.plot(x_ref[xl_ref_idx:xr_ref_idx], y_ref[xl_ref_idx:xr_ref_idx])

    plt.subplot(222)
    plt.title("FTCS")
    plt.plot(x[xl_idx:xr_idx], y2[xl_idx:xr_idx])

    plt.subplot(223)
    plt.title("Lax")
    plt.plot(x[xl_idx:xr_idx], y3[xl_idx:xr_idx],label="Lax")
    plt.plot(x_ref[xl_ref_idx:xr_ref_idx], y_ref[xl_ref_idx:xr_ref_idx],ls="-",label="Analytic")
    plt.legend()

    plt.subplot(224)
    plt.title("Lax-Wendroff")
    plt.plot(x[xl_idx:xr_idx], y4[xl_idx:xr_idx],label="Lax-Wendroff")
    plt.plot(x_ref[xl_ref_idx:xr_ref_idx], y_ref[xl_ref_idx:xr_ref_idx],ls="-",label="Analytic")
    plt.legend(loc="upper right")

    if fn is not None:
        plt.savefig(fn)
    else:
        plt.show()
    plt.close('all')




def perfect_shift(y):
    """
    Shift perfectly, no calculations of y; just y(n) -> y((n+1)%L)
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
        _y /= 1e30 #rescale, but keep shape (it is a mess anyway, so there is no real harm)

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

    NOTICE: for choice of dt s|t the |Xi| of CFL criterion is exactly 1, this reduces to Lax

    """

    s = _cs * _dt / (2. * _dx)
    s2 = (_cs**2.) * (_dt**2.) / (2. * (_dx**2.))

    next_y = _y[:] - s*(np.append(_y[1:], _y[0]) - np.append(_y[-1], _y[:-1])) \
             + s2*(np.append(_y[1:], _y[0]) + np.append(_y[-1], _y[:-1]) - 2*_y[:])

    return next_y



def main():



    dx = 2. * np.pi / dx_scale #bin width (not the best choice of naming)

    dt = dx/cs * dt_scale #better than minimum CFL criterion; time step width
    #reminder to self ... these are bin widths, dx/dt is NOT the propogation "velocity"

    steps = int(10 * dx_scale /dt_scale) #10x cycles in steps (dx_scale becase dt is defined in terms of it)

    animate_step = steps//100 #show xxx snap-shot states per run

    lam = 100.*dx

    ######################
    # FTCS
    ######################

    print("FTCS High resolution")
    x, y = initial_wave(lam, dx)

    for i in range(steps):
        y = ftcs(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % animate_step == 0:
                animate(x, y, title="FTCS", step=i)

    y_ftcs = y

    ######################
    # Lax
    ######################

    print("Lax High resolution")

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % animate_step == 0:
                animate(x, y, title="Lax", step=i)

    y_lax = y

    ######################
    #Lax-Wendroff
    ######################

    print("Lax-Wendroff High resolution")

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax_wen(x,y,cs,dx,dt)

        if show_animation: #animation
            if i % animate_step == 0:
                animate(x,y,title="Lax-Wen", step=i)

    y_lax_wen = y


    #plot up all methoods
    plot(x,y_ftcs,y_lax,y_lax_wen,title="High Resolution",fn="high_res.png")



    #################################
    #Low Resolution
    ################################

    lam = 10. * dx

    ######################
    # FTCS
    ######################

    print("FTCS Low resolution")

    x, y = initial_wave(lam, dx)

    for i in range(steps):
        y = ftcs(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % animate_step == 0:
                animate(x, y, title="FTCS", step=i)
    y_ftcs = y

    ######################
    # Lax
    ######################
    print("Lax Low resolution")

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % animate_step == 0:
                animate(x, y, title="Lax", step=i)
    y_lax = y

    ######################
    # Lax-Wendroff
    ######################

    print("Lax-Wendroff Low resolution")

    x, y = initial_wave(lam, dx)
    for i in range(steps):
        y = lax_wen(x, y, cs, dx, dt)

        if show_animation:  # animation
            if i % animate_step == 0:
                animate(x, y, title="Lax-Wen", step=i)

    y_lax_wen = y

    # plot up all methoods
    plot(x, y_ftcs, y_lax, y_lax_wen, title="Low Resolution",fn="low_res.png")



if __name__ == '__main__':
    main()


