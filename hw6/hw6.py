#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #6
#April 18, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""


import numpy as np
import tables
import os.path as op
import matplotlib.pyplot as plt

mid_x = 40.0

BASEDIR = "/home/dustin/code/python/ast381_compastro/hw6/"


def pressure(int_energy, density, gamma=1.4):
    return (gamma-1.)*(int_energy * density) #

class Particle:
    def __index__(self):
        self.x = -1.
        self.vx = 0.
        self.density = 0.
        self.pressure = 0.
        self.mass = 0.

    def clean_noise(self):
        if self.vx < 1e-17:
            self.vx = 0.0

        if self.density < 1e-17:
            self.density = 0.0

        if self.pressure < 1e-17:
            self.pressure = 0.0

#testing
if False:
    h5 = tables.open_file(BASEDIR+"out/snapshot_000.hdf5")
    coords = h5.root.PartType0.Coordinates
    x = [c[0] for c in coords]
    vels = h5.root.PartType0.Velocities
    vx = [v[0] for v in vels]
    densities = h5.root.PartType0.Density
    internal_e = h5.root.PartType0.InternalEnergy



def read_snapshot(fn):

    h5 = tables.open_file(BASEDIR+fn)

    particles = []

    # coords = h5.root.PartType0.Coordinates
    # x = [c[0] for c in coords]
    # vels = h5.root.PartType0.Velocities
    # vx = [v[0] for v in vels]
    # densities = h5.root.PartType0.Density
    # internal_e = h5.root.PartType0.InternalEnergy

    for i in range(len(h5.root.PartType0.Coordinates)):
        p = Particle()
        p.mass = h5.root.PartType0.Masses[i]
        p.x = h5.root.PartType0.Coordinates[i][0]

        #handle the coordinate rotation
        # 40 < x < 80 --> -40 < x < 0
        # 0 < x < 40 --> 0 < x < 40
        if p.x > mid_x:
            p.x -= 2*mid_x
        p.vx = h5.root.PartType0.Velocities[i][0]
        p.density = h5.root.PartType0.Density[i]
        p.pressure = pressure(h5.root.PartType0.InternalEnergy[i],p.density)
        p.clean_noise()
        particles.append(p)

    h5.close()

    particles.sort(key=lambda a:a.x )


    return particles

def make_plot(particles,ts=-1,ic=None,fn=None):

    plt.close('all')

    plt.figure(figsize=(4,8))

    plt.subplots_adjust(hspace=0.4)

    x = [p.x for p in particles]

    if ic is not None:
        ic_x = [p.x for p in ic]

    plt.subplot(411)
    plt.title("Sod Shock Tube (ts=%0.1f)" % ts)
    plt.xlim((-10, 10))
    plt.plot(x, [p.density for p in particles],c='b',label="Current")
    if ic is not None:
        plt.plot(ic_x, [p.density for p in ic], c='r',label="Initial")
    plt.ylabel(r"$\rho$")

    plt.subplot(412)
    plt.xlim((-10, 10))
    plt.plot(x, [p.pressure for p in particles],c='b',label="Current")
    if ic is not None:
        plt.plot(ic_x, [p.pressure for p in ic], c='r',label="Initial")
    plt.ylabel("P")

    plt.subplot(413)
    plt.xlim((-10, 10))
    #plt.ylim((0,0.1))
    plt.ylim(bottom=-0.01)
    plt.plot(x, [p.vx for p in particles],c='b',label="Current")
    if ic is not None:
        plt.plot(ic_x, [p.vx for p in ic], c='r',label="Initial")
    plt.ylabel(r"$v_{x}$")

    plt.subplot(414)
    plt.xlim((-10, 10))
    plt.plot(x, np.array([p.pressure for p in particles])/(np.array([p.density for p in particles])**1.4),
             c='b',label="Current")
    if ic is not None:
        plt.plot(ic_x, np.array([p.pressure for p in ic])/(np.array([p.density for p in ic])**1.4),
                 c='r',label="Initial")

    plt.ylabel(r"P/$\rho^{\gamma}$")

    plt.xlabel("x")

    plt.legend(loc='center left')

    plt.tight_layout()

    if fn is not None:
        plt.savefig(BASEDIR+fn)
    else:
        plt.show()



def main():

    ic = read_snapshot("out/snapshot_000.hdf5")

    make_plot(ic, ts=0.0,ic=None,fn="t00.png")
    make_plot(read_snapshot("out/snapshot_001.hdf5"), ts=1.0,ic=ic, fn="t01.png")

    make_plot(read_snapshot("out/snapshot_002.hdf5"), ts=2.0, ic=ic, fn="t02.png")
    make_plot(read_snapshot("out/snapshot_003.hdf5"), ts=3.0, ic=ic, fn="t03.png")
    make_plot(read_snapshot("out/snapshot_004.hdf5"), ts=4.0, ic=ic, fn="t04.png")

    make_plot(read_snapshot("out/snapshot_005.hdf5"), ts=5.0, ic=ic, fn="t05.png")

    make_plot(read_snapshot("out/snapshot_006.hdf5"), ts=6.0, ic=ic, fn="t06.png")
    make_plot(read_snapshot("out/snapshot_007.hdf5"), ts=7.0, ic=ic, fn="t07.png")
    make_plot(read_snapshot("out/snapshot_008.hdf5"), ts=8.0, ic=ic, fn="t08.png")
    make_plot(read_snapshot("out/snapshot_009.hdf5"), ts=9.0, ic=ic, fn="t09.png")


    make_plot(read_snapshot("out/snapshot_010.hdf5"), ts=10.0, ic=ic, fn="t10.png")




if __name__ == '__main__':
    main()