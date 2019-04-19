#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #5
#April 18, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""



import yt
import numpy as np
import tables
import os.path as op
import matplotlib.pyplot as plt

M_Halo = 1e15 #solar masses
U_Mass = 1e10 #gizmo unit of mass in solar masses
Nparticles = 1e4

PART_MASS = M_Halo / Nparticles / U_Mass

BASEDIR = "/home/dustin/code/python/ast381_compastro/hw5/"

def dist(c1,c2):
    return np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2 + (c2[2]-c1[2])**2 )


def sum_r2(coords):
    sq_sum = 0
    for i in range(len(coords)):
        print(i)
        for j in range(i+1,len(coords)):
           sq_sum += (dist(coords[i],coords[j]))**2

    return sq_sum



def make_plot(time,pot,kin,t_mul=0.2,title="",savefn=None):

    plt.close('all')
    plt.figure(figsize=(16,9))

    plt.title(title)

    plt.plot(time * t_mul,kin/abs(pot))
    plt.axhline(y=0.5,ls=":")

    plt.ylabel("Kinetic / |Potential|")
    plt.xlabel("Time [Free Fall Time]")

    if savefn:
        plt.savefig(op.join(BASEDIR,savefn))
    else:
        plt.show()

def read_energy(fn):
    #tecnically want ParticleType1 not the totals, but since this is the only particel this is okay
    t,p,k, = np.loadtxt(op.join(BASEDIR,fn),usecols=(0,2,3),unpack=True)
    return t,p,k



#testing
if False:
    # ds = yt.load(BASEDIR+"out_n4/snapshot_000.hdf5") # ds contains all info in your_snapshot
    # t = ds['Time'] # time of your_snapshot in code_time_unit = code_length_unit/code_velocity_unit (set in the .param file)
    # ad = ds.all_data()  # ad contains all data, in terms of fields.
    #
    # pos =  np.array(ad[('Halo', 'Coordinates')]) # positions of type 'Halo' particles in code_length_unit
    #
    # # Except 'Gas', all other particle types are dissipationless & collisionless in  GIZMO/Gadget by default.
    # # ('Halo', 'Coordinates') is a field in yt. You can see what fields are available with ds.field_list.
    # # Note that the field name for DM particles from your_snapshot may not be the same with what is shown here.
    # # As far as I know, if the output file is in hdf5 format, the field name will be 'PartType1' rather than 'Halo'.
    #
    # vel = np.array(ad[('Halo', 'Velocities')]) # velocities of type 'Halo' particles in code_velocity_unit

    h5 = tables.open_file(BASEDIR+"out_n4/snapshot_000.hdf5")
    coords = h5.root.PartType1.Coordinates
    vels = h5.root.PartType1.Velocities
    kin = np.array([(v[0]**2 + v[1] **2 + v[2]**2) for v in vels]) * PART_MASS * 0.5
    #don't worry about mass units, etc since will divide out from potential

    pot = h5.root.PartType1.Potential


def main():

    #1e4
    time, pot, kin = read_energy("out_n4/energy.txt")
    make_plot(time,pot,kin,t_mul=0.2,title="10,000 Particles (13.9 kpc softening)",savefn="n4.png")


    #1e5 (base)
    time5, pot5, kin5 = read_energy("out_n5.base/energy.txt")

    #decimate the data to plot more clearly
    time5 = time5[::10]
    pot5 = pot5[::10]
    kin5 = kin5[::10]

    make_plot(time5,pot5,kin5,t_mul=0.2,title="100,000 Particles (6.48 kpc softening)",savefn="n5.png")


    #1e5 (short softening length)
    time5s, pot5s, kin5s = read_energy("out_n5.short/energy.txt")
    make_plot(time5s,pot5s,kin5s,t_mul=0.2,title="100,000 Particles (0.648 kpc softening)",savefn="n5_short.png")

    #1e5 (short long length)
    time5l, pot5l, kin5l = read_energy("out_n5.long/energy.txt")
    make_plot(time5l,pot5l,kin5l,t_mul=0.2,title="100,000 Particles (64.8 kpc softening)",savefn="n5_long.png")




    plt.close('all')
    plt.figure(figsize=(16,9))

    plt.title("100,000 Particles (Various Softening Length)")

    plt.plot(time5s * 0.2,kin5s/abs(pot5s),c='g', label="0.648 kpc softening")
    plt.plot(time5 * 0.2,kin5/abs(pot5),c='k', label="6.48 kpc softening")
    plt.plot(time5l * 0.2,kin5l/abs(pot5l), c='r',label="64.8 kpc softening")
    plt.axhline(y=0.5,ls=":")

    plt.ylabel("Kinetic / |Potential|")
    plt.xlabel("Time [Free Fall Time]")

    plt.legend(loc="upper center")

    plt.savefig("n5_mixed.png")

if __name__ == '__main__':
    main()