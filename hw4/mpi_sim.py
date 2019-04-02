from mpi4py import MPI

import matplotlib
matplotlib.use('agg')

import photon
import utilities
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys



h = 6.626e-27
k = 1.381e-16
T = 50000
c = 2.998e10
R_star = 200 * 69.551e9
sb = 5.6704e-5

PHOTONS_PER_BIN = 100 #100
TOTAL_SIMS_TO_RUN = 0 #1000 (NOT PER PROC, but overall total)
SIZE_OF_WAVEBINS = 1000  # 1000 bins in logspace between 10^-2 and 10^0.5 microns
SIM_SAMPLES = 10

DEBUG_TEST = False
DEBUG_PRINT = False

def blackbody(wavelengths):
    def B(lambd):
        return (2 * h * c ** 2 / lambd ** 5) * (1 / (np.exp(h * c / (lambd * k * T)) - 1))

    L_lambd = []
    for lambd in wavelengths:
        L_lambd.append(4 * np.pi * R_star ** 2 * np.pi * B(lambd / 10000))

    delta_lambds = [(wavelengths[i + 1] - wavelengths[i]) / 1e4 for i in range(len(wavelengths) - 1)]
    delta_lambds.append(delta_lambds[-1])
    L_tot = sum(np.multiply(L_lambd, delta_lambds))
    print("Integrated luminosity:", L_tot)
    print("Theoretical luminosity:", 4 * np.pi * R_star ** 2 * sb * T ** 4)

    return L_lambd


def amdhal(t1,tp,p):
    return (1.-1./(t1/tp))/(1. - 1/float(p))

def scaling_plot():

    #times manually copied from TACC runs
    desktop_time = 106.69005069732665
    norm_times = [76.690588633219400,19.808851182460785,11.202456245819727,5.841509188214938,3.187032667299112]
    procs = np.array([1.,4.,8.,16.,32.])
    continuous_procs = np.arange(1,33,1)


    #plot prop to 1/procs (scaled to pass through the first point)
    plt.plot(continuous_procs,1/continuous_procs * norm_times[0],c='b',label="ideal 1/p")
    plt.plot(procs,norm_times, c='r',label="actual")
    plt.scatter(procs, norm_times,s=80, facecolors='none', edgecolors='r')
    plt.scatter([1], [desktop_time], label="desktop")

    plt.title("Mean Walltime vs Number of Processors per Sim\n(Sim = 100 photons x 1000 wavebins)\nTACC Stampede2 SKX-Normal")
    plt.ylabel("Mean Walltime [s] (per 100,000 photons)")
    plt.xlabel("Number of processors")

    plt.legend()

    plt.savefig("mean_times.png")
    plt.close('all')

def amdhal_report(times, procs):
    for i in range(len(procs)-1):
        print("p(%d), f_parallel = %0.4f" %(int(procs[i+1]),amdhal(times[0],times[i+1],procs[i+1])))

"""
Each worker will simulate over the entire wavelength grid, since shorter wavelengths die quickly, they cost
disproportionatly less time to simulate, so this keeps all workers on an even cost

The wavelength bins are fixed, but it is cheaper to generate in each worker than to distribute the list to all.

The manager decides the total number of simulations to run and broadcasts that number to all workers. A simulation
consists of 100 photons per wavelength bin (for 100x1000 photons). This is just for learning purposes as it is
actually more efficient in this case to not broadcast, since the total number is a constant.

The manager generates the random seeds (though that could easily be done in each worker if want to just use the 
RANK as the seed) and scatters.

All cores (manager included) execute 1/p simulations. If 1/p is not an integer, then if the RANK is < remainder,
the worker performs 1 extra simulation. 

"""

def sim_initialize(rank): #all cores do this
    wavelengths = np.logspace(-2, 0.5, SIZE_OF_WAVEBINS)  # 10AA ~ 30,000AA
    np.random.seed(rank)

    return wavelengths


def run_sim(wavelengths,photons_to_simulate,seed=None): #all cores do this
    """
    Run a single simulation of 100 photons in each bin
    :param wavelengths:
    :return: n_esc ... array of the number of photons in each bin that escaped
    """

    len_waves = len(wavelengths)
    utilities.prand_seed(seed)

    if DEBUG_PRINT:
        print("Running sim ....")

    #build up photons
    photons = []
    for w in wavelengths:
        same_w = []
        for i in range(photons_to_simulate):
            same_w.append(photon.Photon(w))
        photons.append(same_w)
    photons = np.array(photons)

    # Propagate photons, build up f_esc as we go
    n_esc = np.zeros(len_waves).astype(int)
    for i in range(len_waves):
        ct = 0
        for j in range(len(photons[i])):
            while photons[i][j].status == 0:
                photons[i][j].propagate()
            if photons[i][j].status == 1:
                #print("Escaped ...")
                ct += 1
        n_esc[i] = ct

    if DEBUG_PRINT:
        print("n_esc", type(n_esc[0]) , n_esc)
    return n_esc

def run_sim_mem_alt(wavelengths,photons_to_simulate,seed=None): #all cores do this
    """
    Run a single simulation of 100 photons in each bin
    :param wavelengths:
    :return: n_esc ... array of the number of photons in each bin that escaped
    """

    len_waves = len(wavelengths)
    utilities.prand_seed(seed)

    if DEBUG_PRINT:
        print("Running sim ....")

    # Propagate photons, build up f_esc as we go
    n_esc = np.zeros(len_waves).astype(int)
    for i in range(len_waves):
        ct = 0
        for j in range(photons_to_simulate):
            p = photon.Photon(wavelengths[i])
            while p.status == 0:
                p.propagate()
            if p.status == 1:
                #print("Escaped ...")
                ct += 1
                #else, was absorbed
        n_esc[i] = ct

    if DEBUG_PRINT:
        print("n_esc", type(n_esc[0]) , n_esc)
    return n_esc

def run_all(rank,total_sims_to_run,total_cores,seed=None):
    wavelengths = sim_initialize(rank)

    sims_to_run = total_sims_to_run // total_cores

    if rank < (total_sims_to_run % total_cores):
        sims_to_run += 1


    if DEBUG_TEST:
        sum_n_esc = np.full(len(wavelengths),rank)
        return sum_n_esc

    #sum_n_esc = np.zeros(len(wavelengths))
    photons_to_simulate = PHOTONS_PER_BIN * sims_to_run #ie. PER wavebin

    print("Proc(%d) running %d photons (%d sim equivalents of %d total sims). seed(%d) ..."
          %(rank,photons_to_simulate * SIZE_OF_WAVEBINS,sims_to_run, total_sims_to_run,seed))
    #sum_n_esc = run_sim(wavelengths,photons_to_simulate,seed)
    sum_n_esc = run_sim_mem_alt(wavelengths, photons_to_simulate, seed)

    return sum_n_esc


def main():

    global TOTAL_SIMS_TO_RUN, PHOTONS_PER_BIN, SIZE_OF_WAVEBINS

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()

    #for convenenience
    if TOTAL_SIMS_TO_RUN < 1:
        TOTAL_SIMS_TO_RUN = SIM_SAMPLES * SIZE #give everyone SIM_SAMPLES, and take an average time for comparision

    #just for learning purposes
    #in this specific case, could skip this entirely and just have each worker use its RANK as the seed
    #as well as the total sims to run ... since this is previously agreed on (as a global constant)
    seed = None
    if (RANK == 0):
        # MPI start the clock
        walltime = MPI.Wtime()
        # print ("Start time:", walltime)

        #since the program has this defined, no need to actually communicate here
        #build the seeds:
        seed = np.arange(SIZE,dtype=int)

    #send everybody the total number to run
    COMM.bcast(TOTAL_SIMS_TO_RUN)

    #send each core its own seed
    seed = COMM.scatter(seed)

    #ALL cores (manager and workers) run simulations
    n_esc = run_all(RANK,TOTAL_SIMS_TO_RUN,SIZE,seed)

    #ALL cores send results to manager
    recvbuf = None
    if RANK == 0:
        recvbuf = np.empty([SIZE, SIZE_OF_WAVEBINS], dtype=int)

    #I think there is another way to do this ... like a Reduce call?

    #gather up everyone's n_esc
    COMM.Gather(n_esc, recvbuf, root=0)

    #sum up ... manager
    sum = []
    count = 0
    if RANK==0:
        sum = np.zeros(SIZE_OF_WAVEBINS)
        for i in range(SIZE):
            if DEBUG_PRINT:
                print("Summing", i, recvbuf[i])
            sum += recvbuf[i]

        if DEBUG_PRINT:
            print("SUM", sum)

        sum = np.array(sum).astype(float)

        f_esc = sum / (PHOTONS_PER_BIN * TOTAL_SIMS_TO_RUN)

        if DEBUG_PRINT:
            print("f_esc", f_esc)



    if RANK==0:
        # MPI stop the clock, get ellapsed time
        walltime = MPI.Wtime() - walltime

        print("Delta-time: ", walltime)
        print("   Per-sim: ", walltime/TOTAL_SIMS_TO_RUN)

    MPI.Finalize()

    #outside of MPI
    #todo: here is where we would make the plots, but don't bother since this is just
    #todo: a timing exercise

    if (RANK == 0):
        #sanity check .. make sure the escape curve matches the original data
        wavelengths = np.logspace(-2, 0.5, SIZE_OF_WAVEBINS)  # 10AA ~ 30,000AA
        plt.close('all')
        plt.plot(wavelengths, f_esc, label="mean f_esc")
        #plt.fill_between(wavelengths, f_esc_low, f_esc_high, color='k', alpha=0.3, label=r"1-$\sigma$")
        plt.xscale('log')
        plt.legend()
        plt.title("f_esc by wavelength (%d simulations)" % (TOTAL_SIMS_TO_RUN))
        plt.xlabel("wavelength bin [microns]")
        plt.ylabel("fraction of escaped photons")

        plt.savefig("rel_f_esc.png")

    #MPI.Finalize()

if __name__ == '__main__':
    main()
