from mpi4py import MPI
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

PHOTONS_PER_BIN = 10 #100
TOTAL_SIMS_TO_RUN = 2 #1000 (NOT PER PROC, but overall total)
SIZE_OF_WAVEBINS = 10  # 1000 bins in logspace between 10^-2 and 10^0.5 microns

DEBUG_TEST = False
DEBUG_PRINT = True

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


def plot(fn=None):
    # just plot
    # todo: maybe show 3-sigma if using super_f_esc()?
    wavelengths, f_esc, f_esc_err, count = get_super_f_esc()  # get_f_esc()

    f_esc_high = f_esc + f_esc_err
    f_esc_low = f_esc - f_esc_err
    f_esc_high[f_esc_high > 1.0] = 1.0
    f_esc_low[f_esc_low < 0.0] = 0.0

    plt.close('all')
    plt.plot(wavelengths, f_esc, label="mean f_esc")
    plt.fill_between(wavelengths, f_esc_low, f_esc_high, color='k', alpha=0.3, label=r"1-$\sigma$")
    plt.xscale('log')
    plt.legend()
    plt.title("f_esc by wavelength (%d simulations)" % (count))
    plt.xlabel("wavelength bin [microns]")
    plt.ylabel("fraction of escaped photons")

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)


def get_super_f_esc():
    # re-sampling ALL the runs to beat down the error
    wavelengths = np.load("run_data/wavelengths.npy")

    simruns = glob.glob("run_data/simrun_*.npy")
    sims = []
    for s in simruns:
        try:
            f_esc = np.load(s)
            sims.append(f_esc)
        except:
            pass

    f_esc = np.zeros(len(wavelengths))
    f_esc_err = np.zeros(len(wavelengths))

    sims = np.array(sims)

    for i in range(len(wavelengths)):
        f_super = sims[:, i]

        sample = []
        for _ in range(100):
            sample.append(np.mean(np.random.choice(f_super, size=100, replace=True)))

        f_esc[i] = np.mean(sample)
        f_esc_err[i] = np.std(sample)

    return wavelengths, f_esc, f_esc_err, len(simruns)


def get_alt_super_f_esc():
    # re-sampling ALL the runs to beat down the error
    wavelengths = np.load("run_data/wavelengths.npy")

    simruns = glob.glob("run_data/simrun_*.npy")
    sims = []
    for s in simruns:
        try:
            f_esc = np.load(s)
            sims.append(f_esc)
        except:
            pass

    f_esc = np.zeros(len(wavelengths))
    f_esc_err = np.zeros(len(wavelengths))

    sims = np.array(sims)

    for i in range(len(wavelengths)):
        progress_bar(float(i) / len(wavelengths))
        f_super = sims[:, i]  # 1000 entries
        # turn the 1000 into 100,000 photon results

        p_super = np.zeros((100 * len(f_super)))
        for j in range(len(f_super)):
            escape = int(f_super[j] * 100.)
            p_super[j * 100:j * 100 + escape + 1] = 1  # since this is going to be randomly re-sampled below it
            # is okay to just group this way

        sample = []
        for j in range(100):  # the loop is like the replacement part
            re_sample = np.random.choice(p_super, size=10000, replace=False)
            escape = float(np.sum(re_sample)) / float(
                len(re_sample))  # out of 10,000 photons, this is the escape fraction
            sample.append(escape)

        f_esc[i] = np.mean(sample)
        f_esc_err[i] = np.std(sample)

    print("\n")

    return wavelengths, f_esc, f_esc_err, len(simruns)


def get_f_esc():
    # read the local numpy simruns and return the wavebin, f_esc, and error arrays
    wavelengths = np.load("run_data/wavelengths.npy")

    simruns = glob.glob("run_data/simrun_*.npy")
    sims = []
    for s in simruns:
        try:
            f_esc = np.load(s)
            sims.append(f_esc)
        except:
            pass

    f_esc = np.zeros(len(wavelengths))
    f_esc_err = np.zeros(len(wavelengths))

    sims = np.array(sims)

    for i in range(len(f_esc)):
        wave_bin = sims[:, i]
        f_esc[i] = np.mean(wave_bin)
        f_esc_err[i] = np.std(wave_bin)

    return wavelengths, f_esc, f_esc_err, len(simruns)


# def progress_bar(percent, barLen=20):
#     sys.stdout.write("\r")
#     progress = ""
#     for i in range(barLen):
#         if i < int(barLen * percent):
#             progress += "="
#         else:
#             progress += " "
#     sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
#     sys.stdout.flush()
#
#
# def run_sim():
#     # uncomment to set seed and make repeateable
#     # utilities.prand_seed(137)
#
#     # load the wavelenghts if they exist, otherwise create and save them (for future runs)
#     try:
#         wavelengths = np.load("wavelengths.npy")
#     except:
#         wavelengths = np.logspace(-2, 0.5, SIZE_OF_WAVEBINS)  # 10AA ~ 30,000AA
#         np.save("wavelengths", wavelengths)
#     len_waves = len(wavelengths)
#
#     # calculate next run id
#     next_run_id = 0
#     simruns = glob.glob("simrun_*.npy")
#     for s in simruns:
#         try:
#             f_esc = np.load(s)
#             run_id = int(s.split("_")[1].split('.')[0])
#             if run_id > next_run_id:
#                 next_run_id = run_id
#         except:
#             pass
#     next_run_id += 1
#
#     # create 100 initial photons per wave bin
#     print("Creating 100x photon packets [%f, %f microns] in %d bins ..." % (wavelengths[0], wavelengths[-1], len_waves))
#     photons = []
#     for w in wavelengths:
#         same_w = []
#         for i in range(100):
#             same_w.append(photon.Photon(w))
#         photons.append(same_w)
#     photons = np.array(photons)
#
#     # Propagate photons, build up f_esc as we go
#     print("Propagating simulation #%d ..." % (next_run_id))
#     f_esc = np.zeros(len_waves)
#     for i in range(len_waves):
#         ct = 0
#         for j in range(len(photons[i])):
#             # todo: optionally re-seed
#             utilities.prand_seed(None)  # to use time
#
#             while photons[i][j].status == 0:
#                 photons[i][j].propagate()
#             if photons[i][j].status == 1:
#                 ct += 1
#
#         f_esc[i] = float(ct) / len(photons[i])
#
#         # print("wavelength: %f [microns] f_esc = %0.1f" % (wavelengths[i],f_esc[i]*100.))
#         progress_bar(float(i) / len_waves)
#
#     progress_bar(1.0)
#     print("\n")
#     np.save("simrun_%d" % next_run_id, f_esc)


def old_main():
    wavelengths, f_esc, f_esc_err, _ = get_super_f_esc()

    f_esc_high = f_esc + f_esc_err
    f_esc_low = f_esc - f_esc_err

    L_lambd_source = blackbody(wavelengths)
    L_lambd_out = f_esc * L_lambd_source
    L_lambd_low_err = f_esc_low * L_lambd_source
    L_lambd_high_err = f_esc_high * L_lambd_source

    plt.title("Radiation Spectra: source vs. escaped")
    plt.plot(wavelengths, L_lambd_source, label="Source radiation", color='r')
    plt.plot(wavelengths, L_lambd_out, label="Escaped radiation", color='b')
    plt.fill_between(wavelengths, L_lambd_low_err, L_lambd_high_err, color='k', alpha=0.3, label=r"1-$\sigma$")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("wavelength bin [microns]")
    plt.ylabel(r"$L_\lambda$ [erg s${}^{-1}$ cm${}^{-1}$]")

    # plt.savefig("spectrum_1.png")
    plt.show()
    plt.close('all')

    mx = np.max(L_lambd_source)
    plt.title("Radiation Spectra: source vs. escaped (zoom and scaled)")
    plt.plot(wavelengths, L_lambd_source / mx, label="Source radiation", color='r')
    plt.plot(wavelengths, L_lambd_out / mx * 100, label="Escaped radiation x100", color='b')
    plt.axvline(0.05, label="~0.20 constant absorption")

    plt.legend(loc=4)
    plt.ylim(0, 1.05)
    plt.xlim(0.03, 0.07)
    plt.xlabel("wavelength bin [microns]")
    plt.ylabel(r"Dimensionless Lum (scaled at $\lambda_{max}$)")
    # plt.savefig("spectrum_zoom.png")
    plt.show()




"""
Each worker will simulate over the entire wavelength grid, since shorter wavelengths die quickly, they cost
disproportionatly less time to simulate, so this keeps all workers on an even cost

The wavelength bins are fixed, but it is cheaper to generate in each worker than to distribute the list to all.

The manager decides the total number of simulations to run and broadcasts that number to all workers. A simulation
consists of 100 photons per wavelength bin (for 100x1000 photons).

Each worker generates its own pRNG seed as its MPI_Rank.

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


def run_all(rank,total_sims_to_run,total_cores,seed=None):
    wavelengths = sim_initialize(rank)

    sims_to_run = total_sims_to_run // total_cores

    if rank < (total_sims_to_run % total_cores):
        sims_to_run += 1


    if DEBUG_TEST:
        sum_n_esc = np.full(len(wavelengths),rank)
        return sum_n_esc

    #sum_n_esc = np.zeros(len(wavelengths))
    photons_to_simulate = PHOTONS_PER_BIN * sims_to_run

    print("Proc(%d) running %d photons (%d sim equivalents of %d total sims). seed(%d) ..."
          %(rank,photons_to_simulate,sims_to_run, total_sims_to_run,seed))
    sum_n_esc = run_sim(wavelengths,photons_to_simulate,seed)

    return sum_n_esc



def main():


    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    NAME = MPI.Get_processor_name()


    #MPI start the clock
    walltime = MPI.Wtime()
    print ("Start time:", walltime)

    #just for learning purposes
    #in this specific case, could skip this entirely and just have each worker use its RANK as the seed
    #as well as the total sims to run ... since this is previously agreed on (as a global constant)
    seed = None
    if (RANK == 0):
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


    #MPI stop the clock, get ellapsed time
    walltime = MPI.Wtime() - walltime

    print("Delta-time: ", walltime)

    MPI.Finalize()

    #outside of MPI
    #todo: here is where we would make the plot, but don't bother since this is just
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

        plt.show()

    #MPI.Finalize()

if __name__ == '__main__':
    main()
