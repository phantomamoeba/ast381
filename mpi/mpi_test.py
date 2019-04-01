from mpi4py import MPI

import numpy as np


class Dumclass:
    def __init__(self):
        self.a = 'a'
        self.b = 123.4
        self.c = ['a','b','c']

    def __str__(self):
        return "Dumclass a:%s, b:%f c:" %(self.a, self.b)+ str(self.c)


TOTAL_SIMS_TO_RUN = 1000
SIZE_OF_WAVEBINS = 1000
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
NAME = MPI.Get_processor_name()

# MPI start the clock
walltime = MPI.Wtime()

#print(RANK,NAME,walltime)
if True:
    data = None
    r = None
    seed = None
    dc = Dumclass() #everybody has this ...

    dcs = [None]*SIZE #everybody has this

    if (RANK == 0):
        # todo: manager communicate the TOTAL number of simulations to run
        # since the program has this defined, no need to actually communicate here

        # dummy:
        r = np.arange(SIZE, dtype=float)
        r += np.random.random(SIZE)

        dc = Dumclass()

        for i in range(SIZE):
            dcs[i] = Dumclass()
            dcs[i].b = float(i)*100.0
            #dcs = dcs.append(d)

        seed = np.arange(SIZE,dtype=float)
        seed += np.random.random(SIZE)

        pass

    #every body recieves ?

    total_sims_to_run = COMM.bcast(TOTAL_SIMS_TO_RUN)
    seed = COMM.scatter(seed,root=0)
    print(RANK,seed)

    dcr = COMM.bcast(dc)

    data = COMM.scatter(r)

    datax = COMM.scatter(dcs,root=0)


    #print(RANK,data,bc)
    #print(RANK,str(dcs))
    print(RANK,str(datax))


    #everyone:
    sim_data = np.arange(SIZE_OF_WAVEBINS) + RANK
    datax.b = RANK

    recvbuf = None
    if RANK == 0:
        recvbuf = np.empty([SIZE, SIZE_OF_WAVEBINS], dtype=np.int64)

    COMM.Gather(sim_data, recvbuf, root=0)

    if RANK==0:
        sum = np.zeros(10)
        for i in range(SIZE):
            #print(recvbuf[i])
            sum += recvbuf[i]

        print(sum)

    if (RANK != 0):
        # todo: receive the TOTAL number of simulations to run
        # since the program has this defined, no need to actually communicate here
        pass

    # ALL cores (manager and workers) run simulations
    #n_esc = run_all(RANK, TOTAL_SIMS_TO_RUN, SIZE)

    # ALL cores send results to manager
    #COMM.Gather(n_esc, root=0)

    if (RANK != 0):
        # todo: communicate back the n_esc AND number sims run
        pass

    if (RANK == 0):
        # todo: receive the n_esc and number of sims run
        pass

        # todo: then add it to the manager's own run

# MPI stop the clock, get ellapsed time
walltime = MPI.Wtime() - walltime
#print(RANK,NAME,walltime)