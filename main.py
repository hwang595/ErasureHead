from __future__ import print_function
import time
import sys
sys.path.append('./src/')
from naive import *
from coded import *
from replication import *
from avoidstragg import *
from partial_replication import *
from partial_coded import *
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 11:
    print("Usage: python main.py n_procs n_rows n_cols input_dir is_real dataset is_coded n_stragglers partial_straggler_partitions coded_ver num_itrs")
    sys.exit(0)

n_procs, n_rows, n_cols, input_dir, is_real, dataset, is_coded, n_stragglers , partitions, coded_ver  = [x for x in sys.argv[1:]]
n_procs, n_rows, n_cols, is_real, is_coded, n_stragglers , partitions, coded_ver = int(n_procs), int(n_rows), int(n_cols), int(is_real), int(is_coded), int(n_stragglers), int(partitions), int(coded_ver)
input_dir = input_dir+"/" if not input_dir[-1] == "/" else input_dir


# ---- Modifiable parameters
num_itrs = 100 # Number of iterations

alpha = 1.0/n_rows #sometimes we used 0.0001 # --- coefficient of l2 regularization

learning_rate_schedule = 10.0*np.ones(num_itrs)
# eta0=10.0
# t0 = 90.0
# learning_rate_schedule = [eta0*t0/(i + t0) for i in range(1,num_itrs+1)]

# -------------------------------------------------------------------------------------------------------------------------------

params = []
params.append(num_itrs)
params.append(alpha)
params.append(learning_rate_schedule)

if not size == n_procs:
    print("Number of processers doesn't match!")
    sys.exit(0)

if not is_real:
    dataset = "artificial-data/" + str(n_rows) + "x" + str(n_cols)

if is_coded:

    if partitions:
        if(coded_ver == 1):
            partial_replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
        elif(coded_ver == 0):
            partial_coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/partial/" + str((partitions-n_stragglers)*(n_procs-1)) + "/", n_stragglers, partitions, is_real, params)
            
    else:
        if(coded_ver == 0):
            coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params)
            
        elif(coded_ver == 1):
            replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params)

        elif(coded_ver ==2):
            avoidstragg_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params)
else:
    naive_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", is_real, params)

comm.Barrier()
MPI.Finalize()
