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
from approximate_coding import *

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 14:
    print("Usage: python main.py n_procs n_rows n_cols input_dir is_real dataset is_coded n_stragglers partial_straggler_partitions coded_ver num_itrs")
    sys.exit(0)

n_procs, n_rows, n_cols, input_dir, is_real, dataset, is_coded, n_stragglers , partitions, coded_ver, num_collect, add_delay, update_rule  = [x for x in sys.argv[1:]]
n_procs, n_rows, n_cols, is_real, is_coded, n_stragglers , partitions, coded_ver = int(n_procs), int(n_rows), int(n_cols), int(is_real), int(is_coded), int(n_stragglers), int(partitions), int(coded_ver)
num_collect = int(num_collect)
add_delay = int(add_delay)
input_dir = input_dir+"/" if not input_dir[-1] == "/" else input_dir


# ---- Modifiable parameters
num_itrs = 100 # Number of iterations

alpha = 1.0/n_rows #sometimes we used 0.0001 # --- coefficient of l2 regularization

# for amazon dataset
learning_rate_schedule = 10.0*np.ones(num_itrs)
# covtype dataset

#learning_rate_schedule = 0.1*np.ones(num_itrs)
# eta0=10.0
# t0 = 90.0
# learning_rate_schedule = [eta0*t0/(i + t0) for i in range(1,num_itrs+1)]

# for regression task
#learning_rate_schedule = [0.1*(0.98)**i for i in range(1, num_itrs+1)]

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
            coded_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, add_delay, update_rule)
            
        elif(coded_ver == 1):
            if dataset != 'kc_house_data':
                replication_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, add_delay, update_rule)
            else:
                replication_linear_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, add_delay, update_rule)
        elif(coded_ver ==2):
            avoidstragg_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params)

        elif(coded_ver == 3):
            if dataset != 'kc_house_data':
            #approx_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, num_collect, time_sleep)
                approx_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, num_collect, add_delay, update_rule)
            else:
                approx_linear_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, num_collect, add_delay, update_rule)
else:
    if dataset != 'kc_house_data':
        naive_logistic_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, add_delay, update_rule)
    else:
        naive_linear_regression(n_procs, n_rows, n_cols, input_dir + dataset +"/" + str(n_procs-1) + "/", n_stragglers, is_real, params, add_delay, update_rule)

comm.Barrier()
MPI.Finalize()