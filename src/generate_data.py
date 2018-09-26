from __future__ import print_function
import sys
import os
import numpy as np
import random
from util import *

def generate_data(partitions,out_dir):

    # make sure the number of rows is a multiple of number of total partitions
    assert(n_rows % partitions== 0)

    # num workers, and number of rows per worker
    n_rows_per_worker = n_rows / partitions

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Generating Partitioned Matrix Of size %d x %d for a total of %d partitions" % (n_rows, n_cols, partitions))
    print(">>> Each worker gets a matrix of %d x %d doubles, %.2f MB each" % (n_rows_per_worker, n_cols, (n_rows_per_worker * n_cols * 8) / 1000000.0))

    random_beta = generate_random_binvec(n_cols)
    alpha = 1.5
    mu1 = np.multiply(alpha/n_cols, random_beta)
    mu2 = np.multiply(-alpha/n_cols, random_beta)
    label_vector=np.ndarray(n_rows)

    for i in range(1, partitions+1):

        random_matrix = generate_random_matrix_normal(mu1, mu2, n_rows_per_worker, n_cols)
        save_matrix(random_matrix, out_dir + str(i) + ".dat")
        
        prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(random_matrix, random_beta))))
        label_vector[(i-1)*n_rows_per_worker:i*n_rows_per_worker]= np.add(np.multiply(2,np.random.binomial(1,prob_vals)),-1)
        
        print("\t >>> Done with partition %d" % (i))

    save_vector(label_vector, out_dir + "label.dat")

    random_matrix_test = generate_random_matrix_normal(mu1, mu2, int(0.2*n_rows), n_cols)
    prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(random_matrix_test, random_beta))))
    label_vector_test = np.add(np.multiply(2,np.random.binomial(1,prob_vals)),-1)
    save_matrix(random_matrix_test, out_dir + "test_data.dat")
    save_vector(label_vector_test, out_dir + "label_test.dat")
    print("\t >>> Done with Test data")


# ---------- code for call from Makefile starts here
if len(sys.argv) != 8:
    print("Usage: python generate_data.py n_procs n_rows n_cols output_dir n_stragglers n_partitions partial_coded")
    sys.exit(0)

#np.random.seed(0)
n_procs, n_rows, n_cols, output_dir, n_stragglers, n_partitions, partial_coded  = [x for x in sys.argv[1:]]
n_procs, n_cols, n_rows, n_stragglers, n_partitions, partial_coded = int(n_procs), int(n_cols), int(n_rows), int(n_stragglers), int(n_partitions), int(partial_coded)
output_dir = output_dir+"/" if not output_dir[-1] == "/" else output_dir

if not partial_coded:
    output_dir = output_dir + "artificial-data/" + str(n_rows)+"x"+str(n_cols)+ "/" + str(n_procs-1) + "/"
else:
    output_dir = output_dir + "artificial-data/" + str(n_rows)+"x"+str(n_cols)+"/partial/" + str((n_procs-1)*(n_partitions-n_stragglers)) + "/"

# generate data
if not partial_coded:
    generate_data(n_procs-1, output_dir)
else:
    # generate data for partial replication
    generate_data((n_procs-1)*(n_partitions-n_stragglers), output_dir)

print("Data Generation Finished.")
