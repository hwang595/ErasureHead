from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import scipy.sparse as sps
import time
from mpi4py import MPI

def naive_logistic_regression(n_procs, n_samples, n_features, input_dir, is_real_data, params):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    rounds = params[0]

    beta=np.zeros(n_features)

    # Loading data on workers
    if (rank):

        if not is_real_data:
            X_current = load_data(input_dir+str(rank)+".dat")
            y = load_data(input_dir+"label.dat")
        else:
            X_current = load_sparse_csr(input_dir+str(rank))
            y = load_data(input_dir+"label.dat")

        rows_per_worker = X_current.shape[0]
        y_current=y[(rank-1)*rows_per_worker:rank*rows_per_worker]

    # Initializing relevant variables
    if (rank):

        predy = X_current.dot(beta)
        g = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        send_req = MPI.Request()
        recv_reqs = []

    else:

        msgBuffers = [np.zeros(n_features) for i in range(n_procs-1)]
        g=np.zeros(n_features)
        betaset = np.zeros((rounds, n_features))
        timeset = np.zeros(rounds)
        worker_timeset=np.zeros((rounds, n_procs-1))
        
        request_set = []
        recv_reqs = []

        cnt_completed = 0

        status = MPI.Status()

        eta0=params[2] # ----- learning rate schedule
        alpha = params[1] # --- coefficient of l2 regularization
        utemp = np.zeros(n_features) # for accelerated gradient descent
   
    # Posting all Irecv requests for master and workers
    if (rank):

        for i in range(rounds):
            req = comm.Irecv([beta, MPI.DOUBLE], source=0, tag=i)
            recv_reqs.append(req)

    else:

        for i in range(rounds):
            recv_reqs = []
            for j in range(1,n_procs):
                req = comm.Irecv([msgBuffers[j-1], MPI.DOUBLE], source=j, tag=i)
                recv_reqs.append(req)
            request_set.append(recv_reqs)

    ########################################################################################################
    comm.Barrier()

    if rank==0:
        orig_start_time= time.time()
        print("---- Starting Naive Iterations ----")

    for i in range(rounds):
        
        if rank==0:

            if(i%10 == 0):
                print("\t >>> At Iteration %d" %(i))

            start_time = time.time()

            for l in range(1,n_procs):
                comm.Isend([beta,MPI.DOUBLE],dest=l,tag=i)

            g[:]=0
            cnt_completed = 0

            while cnt_completed < n_procs-1:
                req_done = MPI.Request.Waitany(request_set[i], status)
                src = status.Get_source()
                worker_timeset[i,src-1]=time.time()-start_time
                request_set[i].pop(req_done)
                
                g+=msgBuffers[src-1]   # add the partial gradients
                cnt_completed+=1

            grad_multiplier = eta0[i]/n_samples
            # ---- update step for gradient descent
            # np.subtract((1-2*alpha*eta0[i])*beta , grad_multiplier*g, out=beta)

            # ---- updates for accelerated gradient descent
            theta = 2.0/(i+2.0)
            ytemp = (1-theta)*beta + theta*utemp
            betatemp = ytemp - grad_multiplier*g - (2*alpha*eta0[i])*beta
            utemp = beta + (betatemp-beta)*(1/theta)
            beta[:] = betatemp
            
            timeset[i] = time.time() - start_time
            betaset[i,:] = beta

        else:

            recv_reqs[i].Wait()
            
            # sendTestBuf = send_req.test()
            # if not sendTestBuf[0]:
            #     send_req.Cancel()

            predy = X_current.dot(beta)
            g = X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
            g *= -1
            send_req = comm.Isend([g, MPI.DOUBLE], dest=0, tag=i)

    #####################################################################################################
    comm.Barrier()
    if rank==0:
        elapsed_time= time.time() - orig_start_time
        print ("Total Time Elapsed: %.3f" %(elapsed_time))
        # Load all training data
        if not is_real_data:
            X_train = load_data(input_dir+"1.dat")
            print(">> Loaded 1")
            for j in range(2,n_procs-1):
                X_temp = load_data(input_dir+str(j)+".dat")
                X_train = np.vstack((X_train, X_temp))
                print(">> Loaded "+str(j))
        else:
            X_train = load_sparse_csr(input_dir+"1")
            for j in range(2,n_procs-1):
                X_temp = load_sparse_csr(input_dir+str(j))
                X_train = sps.vstack((X_train, X_temp))

        y_train = load_data(input_dir+"label.dat")
        y_train = y_train[0:X_train.shape[0]]

        # Load all testing data
        y_test = load_data(input_dir + "label_test.dat")
        if not is_real_data:
            X_test = load_data(input_dir+"test_data.dat")
        else:
            X_test = load_sparse_csr(input_dir+"test_data")

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        training_loss = np.zeros(rounds)
        testing_loss = np.zeros(rounds)
        auc_loss = np.zeros(rounds)

        from sklearn.metrics import roc_curve, auc

        for i in range(rounds):
            beta = np.squeeze(betaset[i,:])
            predy_train = X_train.dot(beta)
            predy_test = X_test.dot(beta)
            training_loss[i] = calculate_loss(y_train, predy_train, n_train)
            testing_loss[i] = calculate_loss(y_test, predy_test, n_test)
            fpr, tpr, thresholds = roc_curve(y_test,predy_test, pos_label=1)
            auc_loss[i] = auc(fpr,tpr)
            print("Iteration %d: Train Loss = %5.3f, Test Loss = %5.3f, AUC = %5.3f, Total time taken =%5.3f"%(i, training_loss[i], testing_loss[i], auc_loss[i], timeset[i]))
        
        output_dir = input_dir + "results/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_vector(training_loss, output_dir+"naive_acc_training_loss.dat")
        save_vector(testing_loss, output_dir+"naive_acc_testing_loss.dat")
        save_vector(auc_loss, output_dir+"naive_acc_auc.dat")
        save_vector(timeset, output_dir+"naive_acc_timeset.dat")
        save_matrix(worker_timeset, output_dir+"naive_acc_worker_timeset.dat")
        print(">>> Done")

    comm.Barrier()