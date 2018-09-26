from __future__ import print_function
import sys
import random
from util import *
import os
import numpy as np
import time
from mpi4py import MPI
import scipy.sparse as sps

def partial_replication_logistic_regression(n_procs, n_samples, n_features, input_dir, n_stragglers, n_partitions, is_real_data, params):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rounds = params[0]
    n_workers = n_procs-1

    if (n_workers%(n_stragglers+1)):
        print("Error: n_workers must be multiple of n_stragglers+1!")
        sys.exit(0)

    rows_per_worker = n_samples//((n_partitions-n_stragglers)*n_workers) # per partition num of samples
    n_groups=n_workers/(n_stragglers+1)
    n_separate = n_partitions-n_stragglers-1
    sep_lim = n_separate*rows_per_worker

    # Loading the data    
    if (rank):

        if not is_real_data:

            y = load_data(input_dir+"label.dat")
            X_current = np.zeros([n_partitions*rows_per_worker,n_features])
            y_current = np.zeros(n_partitions*rows_per_worker)
            

            for i in range(n_separate):
                idx = i+n_separate*(rank-1)
                X_current[i*rows_per_worker:(i+1)*rows_per_worker,:] = load_data(input_dir+str(idx+1)+".dat")
                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[idx*rows_per_worker:(idx+1)*rows_per_worker]

            for i in range(n_separate,n_partitions):
                a = (rank-1)/(n_stragglers+1) # index of group
                b = i-n_separate # position inside the group
                idx = n_separate*n_workers+a*(n_stragglers+1)+b
                
                X_current[i*rows_per_worker:(i+1)*rows_per_worker,:] = load_data(input_dir+str(idx+1)+".dat")
                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[idx*rows_per_worker:(idx+1)*rows_per_worker]

        else:

            y = load_data(input_dir + "label.dat")
            
            y_current = np.zeros(n_partitions*rows_per_worker)

            for i in range(n_separate):
                idx = i+n_separate*(rank-1)
                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[idx*rows_per_worker:(idx+1)*rows_per_worker]

                if i==0:
                    X_current = load_sparse_csr(input_dir+str(idx+1))
                else:
                    X_temp = load_sparse_csr(input_dir+str(idx+1))
                    X_current = sps.vstack((X_current,X_temp))

            for i in range(n_separate,n_partitions):
                a = (rank-1)/(n_stragglers+1) # index of group
                b = i-n_separate # position inside the group
                idx = n_separate*n_workers+a*(n_stragglers+1)+b

                y_current[i*rows_per_worker:(i+1)*rows_per_worker] = y[idx*rows_per_worker:(idx+1)*rows_per_worker]

                X_temp = load_sparse_csr(input_dir+str(idx+1))
                X_current = sps.vstack((X_current,X_temp))

            X_current = X_current.tocsr()

    # Initializing relevant variables
    beta=np.zeros(n_features)
    if(rank):

        predy = X_current.dot(beta)
        g_firstpart = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        g_secondpart = -X_current.T.dot(np.divide(y_current,np.exp(np.multiply(predy,y_current))+1))
        
        send_req1 = MPI.Request()
        send_req2 = MPI.Request()
        recv_reqs = []
    
    else:

        print('Stragglers are allowed to be atmost %.2f times slower'%(n_partitions*1.0/(n_partitions-n_stragglers-1)) )
        
        msgBuffers_firstparts = [np.zeros(n_features) for i in range(n_procs-1)]
        msgBuffers_secondparts = [np.zeros(n_features) for i in range(n_procs-1)]

        g=np.zeros(n_features)
        betaset = np.zeros((rounds, n_features))
        timeset = np.zeros(rounds)
        worker_timeset=np.zeros((rounds, n_procs-1))

        request_set = []
        recv_reqs = []
        send_set = []

        cnt_groups = 0
        cnt_firstpart = 0
        completed_groups=np.ndarray(n_groups,dtype=bool)
        completed_workers = np.ndarray(n_workers,dtype=bool)
        completed_firstparts=np.ndarray(n_workers,dtype=bool)
        
        status = MPI.Status()

        eta0= params[2] # ----- learning rate
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
                req1 = comm.Irecv([msgBuffers_firstparts[j-1], MPI.DOUBLE], source=j, tag=2*rounds + i)
                recv_reqs.append(req1)
                req2 = comm.Irecv([msgBuffers_secondparts[j-1], MPI.DOUBLE], source=j, tag=i)
                recv_reqs.append(req2)

            request_set.append(recv_reqs)

    #######################################################################################################################
    comm.Barrier()
    if rank==0:
        print("---- Starting Partial Replication Iterations for " +str(n_stragglers) + " stragglers ----")
        orig_start_time= time.time()

    for i in range(rounds):
        if rank==0:
            
            if(i%10 == 0):
                print("\t >>> At Iteration %d" %(i))

            send_set[:] = []
            g[:]= 0 
            
            cnt_firstpart=0
            completed_firstparts[:]=False

            completed_groups[:]=False
            cnt_groups=0
            completed_workers[:]=False

            start_time = time.time()
            
            for l in range(1,n_procs):
                sreq = comm.Isend([beta, MPI.DOUBLE], dest = l, tag = i)
                send_set.append(sreq)

            while cnt_groups<n_groups or cnt_firstpart<n_workers:

                req_done = MPI.Request.Waitany(request_set[i], status)
                src = status.Get_source()
                tag= status.Get_tag()
                worker_timeset[i,src-1]=time.time()-start_time
                request_set[i].pop(req_done)

                if tag == i:

                    completed_workers[src-1] = True
                    groupid = (src-1)/(n_stragglers+1)
                    
                    if not completed_groups[groupid]:
                        completed_groups[groupid] = True
                        g += msgBuffers_secondparts[src-1]
                        cnt_groups += 1

                elif tag == 2*rounds + i:
                    g += msgBuffers_firstparts[src-1]
                    completed_firstparts[src-1] = True
                    cnt_firstpart += 1

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
            ind_set = [l for l in range(1,n_procs) if not completed_workers[l-1]]
            for l in ind_set:
                worker_timeset[i,l-1]=-1

        else:

            recv_reqs[i].Wait()

            sendTestBuf = send_req1.test()
            if not sendTestBuf[0]:
                send_req1.Cancel()

            sendTestBuf = send_req2.test()
            if not sendTestBuf[0]:
                send_req2.Cancel()
            
            predy=X_current[0:sep_lim,:].dot(beta)
            g_firstpart = X_current[0:sep_lim,:].T.dot(np.divide(y_current[0:sep_lim],np.exp(np.multiply(predy,y_current[0:sep_lim]))+1))
            g_firstpart *= -1
            send_req1 = comm.Isend([g_firstpart,MPI.DOUBLE], dest=0, tag=2*rounds+i)
            
            predy = X_current[sep_lim:,:].dot(beta)
            g_secondpart = X_current[sep_lim:,:].T.dot(np.divide(y_current[sep_lim:],np.exp(np.multiply(predy,y_current[sep_lim:]))+1))
            g_secondpart *= -1
            send_req2 = comm.Isend([g_secondpart,MPI.DOUBLE], dest=0, tag=i)

    ######################################################################################################################
    comm.Barrier()
    if rank==0:
        elapsed_time= time.time() - orig_start_time
        print ("Total Time Elapsed: %.3f" %(elapsed_time))
        # Load all training data
        if not is_real_data:
            X_train = load_data(input_dir+"1.dat")
            for j in range(2,n_procs-1):
                X_temp = load_data(input_dir+str(j)+".dat")
                X_train = np.vstack((X_train, X_temp))
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

        save_vector(training_loss, output_dir+"partialreplication_%d_%d_training_loss.dat"%(n_stragglers,n_partitions))
        save_vector(testing_loss, output_dir+"partialreplication_%d_%d_testing_loss.dat"%(n_stragglers,n_partitions))
        save_vector(auc_loss, output_dir+"partialreplication_%d_%d_auc.dat"%(n_stragglers,n_partitions))
        save_vector(timeset, output_dir+"partialreplication_%d_%d_timeset.dat"%(n_stragglers,n_partitions))
        save_matrix(worker_timeset, output_dir+"partialreplication_%d_%d_worker_timeset.dat"%(n_stragglers,n_partitions))
        print(">>> Done")

    comm.Barrier()