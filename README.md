# ErasureHead : Distributed Gradient Descent without Delays Using Approximate Gradient Coding
This repository contains source code for reproduce the experimental result in the paper submission of ErasureHead: Distributed Gradient Descent without Delays Using Approximate Gradient Coding. 

This repo extends the [original repo](https://github.com/rashisht1/gradient_coding) of the [Gradient Coding](https://arxiv.org/abs/1612.03301) paper, credit gose to [Rashish Tandon](https://github.com/rashisht1).

## Overview:
ErasureHead is a new approach for distributed gradient descent (GD) that mitigates system delays by employing approximate gradient coding. ErasureHead uses approximate gradient codes to recover an inexact gradient at each iteration, but with significantly higher delay tolerance. Unlike prior work on gradient coding, we provide a performance analysis that combines both delay and convergence guarantees. We establish that down to a small noise floor, ErasureHead converges as quickly as distributed GD and has faster overall runtime under a probabilistic delay model.

<div align="center"><img src="https://github.com/hwang595/approximate_coding_gd/blob/master/images/straggler.jpg" height="350" width="450" ></div>

## Depdendencies:
Tested stable depdencises:
* python 2.7 (Anaconda)
* MPI4Py 0.3.0
* Scikit-learn 0.20.1

We highly recommend installing an [Anaconda](https://www.continuum.io/downloads) environment.
You will get a high-quality BLAS library (MKL) and you get a controlled compiler version regardless of your Linux distro.

We provide [this script](https://github.com/hwang595/ATOMO/blob/master/tools/pre_run.sh) to help you with building all dependencies. To do that you can run:
```
bash ./tools/pre_run.sh
```

## Cluster Setup:
For running on distributed cluster, the first thing you need do is to launch AWS EC2 instances.
### Launching Instances:
[This script](https://github.com/hwang595/ps_pytorch/blob/master/tools/pytorch_ec2.py) helps you to launch EC2 instances automatically, but before running this script, you should follow [the instruction](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to setup AWS CLI on your local machine.
After that, please edit this part in `./tools/pytorch_ec2.py`
``` python
cfg = Cfg({
    "name" : "XXXXXXX",      # Unique name for this specific configuration
    "key_name": "NameOfKeyFile",          # Necessary to ssh into created instances
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 8,
    "num_replicas_to_aggregate" : "8", # deprecated, not necessary
    "method" : "spot",
    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",
    # Machine type - instance type configuration.
    "master_type" : "m4.2xlarge",
    "worker_type" : "m4.2xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-xxxxxxxx",            # id of AMI
    # Launch specifications
    "spot_price" : "0.15",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/dir/to/NameOfKeyFile.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.14.225",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
```
For setting everything up on EC2 cluster, the easiest way is to setup one machine and create an AMI. Then use the AMI id for `image_id` in `pytorch_ec2.py`. Then, launch EC2 instances by running
```
python ./tools/pytorch_ec2.py launch
```
After all launched instances are ready (this may take a while), getting private ips of instances by
```
python ./tools/pytorch_ec2.py get_hosts
```
this will write ips into a file named `hosts_address`, which looks like
```
172.31.16.226 (${PS_IP})
172.31.27.245
172.31.29.131
172.31.18.108
172.31.18.174
172.31.17.228
172.31.16.25
172.31.30.61
172.31.29.30
```
After generating the `hosts_address` of all EC2 instances, running the following command will copy your keyfile to the parameter server (PS) instance whose address is always the first one in `hosts_address`. `local_script.sh` will also do some basic configurations e.g. clone this git repo
```
bash ./tool/local_script.sh ${PS_IP}
```
### SSH related:
At this stage, you should ssh to the PS instance and all operation should happen on PS. In PS setting, PS should be able to ssh to any compute node, [this part](https://github.com/hwang595/ATOMO/blob/master/tools/remote_script.sh#L8-L16) dose the job for you by running (after ssh to the PS)
```
bash ./tools/remote_script.sh
```

## Prepare Datasets
We currently support [Amazon Employee Access](http://yann.lecun.com/exdb/mnist/), [The Forest Covertype](https://www.cs.toronto.edu/~kriz/cifar.html), and [KC Housing](https://www.kaggle.com/harlfoxem/housesalesprediction) datasets. Download, split, and transform datasets by (`data_prepare` dose this for you)
```
bash data_prepare.sh
```

## Job Launching
The script `run_approx_coding.sh` handles the job launching.

To run approximate coding (AGCs):
```
mpirun -np ${N_PROCS} \
--hostfile hosts_address \
python main.py ${N_PROCS} ${N_ROWS} ${N_COLS} ${DATA_FOLDER} ${IS_REAL} ${DATASET} 1 ${N_STRAGGLERS} 0 3 ${N_COLLECT} ${ADD_DELAY} ${UPDATE_RULE}
```

To run exact gradient coding (EGCs):
```
mpirun -np ${N_PROCS} \
--hostfile hosts_address \
python main.py ${N_PROCS} ${N_ROWS} ${N_COLS} ${DATA_FOLDER} ${IS_REAL} ${DATASET} 1 ${N_STRAGGLERS} 0 1 ${N_COLLECT} ${ADD_DELAY} ${UPDATE_RULE}
```

To run exact vanilla (uncoded) gradient descent:
```
mpirun -np ${N_PROCS} \
--hostfile hosts_address \
python main.py ${N_PROCS} ${N_ROWS} ${N_COLS} ${DATA_FOLDER} ${IS_REAL} ${DATASET} 0 ${N_STRAGGLERS} 0 1 ${N_COLLECT} ${ADD_DELAY} ${UPDATE_RULE}
```

## Future Work
Those are potential directions we are actively working on, stay tuned!
* As you may notice from reading the code, we haven't figure out a clean and decent way to fully terminate the stragglers in implementation level, especially when artificial delays are added. We're actively working on in. Though from simulation perspective, we know the exact time when the certain fraction of worker nodes complete their jobs.
* Integrate ErasureHead into the state-of-the-art deep learning frameworks e.g. TensorFlow, PyTorch, and MXNet is another potential direction.
