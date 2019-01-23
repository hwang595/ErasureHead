# No. of workers
N_PROCS=31

# No. of stragglers in our coding schemes
N_STRAGGLERS=3
N_COLLECT=15

#update rule
UPDATE_RULE=AGD

# For partially coded version: how many pieces of workload will one worker be handling.
N_PARTITIONS=30

# Switch to enable partial coded schemes
PARTIAL_CODED=0

# Time sleep
ADD_DELAY=0

# Path to folder containing the data folders
DATA_FOLDER=./straggdata/
HOST_DIR=${HOME}/approximate_coding_gd/hosts_address

IS_REAL=1

#DATASET=covtype
#N_ROWS=396112
#N_COLS=15509

#DATASET=amazon-dataset
#N_ROWS=26215
#N_COLS=241915

DATASET=kc_house_data
N_ROWS=17290
N_COLS=27654

# Note that DATASET is automatically set to artificial-data/ (n_rows)x(n_cols)/... if IS_REAL is set to 0 \
# or artificial-data/partial/ (n_rows)x(n_cols)/... if PARTIAL_CODED is also set to 1

##########
#MODES:
#==========
# 1 0 1: grad coding rep
# 1 0 3: approx coding
# 0 x x: vanilla GD 
mpirun -np ${N_PROCS} \
--hostfile hosts_address \
python main.py ${N_PROCS} ${N_ROWS} ${N_COLS} ${DATA_FOLDER} ${IS_REAL} ${DATASET} 1 ${N_STRAGGLERS} 0 1 ${N_COLLECT} ${ADD_DELAY} ${UPDATE_RULE}