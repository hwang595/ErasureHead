# No. of workers
N_PROCS=21
# No. of stragglers in our coding schemes
N_STRAGGLERS=5
# For partially coded version: how many pieces of workload will one worker be handling.
N_PARTITIONS=20
# Switch to enable partial coded schemes
PARTIAL_CODED=0
# Path to folder containing the data folders
DATA_FOLDER=./straggdata/
IS_REAL=1
DATASET=amazon-dataset
N_ROWS=26215
N_COLS=241915

python ./src/arrange_real_data.py ${N_PROCS} ${DATA_FOLDER} ${DATASET} ${N_STRAGGLERS} ${N_PARTITIONS} ${PARTIAL_CODED}