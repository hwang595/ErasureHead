# No. of workers
N_PROCS=11

# No. of stragglers in our coding schemes
N_STRAGGLERS = 4

# For partially coded version: how many pieces of workload will one worker be handling.
N_PARTITIONS=10

# Switch to enable partial coded schemes
PARTIAL_CODED=0

# Path to folder containing the data folders
DATA_FOLDER=/straggdata/

IS_REAL = 1

DATASET = amazon-dataset
N_ROWS=26210
N_COLS=241915

# Note that DATASET is automatically set to artificial-data/ (n_rows)x(n_cols)/... if IS_REAL is set to 0 \
 or artificial-data/partial/ (n_rows)x(n_cols)/... if PARTIAL_CODED is also set to 1

generate_random_data:
	python ./src/generate_data.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

arrange_real_data:
	python ./src/arrange_real_data.py $(N_PROCS) $(DATA_FOLDER) $(DATASET) $(N_STRAGGLERS) $(N_PARTITIONS) $(PARTIAL_CODED)

naive:   
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) 0 0

cyccoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 0

repcoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 1

avoidstragg:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0 2

partialrepcoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(N_PARTITIONS) 1

partialcyccoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) $(N_PARTITIONS) 0
