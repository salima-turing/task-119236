from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Simulation parameters
N = 1000000  # size of the array
T = 1000  # number of timesteps

# Initialize data array (each process has its local part)
local_N = N // size + (1 if rank < N % size else 0)
local_data = np.zeros(local_N)

# Initialize each process with a different random seed for reproducibility
np.random.seed(rank)

# Simulation loop
for t in range(T):
    # Calculate local updates using some climate model logic
    local_data = local_data + np.random.rand(local_N)

    # Gather results from all processes into a single array
    if rank == 0:
        global_data = np.zeros(N)
    else:
        global_data = None

    comm.Gather(local_data, global_data, root=0)

    if rank == 0:
        # Example: In this script, we'll just print the total sum of the array at each timestep.
        print(f"Timestep {t}: Global sum = {global_data.sum()}")

    # Broadcast updates to all processes (if needed for next iteration)
    #comm.Bcast(global_data, root=0)
