from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# Simple climate model simulation function
def simulate_climate(local_data, steps):
    for step in range(steps):
        local_data = local_data * 1.01
    return local_data


if __name__ == "__main__":
    # Initialize data
    total_data_size = 10000
    local_data_size = total_data_size // size
    local_data = np.random.rand(local_data_size)

    start_time = time.time()

    # Parallel simulation
    local_result = simulate_climate(local_data, 100)

    # Gather results from all processes
    results = comm.gather(local_result, root=0)

    if rank == 0:
        global_result = np.concatenate(results)
        print(f"Simulation completed in {time.time() - start_time} seconds")
    # Add any post-processing here if needed
