from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set simulation parameters
nx = 100  # Number of grid points in x-direction
nt = 1000  # Number of time steps
dx = 1.0 / nx  # Spatial step size
dt = 0.01  # Time step size

# Initialize temperature array
T = np.zeros(nx + 1)
if rank == 0:
    T[0] = 100.0  # Set boundary condition at x=0

# Bcast initial temperature array to all nodes
comm.Bcast(T, root=0)

# Simulation loop
for t in range(nt):
    # Calculate local temperature updates
    local_T = np.copy(T)
    for i in range(1, nx):
        local_T[i] = T[i] + dt * (T[i + 1] - 2.0 * T[i] + T[i - 1]) / dx**2

    # Send and receive updated temperatures with neighboring nodes
    if rank > 0:
        comm.Send(local_T[1:], dest=rank - 1, tag=0)
        comm.Recv(local_T[0], source=rank - 1, tag=0)
    if rank < size - 1:
        comm.Send(local_T[:-1], dest=rank + 1, tag=1)
        comm.Recv(local_T[-1], source=rank + 1, tag=1)

    # Update global temperature array
    T = local_T

# Gather all results on root node
all_T = comm.gather(T, root=0)

if rank == 0:
    # Plot the final temperature distribution
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, nx + 1)
    T_final = np.concatenate(all_T)
    plt.plot(x, T_final)
    plt.xlabel("x")
    plt.ylabel("Temperature")
    plt.title("1D Heat Equation Simulation using MPI")
    plt.show()

    # Print simulation time
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time:.4f} seconds")
