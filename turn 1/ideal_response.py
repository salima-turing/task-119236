import numpy as np
from mpi4py import MPI
import time

def climate_model_step(local_temperature: np.ndarray, dt: float, k: float) -> np.ndarray:
    """
    A simple climate model step function.
    Args:
        local_temperature: Array of local temperatures.
        dt: Time step.
        k: Heat diffusion constant.
    Returns:
        Updated local temperatures.
    """
    new_temp = local_temperature.copy()
    for i in range(1, len(local_temperature) - 1):
        new_temp[i] = local_temperature[i] + dt * k * (
                local_temperature[i + 1] - 2 * local_temperature[i] + local_temperature[i - 1])
    return new_temp


def main() -> None:
    comm = MPI.COMM_WORLD
    size: int = comm.Get_size()
    rank: int = comm.Get_rank()

    # Model Parameters
    total_grid_points: int = 1000
    num_time_steps: int = 100
    dt: float = 0.01
    k: float = 0.1

    # Initialize grid
    local_size: int = total_grid_points // size
    start: int = rank * local_size
    end: int = min((rank + 1) * local_size, total_grid_points)

    local_temperature: np.ndarray = np.zeros(local_size) + 20.0

    if rank == 0:
        # Initial condition for the first process
        local_temperature[0] = 30.0

    # Broadcast initial conditions to all processes
    comm.Bcast(local_temperature, root=0)

    start_time: float = time.time()

    for t in range(num_time_steps):
        # Send right boundary data to next process
        if rank < size - 1:
            comm.Send([local_temperature[-1], MPI.DOUBLE], dest=rank + 1, tag=11)
            comm.Recv([local_temperature[local_size - 1], MPI.DOUBLE], source=rank + 1, tag=11)

        # Send left boundary data to previous process
        if rank > 0:
            comm.Send([local_temperature[0], MPI.DOUBLE], dest=rank - 1, tag=22)
            comm.Recv([local_temperature[0], MPI.DOUBLE], source=rank - 1, tag=22)

        local_temperature = climate_model_step(local_temperature, dt, k)

    comm.Barrier()
    end_time: float = time.time()

    if rank == 0:
        total_time: float = end_time - start_time
        print(f"Simulation completed in {total_time:.4f} seconds")
        print(f"Total grid points: {total_grid_points}")
        print(f"Parallel efficiency: {total_grid_points / (total_time * size):.2f} grid points per second per process")


if __name__ == "__main__":
    main()
