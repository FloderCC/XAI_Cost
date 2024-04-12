# running device MIPS
real_mips = 60000   # Our CPU regarding 7z:

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99
simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100

def simulate_energy_consumption(training_time, cpu_usage):
    """
    :param training_time: the code execution time
    :param cpu_usage: the avg cpu used
    :return: the estimated energy consumption in joules
    """
    simulated_training_time = training_time * (real_mips / simulated_mips)
    energy_joules = simulated_training_time * (simulated_idle_energy + aux_constant * cpu_usage)
    return energy_joules
