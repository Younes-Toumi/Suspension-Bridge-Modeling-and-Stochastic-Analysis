import time
import multiprocessing
import pickle

from Header.model_solver import solve_sample
from Header.PARAMETERS import *
import Header.functions as func

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------------------------------- # 
# 1. Defining our random variables -----------------------------------------------------------------             # 
# ---------------------------------------------------------------------------------------------------------------------- # 


N_samples = 3_000_000 # Number of samples

rv_T, rv_E, rv_m1, rv_b1, rv_m2, rv_b2, rv_k, rv_F = func.generate_random_variables(N_samples)

# We regroup our random variables into one matrix for later use
rv_matrix = np.array([
    rv_T,
    rv_E,
    rv_m1,
    rv_b1,
    rv_m2,
    rv_b2,
    rv_k,
    rv_F
]).T

# ---------------------------------------------------------------------------------------------------------------------- # 
# 2. Descritizing our bridge & specifying parameters for the solver --------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- # 

N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)


# ---------------------------------------------------------------------------------------------------------------------- # 
# 3. Solving our model and running calculations ----------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- # 

if __name__ == '__main__':

    max_displacement = -3.3 # [m] Maximum downward deflection of -3.3 [m]
    
    t1 = time.time()
    with multiprocessing.Pool() as pool:
        # Parallelize the loop
        u_array = pool.starmap(solve_sample, [(i, rv_T, rv_E, rv_m1, rv_b1, rv_m2, rv_b2, rv_k, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I) for i in range(N_samples)])

    t2 = time.time()

    u_array = np.array(u_array)

    failed_samples = np.sum(u_array < max_displacement)
    Pf = failed_samples/N_samples

    print(f"Solved the model in: {(t2-t1):.3f} [s]")
    print(f"Out of {N_samples:_} samples, {failed_samples:_} failed -> Probability of failure {Pf:.3e}")

    with open('Results\\stored_result.pickle', 'wb') as file:
        
        stored_data = {
            'N_samples': N_samples,
            'rv_matrix': rv_matrix,
            'u_array': u_array,
            'Pf': Pf
        }
        
        pickle.dump(stored_data, file)



# Solved the model in: 10637.751 [s]
# Out of 3_000_000 samples, 14 failed -> Probability of failure 4.667e-06