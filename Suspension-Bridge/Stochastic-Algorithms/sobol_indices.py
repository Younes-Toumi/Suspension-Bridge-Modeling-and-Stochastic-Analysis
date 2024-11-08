import time
import multiprocessing

from Header.model_solver import solve_sample
from Header.PARAMETERS import *
import Header.functions as func

import numpy as np
import matplotlib.pyplot as plt


N_samples = 70_000 # Number of samples 20_000


rv_T_A, rv_E_A, rv_m1_A, rv_b1_A, rv_m2_A, rv_b2_A, rv_k_A, rv_F_A = func.generate_random_variables(N_samples)
rv_T_B, rv_E_B, rv_m1_B, rv_b1_B, rv_m2_B, rv_b2_B, rv_k_B, rv_F_B = func.generate_random_variables(N_samples)

#  Building a matrix of random variables

A = np.array([
    rv_T_A, rv_E_A, rv_m1_A, rv_b1_A, rv_m2_A, rv_b2_A, rv_k_A, rv_F_A
]).T

B = np.array([
    rv_T_B, rv_E_B, rv_m1_B, rv_b1_B, rv_m2_B, rv_b2_B, rv_k_B, rv_F_B
]).T



N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)


if __name__=='__main__':
    with multiprocessing.Pool() as pool:
        # Parallelize the loop
        results_A = pool.starmap(solve_sample, [(i, rv_T_A, rv_E_A, rv_m1_A, rv_b1_A, rv_m2_A, rv_b2_A, rv_k_A, rv_F_A, dt, N_time, time_points, period, N_space, space_points, L, I) for i in range(N_samples)])
        print('a')
        results_B = pool.starmap(solve_sample, [(i, rv_T_B, rv_E_B, rv_m1_B, rv_b1_B, rv_m2_B, rv_b2_B, rv_k_B, rv_F_B, dt, N_time, time_points, period, N_space, space_points, L, I) for i in range(N_samples)])
        print('b')

        y_A = np.array(results_A)
        y_B = np.array(results_B)


    f_0 = np.mean(y_A)

    S = np.zeros(8)
    ST = np.zeros(8)

    for i in range(8):
        C = np.copy(B)
        C[:, i] = A[:, i]

        rv_T_C, rv_E_C, rv_m1_C, rv_b1_C, rv_m2_C, rv_b2_C, rv_k_C, rv_F_C = C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4], C[:, 5], C[:, 6], C[:, 7]
        
        with multiprocessing.Pool() as pool:
            # Parallelize the loop

            results_C = pool.starmap(solve_sample, [(i, rv_T_C, rv_E_C, rv_m1_C, rv_b1_C, rv_m2_C, rv_b2_C, rv_k_C, rv_F_C, dt, N_time, time_points, period, N_space, space_points, L, I) for i in range(N_samples)])
            y_C = np.array(results_C)

        S[i] =      ( np.mean(y_A * y_C) - f_0**2 ) / ( np.mean(y_A**2) - f_0**2 )
        ST[i] = 1 - ( np.mean(y_B * y_C) - f_0**2 ) / ( np.mean(y_A**2) - f_0**2 )

        print(f"S{i} = {S[i]:.3f} | ST{i} = {ST[i]:.3f} ")

    import pickle
    with open('Results\\sobol_result.pickle', 'wb') as file:
        
        stored_data = {
            'first_order': S,
            'total_order': ST,

        }
        
        pickle.dump(stored_data, file)

    fig, axis = plt.subplots()
    index = np.arange(8)
    bar_width = 0.35

    rects1 = plt.bar(index, S, bar_width,
    color='blue',
    label='First Order',
    edgecolor='black')

    rects2 = plt.bar(index + bar_width, ST, bar_width,
    color='orange',
    label='Total Order',
    edgecolor='black')


    plt.xlabel('Random variables')
    plt.ylabel("Sobol's Indices")
    plt.title('Sensitivity analysis')
    plt.xticks(index + bar_width, ("Tension T",  "Young's modulus E", "m1 cable", "b1 cable", "m2 deck", "b2 deck", "coupling k", "External force F"))
    plt.legend()

    plt.tight_layout()
    plt.show()


# S0 = 0.032 | ST0 = -0.067 
# S1 = 0.702 | ST1 = 0.588 
# S2 = 0.044 | ST2 = -0.069 
# S3 = 0.032 | ST3 = -0.067 
# S4 = 0.315 | ST4 = 0.255 
# S5 = 0.032 | ST5 = -0.067 
# S6 = 0.032 | ST6 = -0.067 
# S7 = 0.042 | ST7 = -0.012 