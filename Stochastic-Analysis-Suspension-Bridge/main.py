import numpy as np
from monte_carlo import monte_carlo
from form import form
from header import solve_suspension_bridge

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from header import *
from model_parameters import *
from functions import *
from model import * 

n_samples = 100
# Solve the suspension bridge system to get displacements of main cable and deck over time
limit_state = -3.3

rv_T, rv_E, rv_m1, rv_b1, rv_m2, rv_b2, rv_k, rv_F = generate_random_variables(n_samples)

rv_F_cable = lambda t: - rv_m1 *9.81 - rv_F * np.sin(np.pi*t/period)
rv_F_deck = lambda t:  - rv_m2 *9.81 - rv_F * np.sin(np.pi*t/period)  


# We regroup our random variables into one matrix for later use
rv_matrix = np.array([
    rv_T,
    rv_E,
    rv_m1,
    rv_b1,
    rv_m2,
    rv_b2,
    rv_k,
    rv_F_cable,
    rv_F_deck
], dtype=object).T

if __name__ == '__main__':
    # Monte Carlo #
    response, performance = evaluate_model(rv_matrix, solve_suspension_bridge, limit_state, use_multiprocessing=True)
    
    # prob_failure, coef_var = monte_carlo(response) # Works