import numpy as np
from mathematical_model import mathematical_model

from monte_carlo import monte_carlo
from form import form
from header import *
from model_parameters import *
from header import * 

n_samples = 1000
# Solve the suspension bridge system to get displacements of main cable and deck over time
limit_state = -3.3

rv_T, rv_E, rv_m1, rv_b1, rv_m2, rv_b2, rv_k, rv_F = generate_random_variables(n_samples)

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

if __name__ == '__main__':
    # Monte Carlo #
    import time
    t1 = time.time()
    response, performance = evaluate_model(rv_matrix, mathematical_model, limit_state, use_multiprocessing=True)
    t2 = time.time()
    print(f"Solved in {t2-t1}")
    # prob_failure, coef_var = monte_carlo(response) # Works