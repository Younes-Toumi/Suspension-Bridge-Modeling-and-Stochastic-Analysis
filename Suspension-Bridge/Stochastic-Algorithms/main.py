import numpy as np
from monte_carlo import monte_carlo
from form import form
from model import *

n_samples = 10
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)

rv_matrix = np.array([x1, x2]).T

limit_state = 0.7


if __name__ == '__main__':
    # Monte Carlo #
    # response, performance = evaluate_model(rv_matrix, mathematical_model, limit_state, use_multiprocessing=True)
    # prob_failure, coef_var = monte_carlo(response) # Works

    # FORM #
    delta_z = 0.5
    rv_sns_array = [
        lambda z: 0 + 1*z,
        lambda z: 0 + 1*z
    ]

    prob_failure = form(rv_matrix, limit_state, delta_z, rv_sns_array)
