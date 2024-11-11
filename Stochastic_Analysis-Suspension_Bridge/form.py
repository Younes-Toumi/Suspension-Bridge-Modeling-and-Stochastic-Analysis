from scipy.stats import norm
from mathematical_model import mathematical_model
from header import evaluate_model

import numpy as np

def form(rv_matrix, limit_state, delta_z, rv_sns_array):
    
    n_samples, n_random_variables = rv_matrix.shape

    # Initial geuss and Parameters of FORM
    Z = np.zeros(n_random_variables)
    Z_array = [Z]
    beta_array = [np.inf]

    for i in range(20): # Number of iterations
        # generate samples in SNS Z
        Z = np.array([rv_sns(Z[j]) for j, rv_sns in enumerate(rv_sns_array)])
        # generate samples in SNS Z +/- delta_z
        
        # Create perturbed matrices
        Z_plus_delta = np.tile(Z, (n_random_variables, 1)) + delta_z*np.eye(n_random_variables)
        Z_minus_delta = np.tile(Z, (n_random_variables, 1)) - delta_z*np.eye(n_random_variables)

        # evaluate performance function in SNS h(Z)
        _, h_Z =               evaluate_model(Z,               mathematical_model, limit_state, use_multiprocessing = False)
        _, h_Z_plus_delta =    evaluate_model(Z_plus_delta,    mathematical_model, limit_state, use_multiprocessing = False)
        _, h_Z_minus_delta =   evaluate_model(Z_minus_delta,   mathematical_model, limit_state, use_multiprocessing = False)

        dh_dZ = (h_Z_plus_delta - h_Z_minus_delta)/(2*delta_z)

        # compute gradient of performance function h(Z)
        dh_dZ = np.array(dh_dZ)

        # Reliability index (beta) update
        beta_new = (h_Z - np.dot(dh_dZ, Z)) / np.linalg.norm(dh_dZ)

        # Important direction (alpha)
        alpha_new = - dh_dZ / np.linalg.norm(dh_dZ)

        # Update design point (z)
        Z_new = alpha_new * beta_new

        
        Z = Z_new
        beta_array.append(beta_new)
        Z_array.append(Z_new)

    prob_failure = 1 - norm._cdf(beta_array[-1])

    return prob_failure

