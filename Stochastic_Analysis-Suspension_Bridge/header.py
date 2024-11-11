import numpy as np
from typing import Callable
from multiprocessing import Pool
from scipy.stats import norm
from model_parameters import *

def evaluate_model(rv_matrix: np.ndarray, mathematical_model: Callable, limit_state: float, use_multiprocessing: bool = False) -> np.ndarray:
    """
    Perform Monte Carlo simulation for a given model.
    
    Parameters:
    ----------
    - rv_matrix : np.ndarray
        Matrix of random variable samples with shape (n_samples, n_variables).

    - evaluate_model : Callable
        The model to evaluate. It should accept the input vector as arguments.

    - limit_state : float
        Threshold where the structure reaches the critical domain `g(X) = 0`

    - use_multiprocessing : bool, optional
        If True, use multiprocessing for faster computation (default is False).
    
    Returns:
    -------
    - stochastic_response : np.ndarray
        A list of model outputs for each sample.
    
    - stochastic_perofrmance : np.ndarray
        The performance function g(X) for each sample.
    """

    if rv_matrix.ndim == 1: # Treating the special case where we want to evaluate the model for one sample, this will be usefull when calling the `form` function
        stochastic_response = mathematical_model(*rv_matrix)

    else:

        n_samples, n_random_variables = rv_matrix.shape

        if use_multiprocessing:
        # Using multiprocessing for parallel evaluation
            with Pool() as pool:
                stochastic_response = pool.starmap(mathematical_model, rv_matrix[:n_samples])
        else:
            # Using a standard loop for evaluation
            stochastic_response = [mathematical_model(*rv_matrix[i, :]) for i in range(n_samples)]

    stochastic_response = np.array(stochastic_response)
    stochastic_performance = limit_state - stochastic_response

    return stochastic_response, stochastic_performance 




def gumbel(E_X, std_X, N_samples):
    """
    A function that aims to generate a random variable that follows a gumbel 
    distribution.
    """

    beta = np.sqrt(6) * std_X / np.pi
    mu = E_X - 0.57721 * beta
    return np.random.gumbel(mu, beta, N_samples)



def generate_random_variables(N_samples):
    """
    A function that generates random variables of the model
    """

    T_mu, T_std = T, 0.05*T
    rv_T = np.random.normal(T_mu, T_std, N_samples)

    E_mu, E_std = E, 0.05*E
    
    m = np.log( E_mu**2 / (np.sqrt(E_std**2 + E_mu**2))  )
    v = np.sqrt(np.log(E_std**2/E_mu**2 + 1))
    
    rv_E = np.random.lognormal(m, v, N_samples)

    m1_mu, m1_std = m1, 0.05*m1
    rv_m1 = np.random.normal(m1_mu, m1_std, N_samples)

    b1_mu, b1_std = b1, 0.05*b1
    rv_b1 = np.random.normal(b1_mu, b1_std, N_samples)
    rv_b1[rv_b1 < 0.5*b1] = 0.5*b1

    m2_mu, m2_std = m2, 0.05*m2
    rv_m2 = np.random.normal(m2_mu, m2_std, N_samples)

    b2_mu, b2_std = b2, 0.05*b2
    rv_b2 = np.random.normal(b2_mu, b2_std, N_samples)
    rv_b2[rv_b1 < 0.5*b2] = 0.5*b2

    k_mu, k_std = k, 0.05*k
    rv_k = np.random.normal(k_mu, k_std, N_samples)

    F_mu, F_std = F, 0.05*F
    rv_F = gumbel(F_mu, F_std, N_samples)

    return rv_T, rv_E, rv_m1, rv_b1, rv_m2, rv_b2, rv_k, rv_F

def normal_to_sns(rv_mu, rv_std, z):
    return rv_mu + rv_std * z


def gumbel_to_sns(E_X, std_X, z):
    
    beta = np.sqrt(6) * std_X / np.pi
    mu = E_X - 0.57721 * beta
    
    return mu - beta*np.log( -np.log( norm.cdf(z) ) )


def lognormal_to_sns(E_X, std_X, z):
    
    m = np.log( E_X**2 / (np.sqrt(std_X**2 + E_X**2))  )
    v = np.sqrt(np.log(std_X**2/E_X**2 + 1))
    
    return np.exp(v*z+m)

