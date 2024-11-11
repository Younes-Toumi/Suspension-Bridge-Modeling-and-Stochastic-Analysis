import numpy as np
from typing import Callable
from multiprocessing import Pool

def mathematical_model(x1, x2):

    return x1 * x2


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