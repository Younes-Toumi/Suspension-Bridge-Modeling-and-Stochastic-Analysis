import numpy as np

def monte_carlo(stochastic_performance: np.ndarray) -> tuple[float, float]:
    """
    Perform Monte Carlo simulation for a given model.
    
    Parameters:
    ----------
    - stochastic_response : np.ndarray
        A list of model outputs for each sample.

    - limit_state : float
        Threshold where the structure reaches the critical domain `g(X) = 0`
    
    Returns:
    -------
    - prob_failure : float
        The probability of failure.

    - coef_var : float
        Corresponding coefficient of variation.
    """

    # Extracting number of samples
    n_samples = stochastic_performance.shape[0]


    failed_samples = np.sum(stochastic_performance < 0)
    prob_failure = failed_samples/n_samples
    
    coef_var = np.sqrt( (1 - prob_failure) / (n_samples*prob_failure) )

    return prob_failure, coef_var