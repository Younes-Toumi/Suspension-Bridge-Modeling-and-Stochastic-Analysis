import numpy as np
from scipy.integrate import solve_ivp
from model_parameters import *
import numba

@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def system_of_odes(t, S, n, T, E, m1, b1, m2, b2, k, F):
    """
    Defines the system of differential equations for the suspension bridge model.

    Input:
        - t (float): Current time.
        - S (list[float]): System state [z, dz_dt, y, dy_dt] where:
            + z, dz_dt are main cable displacement and velocity,
            + y, dy_dt are deck displacement and velocity.
        - n (int): Mode number for modal analysis.

    Output:
        - dS_dt (list[float]): Derivatives [dz_dt, ddz_ddt, dy_dt, ddy_ddt].
    """
    z, dz_dt, y, dy_dt = S  

    F_cable = - m1 *9.81 - F * np.sin(np.pi*t/period)
    F_deck =  - m2 *9.81 - F * np.sin(np.pi*t/period) 

    ddz_ddt = -1 / m1 * (T * (n * np.pi / deck_length) ** 2 * z + b1 * dz_dt - k * (y - z)) + 1 / m1 * F_cable
    ddy_ddt = -1 / m2 * (E * I * (n * np.pi / deck_length) ** 4 * y + b2 * dy_dt + k * (y - z)) + 1 / m2 * F_deck

    dS_dt = [dz_dt, ddz_ddt, dy_dt, ddy_ddt]
    return dS_dt


def mathematical_model(T,  E,  m1, b1, m2, b2, k,  F):
    """
    Solves the suspension bridge system over time to calculate main cable and deck displacements.

    Output:
        - main_cable_displacement (ndarray): Array of displacements of the main cable at each point in time and space.
        - deck_displacement (ndarray): Array of displacements of the deck at each point in time and space.
    """
    deck_displacement = np.zeros((n_time_points, n_space_points))
    main_cable_displacement = np.zeros((n_time_points, n_space_points))

    # Solve the system for each mode and accumulate the results
    for n in range(1, N_modes + 1):
        solution = solve_ivp(
            fun=system_of_odes,
            t_span=(t_start, t_end),
            y0=initial_conditions,
            t_eval=time_points,
            args=(n, T, E, m1, b1, m2, b2, k, F),
            rtol=1e-4,
            atol=1e-6,
            method='DOP853'
        )

        phi_n = np.sin(n * np.pi * space_points / deck_length)  # Modal shape
        z = solution.y[0, :]  # Main cable
        y = solution.y[2, :]  # Deck

        # Aggregate the mode's displacements
        main_cable_displacement += np.outer(z, phi_n)
        deck_displacement += np.outer(y, phi_n)

    return main_cable_displacement, deck_displacement