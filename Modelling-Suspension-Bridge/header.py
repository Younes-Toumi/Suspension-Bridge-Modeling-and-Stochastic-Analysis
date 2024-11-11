import numpy as np
from scipy.integrate import solve_ivp
from model_parameters import *

def system_of_odes(t, S, n):
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

    ddz_ddt = -1 / m1 * (T * (n * np.pi / deck_length) ** 2 * z + b1 * dz_dt - k * (y - z)) + 1 / m1 * F_cable(t)
    ddy_ddt = -1 / m2 * (E * I * (n * np.pi / deck_length) ** 4 * y + b2 * dy_dt + k * (y - z)) + 1 / m2 * F_deck(t)

    dS_dt = [dz_dt, ddz_ddt, dy_dt, ddy_ddt]
    return dS_dt

def solve_suspension_bridge():
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
            args=(n,)
        )

        phi_n = np.sin(n * np.pi * space_points / deck_length)  # Modal shape
        z = solution.y[0, :]  # Main cable
        y = solution.y[2, :]  # Deck

        # Aggregate the mode's displacements
        main_cable_displacement += np.outer(z, phi_n)
        deck_displacement += np.outer(y, phi_n)

    return main_cable_displacement, deck_displacement

def update_animation(frame, axis, plot_main_cable, plot_deck, main_cable_coords, main_cable_displacement,
                     deck_coords, deck_displacement, *plot_cable_stays):
    """
    Updates the plot data for each animation frame.

    Input:
    - frame (int): Current frame number.
    - axis (matplotlib.axes.Axes): Axis object for the plot.
    - plot_main_cable (Line2D): Line object for the main cable.
    - plot_deck (Line2D): Line object for the deck.
    - main_cable_coords (ndarray): Initial coordinates of the main cable.
    - main_cable_displacement (ndarray): Displacements of the main cable over time.
    - deck_coords (ndarray): Initial coordinates of the deck.
    - deck_displacement (ndarray): Displacements of the deck over time.
    - plot_cable_stays (list[Line2D]): Line objects for the cable stays.

    Output:
    - tuple : Updated line objects.
    """
    # Update the positions of the main cable and deck for this frame
    updated_main_cable_coords = main_cable_coords + main_cable_displacement[frame]
    updated_deck_coords = deck_coords + deck_displacement[frame]
    
    updated_cable_stays_coords = []
    for i in range(n_cable_stays):
        index = int((i + 1) * len(space_points) / (n_cable_stays + 1))
        updated_cable_coords_x = [space_points[index], space_points[index]]
        updated_cable_coords_y = [updated_deck_coords[index], updated_main_cable_coords[index]]
        updated_cable_stays_coords.append([updated_cable_coords_x, updated_cable_coords_y])

    updated_cable_stays_coords = np.array(updated_cable_stays_coords)
    plot_main_cable.set_data(space_points, updated_main_cable_coords)
    plot_deck.set_data(space_points, updated_deck_coords)
    
    # Update the cable stays in the plot
    for i, cable_plot_i in enumerate(plot_cable_stays):
        cable_plot_i.set_data(updated_cable_stays_coords[i])

    axis.set_title(
        f"Suspension Bridge Simulation | Progress: {(frame + 1) / n_time_points:.1%} "
        f"Max displacement main cable: {np.max(np.abs(main_cable_displacement[frame])):.2f} [m] | deck {np.max(np.abs(deck_displacement[frame])):.2f} [m]"
    )

    return plot_main_cable, plot_deck, *plot_cable_stays

def plot_bridge(axis, deck_length, tower_height, space_points, deck_coords, main_cable_coords, cable_stays_coords):
    """
    Plots the initial suspension bridge structure with main cable, deck, and towers.

    Input:
    - axis (matplotlib.axes.Axes): Axis object for the plot.
    - deck_length (float): Length of the deck.
    - tower_height (float): Height of the towers.
    - space_points (ndarray): Spatial coordinates along the bridge deck.
    - deck_coords (ndarray): Initial coordinates of the deck.
    - main_cable_coords (ndarray): Initial coordinates of the main cable.
    - cable_stays_coords (ndarray): Coordinates of the cable stays.

    Output:
    tuple : Line objects for each plotted component of the bridge.
    """
    main_tower_1 = [[0] * 2, [-tower_height, tower_height]]
    main_tower_2 = [[deck_length] * 2, [-tower_height, tower_height]]
    
    plot_deck, = axis.plot(space_points, deck_coords, color='black', lw=3, label='Deck')
    plot_main_cable, = axis.plot(space_points, main_cable_coords, color='black', lw=2, linestyle='--', label='Main Cable')

    plot_main_tower_1, = axis.plot(*main_tower_1, 'o-',             color='black',  lw=10,                  label='Tower 1')
    plot_main_tower_2, = axis.plot(*main_tower_2, 'o-',             color='black',  lw=10,                  label='Tower 2')

    # Plot each cable stay
   
    plot_cable_stays = axis.plot(*cable_stays_coords, 'o-', color='black', lw=2, ms=2)


    return plot_deck, plot_main_cable, plot_main_tower_1, plot_main_tower_2, *plot_cable_stays