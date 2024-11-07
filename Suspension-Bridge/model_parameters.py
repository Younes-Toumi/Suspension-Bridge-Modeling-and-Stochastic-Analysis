import numpy as np

# --- Golden Gate Bridge --- #
n_cable_stays = 15 # The number of cables connecting the deck to the main cable
deck_length = 1280 # [m] Length of the deck
tower_height = 150 # [m] Height of the main tower // deck
m1 = 1609 # [kg/m] Mass per unit length | GGB -> Weight 22_200_000 kg-force, Length of cable 1405.9 m
m2 = 11_130 # [kg/m] Mass per unit length | GGB -> Weight of the deck is 139_790_000 kg-force and weight = mass * g)

# Estimated
T = 10e7 # [N] Tension of the cable stay 
E = 200e9 # [Pa] Young's modulus
I = 1/12 * 27 * 20**3 # [m^4] Moment of Inertia | 1/12 bh³ width is 27m, assuming height is 20m

b1 = 0.01 # [N.s/m] Damping coefficient
b2 = 0.01 # [N.s/m] Damping coefficient

k = 10e6 # [N.m] Stiffness of the cable stays | The coupling effect cable-bridge 

F = 38_000 # [N/m] Force per unit length | Exerted by external loads  


# --- Discretazation points --- # 
# Space
n_space_points = 101
dx = deck_length / (n_space_points - 1)

space_points = np.linspace(-deck_length/2, deck_length/2, n_space_points) + deck_length/2 # shifting it to have from [0, deck_lenght]

# Time
t_start, t_end, dt = 0, 10, 0.025
period = t_end - t_start
n_time_points = int((t_end-t_start) / dt + 1)

time_points = np.linspace(t_start, t_end, n_time_points)


# --- N° of Modes --- #
N_modes = 5

# --- External Forces --- #
F_cable = lambda t: - m1 *9.81 - F * np.sin(4*np.pi*t/period)
F_deck =  lambda t: - m2 *9.81 - F * np.sin(4*np.pi*t/period)  

# --- Initial Conditions --- # 
initial_conditions = [0.0, 0.0, 0.0, 0.0] # (IVP) y, z, dy_dt and dz_dt are 0 at t = 0