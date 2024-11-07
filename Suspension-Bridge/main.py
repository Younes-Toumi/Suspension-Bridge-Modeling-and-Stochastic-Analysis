from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from header import *
from model_parameters import *

# Solve the suspension bridge system to get displacements of main cable and deck over time
main_cable_displacement, deck_displacement = solve_suspension_bridge()

# Approximating the main cable's initial shape as a parabolic curve (f(x) = ax^2 + bx + c)
a = 2 * (tower_height - 3) / 2 * 1 / (deck_length / 2) ** 2
b = 0
c = 2  # Initial height of main cable at x = 0
main_cable_coords = a * (space_points - deck_length / 2) ** 2 + b * (space_points - deck_length / 2) + c

# Initialize deck coordinates (flat at y = 0)
deck_coords = np.zeros_like(space_points)

# Precompute coordinates of cable stays connecting the deck and main cable
cable_stays_coords = []
for i in range(n_cable_stays):
    index = int((i + 1) * len(space_points) / (n_cable_stays + 1))
    cable_stay_coord_x = [space_points[index], space_points[index]]
    cable_stay_coord_y = [deck_coords[index], main_cable_coords[index]]
    cable_stays_coords.append(cable_stay_coord_x)
    cable_stays_coords.append(cable_stay_coord_y)

cable_stays_coords = np.array(cable_stays_coords)

# Plot and animate the suspension bridge simulation
fig, axis = plt.subplots()
plot_deck, plot_main_cable, plot_main_tower_1, plot_main_tower_2, *plot_cable_stays = plot_bridge(
    axis, deck_length, tower_height, space_points, deck_coords, main_cable_coords, cable_stays_coords
)

# Define plot axis limits
axis.set_xlim([0, deck_length])
axis.set_ylim([-tower_height, tower_height])

# Animation function that updates the bridge's visualization for each time frame
animation = FuncAnimation(
    fig=fig,
    func=update_animation,
    interval=25,
    frames=range(0, n_time_points),
    fargs=(
        axis,
        plot_main_cable,
        plot_deck,
        main_cable_coords,
        main_cable_displacement,
        deck_coords,
        deck_displacement,
        *plot_cable_stays
    ),
    blit=False
)

plt.grid()
plt.show()
