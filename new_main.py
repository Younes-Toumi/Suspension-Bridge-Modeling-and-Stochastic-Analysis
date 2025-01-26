from Functions.header import *
from Classes.Network import Network
import matplotlib.pyplot as plt
import numpy as np


# Part 1: Modelling the Network
#####################################################################################################################################################

# Loading data from Germany
german_data_from_internet_filepath = "Data//Germany//germany_data_from_internet.txt"  # File containing data obtained from internet
german_coordinates_filepath = "Data//Germany//germany_city_coordinates.txt"  # File containing city coordinates (latitude, longitude)

# Loading data from Japan
japan_data_from_internet_filepath = "Data//Japan//japan_data_from_internet.txt"  # File containing data obtained from internet
japan_coordinates_filepath = "Data//Japan//japan_city_coordinates.txt"  # File containing city coordinates (latitude, longitude)


# Load the network and coordinates data
germany_city_coords = load_city_coordinates(german_coordinates_filepath)
germany_network_data = load_network_data(german_data_from_internet_filepath, germany_city_coords, country='Germany')

japan_city_coords = load_city_coordinates(japan_coordinates_filepath)
japan_network_data = load_network_data(japan_data_from_internet_filepath, japan_city_coords, country='Japan')


# Building the Networks
germany_network = Network('Germany', germany_network_data)
japan_network = Network('Japan', japan_network_data)

# Printing the two network objects (__repr__)
print(germany_network)
print(100*"-")
print(japan_network)


# Visualization of the Network
#####################################################################################################################################################

# Creating the figure and axis that will hold the network
fig_network, (axis_germany, axis_japan) = plt.subplots(ncols=2, nrows=1)

Network.plot_network(
    germany_network.network,
    axis_germany,
    germany_city_coords,
    node_colors='blue',
    edge_widths=germany_network.edge_frequency
    )

Network.plot_network(
    japan_network.network,
    axis_japan,
    japan_city_coords,
    node_colors='blue',
    edge_widths=japan_network.edge_frequency
    )


# Adding additional elements to the plots (background image of the country)
enhance_plot = True
if enhance_plot:
    # Longitude and Latitude can be obtained from: https://gist.github.com/graydon/11198540
    germany_lon_lat_boundary = np.array([[5.60, 15.50], [46.90, 55.55]]) #  Corresponds to [[lon_min, lon_max], [lat_min, lat_max]]
    germany_img_path = 'Data//Germany//germany_map.png'

    japan_lon_lat_boundary = np.array([[129.527, 142.087], [31.378, 41.406]])
    japan_img_path = 'Data//Japan//japan_map.png'

    import matplotlib.image as mpimg

    # Load and display the map image
    germany_img = mpimg.imread(germany_img_path)
    axis_germany.imshow(germany_img, extent=germany_lon_lat_boundary.flatten()) # The extent should be [left, right, bottom, top] i.e., [lon_min, lon_max, lat_min, lat_max]
    axis_germany.set_xlim(germany_lon_lat_boundary[0])  # Set xlim for longitude
    axis_germany.set_ylim(germany_lon_lat_boundary[1])  # Set ylim for latitude

    # Load and display the map image
    japan_img = mpimg.imread(japan_img_path)
    axis_japan.imshow(japan_img, extent=japan_lon_lat_boundary.flatten()) # The extent should be [left, right, bottom, top] i.e., [lon_min, lon_max, lat_min, lat_max]
    axis_japan.set_xlim(japan_lon_lat_boundary[0])  # Set xlim for longitude
    axis_japan.set_ylim(japan_lon_lat_boundary[1])  # Set ylim for latitude





# Centrality of the Network
###########################

fig_graph_w_centrality, (axis_germany_w_centrality, axis_japan_w_centrality) = plt.subplots(ncols=2, nrows=1)

Network.plot_network(
    germany_network.network,
    axis_germany_w_centrality,
    germany_city_coords,
    node_colors=list(germany_network.local_properties['closeness_centrality'].values()),
    edge_widths=germany_network.edge_frequency
    )


Network.plot_network(
    japan_network.network,
    axis_japan_w_centrality,
    japan_city_coords,
    node_colors=list(japan_network.local_properties['closeness_centrality'].values()),
    edge_widths=japan_network.edge_frequency
    )

# Degree distribution of the Network
####################################

fig_graph_degree, (axis_germany_degree, axis_japan_degree) = plt.subplots(ncols=2, nrows=1)

# Germany
axis_germany_degree.hist(germany_network.local_properties['degrees'], bins=15, color='skyblue', edgecolor='black')
axis_germany_degree.set_title('Histogram of Node Degrees - Germany')
axis_germany_degree.set_xlabel('Degree')
axis_germany_degree.set_ylabel('Frequency')

# Japan
axis_japan_degree.hist(japan_network.local_properties['degrees'], bins=10, color='skyblue', edgecolor='black')
axis_japan_degree.set_title('Histogram of Node Degrees - Japan')
axis_japan_degree.set_xlabel('Degree')
axis_japan_degree.set_ylabel('Frequency')


plt.show()

# # -------------------------------- Measuring percolation -------------------------------- #
# # fig_percolation, (axis_percolation_germany, axis_percolation_japan) = plt.subplots(ncols=2, nrows=1)


# # germany_removed_fraction_targeted, germany_lcc_sizes_targeted = germany_network.percolation_simulation(strategy="targeted")
# # japan_removed_fraction_targeted, japan_lcc_sizes_targeted = japan_network.percolation_simulation(strategy="targeted")


# # germany_removed_fractions_random = []
# # germany_lccs_sizes_random = []

# # japan_removed_fractions_random = []
# # japan_lccs_sizes_random = []

# # for i in range(200):
# #     germany_removed_fraction_random, germany_lcc_sizes_random = germany_network.percolation_simulation(strategy="random")
# #     japan_removed_fraction_random, japan_lcc_sizes_random = japan_network.percolation_simulation(strategy="random")
    
# #     germany_removed_fractions_random.append(germany_removed_fraction_random)
# #     germany_lccs_sizes_random.append(germany_lcc_sizes_random)

# #     japan_removed_fractions_random.append(japan_removed_fraction_random)
# #     japan_lccs_sizes_random.append(japan_lcc_sizes_random)




# # # Convert lists to numpy arrays for easier manipulation
# # germany_removed_fractions_random = np.array(germany_removed_fractions_random)
# # germany_lccs_sizes_random = np.array(germany_lccs_sizes_random)

# # japan_removed_fractions_random = np.array(japan_removed_fractions_random)
# # japan_lccs_sizes_random = np.array(japan_lccs_sizes_random)

# # # Calculate the upper and lower bounds for the shaded region
# # germany_lcc_lower = np.min(germany_lccs_sizes_random, axis=0)
# # germany_lcc_upper = np.max(germany_lccs_sizes_random, axis=0)

# # japan_lcc_lower = np.min(japan_lccs_sizes_random, axis=0)
# # japan_lcc_upper = np.max(japan_lccs_sizes_random, axis=0)

# # # Plot the shaded region using fill_between
# # axis_percolation_germany.fill_between(germany_removed_fractions_random[0], germany_lcc_lower, germany_lcc_upper, color='gray', alpha=0.3)
# # axis_percolation_japan.fill_between(japan_removed_fractions_random[0], japan_lcc_lower, japan_lcc_upper, color='gray', alpha=0.3)

# # # Optionally, plot the median or mean curve as a line
# # germany_lcc_median = np.median(germany_lccs_sizes_random, axis=0)
# # japan_lcc_median = np.median(japan_lccs_sizes_random, axis=0)

# # axis_percolation_germany.plot(germany_removed_fractions_random[0], germany_lcc_median, color='black', label='Median Germany')
# # axis_percolation_japan.plot(japan_removed_fractions_random[0], japan_lcc_median, color='black', label='Median Japan')

# # axis_percolation_germany.plot(germany_removed_fraction_targeted, germany_lcc_sizes_targeted, color='red', label="Targeted Failures", linewidth=3)
# # axis_percolation_germany.set_xlabel("Fraction of Nodes Removed")
# # axis_percolation_germany.set_ylabel("Percolation Probability")
# # axis_percolation_germany.set_title("Network Robustness Analysis - Germany")
# # axis_percolation_germany.grid()
# # axis_percolation_germany.legend()


# # axis_percolation_japan.plot(japan_removed_fraction_targeted, japan_lcc_sizes_targeted, color='red', label="Targeted Failures", linewidth=3)
# # axis_percolation_japan.set_xlabel("Fraction of Nodes Removed")
# # axis_percolation_japan.set_ylabel("Percolation Probability")
# # axis_percolation_japan.set_title("Network Robustness Analysis - Japan")
# # axis_percolation_japan.grid()
# # axis_percolation_japan.legend()


# # Minimum Spanning Tree

# mst_fig_graph, (mst_axis_germany, mst_axis_japan) = plt.subplots(ncols=2, nrows=1)

# # Create a graph and add edges with weights
# MstGermany = MinimalSpanningTree(germany_network.network, germany_network, 'Germany')
# MstJapan = MinimalSpanningTree(japan_network.network, japan_network, 'Japan')


# print(MstGermany)
# print(MstJapan)


# print(germany_network.radius, japan_network.radius)
# print(germany_network.diameter, japan_network.diameter)
# print(germany_network.center, japan_network.center)

# MstGermany.plot_mst_network(mst_axis_germany, node_colors='blue', colorbar_norm=None)
# MstJapan.plot_mst_network(mst_axis_japan, node_colors='blue', colorbar_norm=None)


# plt.show()