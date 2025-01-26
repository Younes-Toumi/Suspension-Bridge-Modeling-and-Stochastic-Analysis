import matplotlib.cm as cm
import matplotlib.colors as mcolors


import math

def haversine(coord1, coord2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return round(2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


# Load the network data (cities and their frequencies) from the first file
def load_network_data(internet_data_path, city_coordinates, country):
    network = []

    with open(internet_data_path, 'r', encoding='utf-8') as file:
        city_connections = []

        for line in file:
            if line[0] == '*':
                # * Tokyo - Yokohama - Nagoya - Kyoto - Osaka\n
                cities = line[2:-1].split(' - ')
        
                # [Tokyo, Yokohama, Nagoya, Kyoto, Osaka]
                city_pairs = [(cities[i], cities[i + 1]) for i in range(len(cities) - 1)]
                # [('Tokyo', 'Yokohama'), ('Yokohama', 'Nagoya'), ('Nagoya', 'Kyoto'), ('Kyoto', 'Osaka')]

                city_connections += city_pairs



    # Counting occurrences of each connection
    connection_counts = {}
    for connection in city_connections:
        connection_counts[connection] = connection_counts.get(connection, 0) + 1

    network = []
    for key, value in connection_counts.items():
        edge = {
            'edge': key,
            'frequency': value
        }
        
        network.append(edge)

    # Evaluate distances for each edge
    for edge_data in network:
        city1, city2 = edge_data['edge']
        coord1 = city_coordinates[city1]
        coord2 = city_coordinates[city2]
        distance = haversine(coord1, coord2)
        edge_data['distance'] = distance  # Add distance to the edge data

        if country == 'Germany':
            train_speed = 250 # [km/h]
            base_travel_time = distance/train_speed # [h]

            waiting_time = base_travel_time/edge_data['frequency'] # might adjust this part

        if country == 'Japan':
            train_speed = 300 # [km/h]
            base_travel_time = distance/train_speed # [h]
            waiting_time = 0 # might adjust this part        

        real_travel_time = base_travel_time + waiting_time # [h]

        edge_data['travel_time'] = real_travel_time  # Add distance to the edge data
    
    return network

# Step 2: Load city coordinates (latitude and longitude) from the second file
def load_city_coordinates(filename):
    coordinates = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Initial line:
            # Delmenhorst: (53.048095, 8.6286066)
            splitted_line = line.split(": ") # ['Delmenhorst', '(53.048095, 8.6286066)\n']
            
            city = splitted_line[0]
            coords_str = splitted_line[1][:-1][1:-1] # '53.048095, 8.6286066'
            (latitude, longitude) = map(float, coords_str.split(', ')) #tuple (53.048095, 8.6286066)

            coordinates[city] = (latitude, longitude)

    return coordinates

def get_centrality_colors(network, network_centrality):
    network_centrality_values = list(network_centrality.values())
    network_norm = mcolors.Normalize(vmin=min(network_centrality_values), vmax=max(network_centrality_values)) # Normalize centrality values for colormap
    network_node_colors = [cm.plasma(network_norm(network_centrality[node])) for node in network.nodes()] # Use the dictionary directly to get the centrality value for each node

    return network_node_colors, network_norm