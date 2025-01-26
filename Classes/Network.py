import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Dict, Tuple

class Network:
    """
    A class to represent a fast train network using NetworkX.

    Attributes:
        name (str): Name of the network (e.g., country name).
        network_data (List[Dict]): List of dictionaries containing edge data.
        network (nx.Graph): NetworkX graph object representing the network.
        properties (Dict): Stores computed properties of the network.
    """

    def __init__(self, name: str, network_data: List[Dict]):
        self.name = name
        self.network_data = network_data
        self.network = nx.Graph()
        self.properties = {}

        self._build_network_graph()
        self._compute_network_properties(weight='travel_time')

    def __repr__(self):
        local_props = self.local_properties
        global_props = self.global_properties

        global_symmetry_measures = (
            f"Anti-Reflexive: {global_props['is_anti_reflexive']}, "
            f"Symmetric: {global_props['is_symmetric']}, "
            f"Anti-Symmetric: {global_props['is_anti_symmetric']}, "
            f"Asymmetric: {global_props['is_asymmetric']}"
        )
  
        global_props_summary = (

            f"2.1. Symmetry measures:\n"
            f"Anti-Reflexive: {global_props['is_anti_reflexive']}, "
            f"Symmetric: {global_props['is_symmetric']}, "
            f"Anti-Symmetric: {global_props['is_anti_symmetric']}, "
            f"Asymmetric: {global_props['is_asymmetric']}\n\n"
            
            f"2.2. Efficiency measures:\n"
            f"Efficiency: Global: {global_props['unweighted_efficiency']:.4f}, "
            f"Weighted Global: {global_props['weighted_efficiency']:.4f}\n\n"

            f"2.3. Eccentricity measures:\n"
            f"Radius: {global_props['radius']:.3f}, Diameter: {global_props['diameter']:.3f}, Center: {global_props['center']}\n\n"

            f"2.4. Averaged measures:\n"
            f"Travel time: {global_props['average_travel_time']:.3f}, Distance: {global_props['average_distance']:.3f}, Frequency: {global_props['average_frequency']:.3f}\n"
            f"Degree: {global_props['average_degree']:.3f}\n\n"
            
            f"2.5. Average Centrality measures:\n"
            f"Closeness: {global_props['average_closeness_centrality']:.3f}, Betweenness: {global_props['average_betweenness_centrality']:.3f}"

        )

        return (f"Network: {self.name}\n"
                f"1. Nodes: {local_props['num_nodes']}, Edges: {local_props['num_edges']}\n"
                f"2. Global Properties:\n"
                f"   ------------------\n"
                f"{global_props_summary}"
                )

    def _build_network_graph(self):
        self.edge_travel_time = []
        self.edge_distance = []
        self.edge_frequency = []

        for edge_data in self.network_data:
            city1, city2 = edge_data['edge']
            self.network.add_edge(
                city1, city2,
                frequency=edge_data['frequency'],
                distance=edge_data['distance'],
                travel_time=edge_data['travel_time']
            )

            self.edge_travel_time.append(edge_data['travel_time'])
            self.edge_distance.append(edge_data['distance'])
            self.edge_frequency.append(edge_data['frequency'])

    def _compute_network_properties(self, weight: str = 'travel_time'):

        self.adjacency_matrix = nx.to_numpy_array(self.network, weight=weight)    


        self.local_properties = {

            # Basic network properties
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'degrees': list(dict(self.network.degree()).values()),

            # Centrality measures
            'closeness_centrality': nx.closeness_centrality(self.network, distance=weight),
            'betweenness_centrality': nx.betweenness_centrality(self.network, weight=weight),
        }



        self.global_properties = {
            # Efficiency measures
            'unweighted_efficiency': nx.global_efficiency(self.network),
            'weighted_efficiency': self._compute_weighted_global_efficiency('travel_time'),
            
            # Symmetry measures
            'is_anti_reflexive': self._is_anti_reflexive(),
            'is_symmetric': self._is_symmetric(),
            'is_anti_symmetric': self._is_anti_symmetric(),
            'is_asymmetric': self._is_asymmetric(),

            # Eccentricity measures
            'radius': nx.radius(self.network, weight=weight),
            'diameter': nx.diameter(self.network, weight=weight),
            'center': nx.center(self.network, weight=weight),

            # Averaged measures
            'average_edge_weight': np.mean(self.adjacency_matrix[self.adjacency_matrix > 0]),
            'average_degree': np.mean(self.local_properties['degrees']),
            'average_closeness_centrality': np.mean(list(self.local_properties['closeness_centrality'].values())),
            'average_betweenness_centrality': np.mean(list(self.local_properties['betweenness_centrality'].values())),

            # Averaged edge properties
            'average_travel_time': np.mean(self.edge_travel_time),
            'average_distance': np.mean(self.edge_distance),
            'average_frequency': np.mean(self.edge_frequency)
        }
        

    def _is_anti_reflexive(self) -> bool:
        return np.all(np.diag(self.adjacency_matrix) == 0)

    def _is_symmetric(self) -> bool:
        return np.allclose(self.adjacency_matrix, self.adjacency_matrix.T)

    def _is_anti_symmetric(self) -> bool:
        return np.all((self.adjacency_matrix * self.adjacency_matrix.T) <= np.eye(self.adjacency_matrix.shape[0]))

    def _is_asymmetric(self) -> bool:
        return np.all((self.adjacency_matrix * self.adjacency_matrix.T) == 0)



    def _compute_weighted_global_efficiency(self, weight: str) -> float:
        total_efficiency, total_pairs = 0, 0
        for u in self.network.nodes:
            lengths = nx.single_source_dijkstra_path_length(self.network, u, weight=weight)
            for v in self.network.nodes:
                if u != v and v in lengths:
                    total_efficiency += 1 / lengths[v]
                    total_pairs += 1
        return total_efficiency / total_pairs if total_pairs > 0 else 0


    @staticmethod
    def plot_network(network, 
                    axis,                    
                    city_coords: Dict[str, Tuple[float, float]],
                    node_colors,  
                    edge_widths):
        """
        Plots the network on a geographical map.

        """

        pos = {city: (lon, lat) for city, (lat, lon) in city_coords.items()}

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(network, pos, ax=axis, node_size=60, node_color=node_colors, alpha=1.0)
        nx.draw_networkx_edges(network, pos, ax=axis, width=edge_widths, edge_color='gray', alpha=1.0)

        # Set axis limits
        axis.set_title(f"Network Visualization")

        # # Add a color bar
        try:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axis)
            cbar.set_label(rotation=270, labelpad=15)

        except:
            pass

    @staticmethod
    def measure_robustness(network, strategy: str = "random") -> Tuple[List[float], List[float]]:
        """
        Simulates node removal and measures the impact on the largest connected component.

        Args:
            strategy (str): Removal strategy ("random" or "targeted").

        Returns:
            Tuple[List[float], List[float]]: Fraction of nodes removed and corresponding LCC sizes.
        """
        graph = network.copy()
        total_nodes = len(graph.nodes)
        removed_fraction, lcc_sizes = [], []

        while graph.nodes:
            # Measure largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            lcc_sizes.append(len(largest_cc) / total_nodes)
            removed_fraction.append((total_nodes - len(graph.nodes)) / total_nodes)

            # Remove a node
            if strategy == "random":
                node_to_remove = random.choice(list(graph.nodes))

            elif strategy == "targeted":
                node_to_remove = max(graph.degree, key=lambda x: x[1])[0]

            else:
                raise ValueError("Invalid strategy. Use 'random' or 'targeted'.")

            graph.remove_node(node_to_remove)

        return removed_fraction, lcc_sizes
