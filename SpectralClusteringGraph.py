import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict
import math

# Define a class to represent a graph and perform spectral clustering
class SpectralClusteringGraph:
    def __init__(self, central_node, sigma=1.0):
        self.central_node = central_node
        self.sigma = sigma
        self.edges = []
        self.features = {}
        self.node_mapping = {}
        self.adjacency_matrix = None
        self.degree_matrix = None
        self.laplacian_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.labels = None
        self.graph = None

    # Function to read edges from a file
    def read_edges(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                nodes = line.strip().split()
                self.edges.append((int(nodes[0]), int(nodes[1])))

    # Function to read features from a file
    def read_features(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                node = int(parts[0])
                feature_vector = np.array(list(map(int, parts[1:])))
                self.features[node] = feature_vector

    # Gaussian kernel similarity
    @staticmethod
    def gaussian_similarity(x, y, sigma):
        diff = x - y
        distance = np.dot(diff, diff)
        return np.exp(-sigma * distance)

    # Create the weighted adjacency matrix
    def create_weighted_adjacency_matrix(self):
        size = len(self.node_mapping)
        self.adjacency_matrix = np.zeros((size, size), dtype=float)
        for u, v in self.edges:
            u_idx, v_idx = self.node_mapping[u], self.node_mapping[v]
            weight = self.gaussian_similarity(self.features[u], self.features[v], self.sigma)
            self.adjacency_matrix[u_idx, v_idx] = weight
            self.adjacency_matrix[v_idx, u_idx] = weight

    # Create the degree matrix
    def create_degree_matrix(self):
        degrees = np.sum(self.adjacency_matrix, axis=1)
        self.degree_matrix = np.diag(degrees)

    # Create the Laplacian matrix
    def create_laplacian_matrix(self):
        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix

    # Analyze the Laplacian matrix to compute eigenvalues and eigenvectors
    def analyze_laplacian(self):
        eigenvalues, eigenvectors = eigh(self.laplacian_matrix)
        sorted_indices = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[sorted_indices][:100]
        self.eigenvectors = eigenvectors[:, sorted_indices]

    # Perform spectral clustering
    def perform_clustering(self):
        eigen_gaps = np.diff(self.eigenvalues)
        largest_gap_index = np.argmax(eigen_gaps)
        k = largest_gap_index + 1  # Number of clusters
        print(f"Largest Eigen Gap for Central Node {self.central_node}: is at index {largest_gap_index}")

        #Viuslizeing eigen vlaues
        plt.figure(figsize=(8, 6))
        plt.plot(self.eigenvalues, 'o-', label="Eigenvalues", markersize=8)
        for i in range(len(eigen_gaps)):
            x_pos = i + 0.5  # Position between two eigenvalues
            y_pos = (self.eigenvalues[i] + self.eigenvalues[i + 1]) / 2  # Midpoint for the text
            if i == largest_gap_index:
                plt.annotate(f"{eigen_gaps[i]:.2f}", xy=(x_pos, y_pos), xytext=(x_pos, y_pos+0.1), ha="center", color="red")
        plt.title(f"Eigenvalues and Gaps for central node {self.central_node}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()

        # Use k eigenvectors for clustering
        selected_eigenvectors = self.eigenvectors[:, :k]
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(selected_eigenvectors)
        self.labels = kmeans.labels_

    # Build the graph and visualize the clustering result
    def build_and_visualize_graph(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from(self.edges)
        isolated_nodes = set(self.features.keys()) - set(self.graph.nodes)
        self.graph.add_nodes_from(isolated_nodes)

        node_colors = [self.labels[self.node_mapping[node]] for node in self.graph.nodes]

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph,
            pos,
            with_labels=False,
            node_color=node_colors,
            cmap=plt.cm.get_cmap("tab10", max(self.labels) + 1),
            node_size=100,
        )
        plt.title(f"Graph Visualization with Spectral Clustering for central Node {self.central_node}")
        plt.show()

    # Evaluate the clustering using ground truth data
    def evaluate_clustering(self, ground_truth_file):
        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file:
                parts = line.strip().split("\t")
                circle_name = parts[0][6:]
                circle_members = list(map(int, parts[1:]))
                ground_truth[circle_name] = circle_members

        node_mapping = {node: idx for idx, node in enumerate(self.graph.nodes())}
        num_nodes = len(node_mapping)

        ground_truth_labels = -1 * np.ones(num_nodes, dtype=int)
        for circle_id, members in enumerate(ground_truth.values()):
            for member in members:
                if member in node_mapping:
                    ground_truth_labels[node_mapping[member]] = circle_id

        valid_indices = ground_truth_labels != -1
        nmi = normalized_mutual_info_score(ground_truth_labels[valid_indices], self.labels[valid_indices])
        ari = adjusted_rand_score(ground_truth_labels[valid_indices], self.labels[valid_indices])

        modularity = nx.algorithms.community.modularity(
            self.graph, nx.algorithms.community.label_propagation_communities(self.graph)
        )

        if len(set(self.labels)) > 1:
            silhouette = silhouette_score(nx.to_numpy_array(self.graph), self.labels)
        else:
            silhouette = float("nan")

        return {
            "silhouette_score": abs(silhouette),
            "nmi": abs(nmi),
            "ari": abs(ari),
            "modularity": abs(modularity),
        }