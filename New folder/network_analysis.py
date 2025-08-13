import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import json

# Read the edges CSV file
edges_df = pd.read_csv('edges.csv')

# Create undirected graph
G = nx.from_pandas_edgelist(edges_df, source='source', target='target')

# Calculate required metrics
edge_count = G.number_of_edges()
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)
average_degree = sum(degrees.values()) / len(degrees)
num_nodes = G.number_of_nodes()
max_possible_edges = num_nodes * (num_nodes - 1) / 2
density = edge_count / max_possible_edges

# Calculate shortest path between Alice and Eve
try:
    shortest_path_alice_eve = nx.shortest_path_length(G, 'Alice', 'Eve')
except nx.NetworkXNoPath:
    shortest_path_alice_eve = -1  # No path exists

# Function to convert plot to base64
def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return img_base64

# Create network graph visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=12, font_weight='bold',
        edge_color='gray', width=2)
plt.title('Network Graph', size=14)
plt.tight_layout()
network_graph_b64 = plot_to_base64(plt.gcf())
plt.close()

# Create degree histogram
plt.figure(figsize=(8, 6))
degree_values = list(degrees.values())
degree_counts = {}
for deg in degree_values:
    degree_counts[deg] = degree_counts.get(deg, 0) + 1

degrees_list = list(degree_counts.keys())
counts_list = list(degree_counts.values())

plt.bar(degrees_list, counts_list, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Degree', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Degree Distribution', fontsize=14)
plt.xticks(degrees_list)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
degree_histogram_b64 = plot_to_base64(plt.gcf())
plt.close()

# Create the JSON response
result = {
    "edge_count": edge_count,
    "highest_degree_node": highest_degree_node,
    "average_degree": average_degree,
    "density": density,
    "shortest_path_alice_eve": shortest_path_alice_eve,
    "network_graph": network_graph_b64,
    "degree_histogram": degree_histogram_b64
}

# Output as JSON
print(json.dumps(result, indent=2))
