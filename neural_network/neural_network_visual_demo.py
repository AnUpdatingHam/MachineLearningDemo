import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import networkx as nx

# Data
x1 = np.array([-3, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
x2 = np.array([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, -2.2204, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
y = np.array([0.6589, 0.2206, -0.1635, -0.4712, -0.6858, -0.7975, -0.8040,
              -0.7113, -0.5326, -0.2875, 0, 0.3035, 0.5966, 0.8553, 1.0600, 1.1975, 1.2618])

# Prepare input data
inputData = np.vstack((x1, x2)).T  # shape (17, 2)

# Initialize the neural network
np.random.seed(88888)  # Set random seed for reproducibility
net = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', solver='lbfgs', max_iter=15000, tol=0.0001, random_state=88888)

# Train the network
net.fit(inputData, y)

# Extract weights and biases
weights = net.coefs_
biases = net.intercepts_

# Create a directed graph
G = nx.DiGraph()

# Add nodes for input, hidden, and output layers
input_nodes = ['x1', 'x2']
hidden_nodes = ['h1', 'h2', 'h3']
output_node = ['y']

# Adding nodes to graph
G.add_nodes_from(input_nodes, layer='input')
G.add_nodes_from(hidden_nodes, layer='hidden')
G.add_nodes_from(output_node, layer='output')

# Connect input layer to hidden layer
for i, input_node in enumerate(input_nodes):
    for h, hidden_node in enumerate(hidden_nodes):
        weight = weights[0][i][h]
        G.add_edge(input_node, hidden_node, weight=f'{weight:.2f}')

# Connect hidden layer to output layer
for h, hidden_node in enumerate(hidden_nodes):
    weight = weights[1][h][0]
    G.add_edge(hidden_node, output_node[0], weight=f'{weight:.2f}')

# Position the nodes in layers
pos = {'x1': (0, 1), 'x2': (0, -1),
       'h1': (1, 2), 'h2': (1, 0), 'h3': (1, -2),
       'y': (2, 0)}

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', arrowsize=20, font_size=10, font_weight='bold')

# Draw edge labels (weights)
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Add bias labels to hidden and output nodes
for i, hidden_node in enumerate(hidden_nodes):
    bias_label = f'bias: {biases[0][i]:.2f}'
    x, y = pos[hidden_node]
    plt.text(x, y-0.4, bias_label, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))

output_bias_label = f'bias: {biases[1][0]:.2f}'
x, y = pos[output_node[0]]
plt.text(x, y-0.4, output_bias_label, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))

plt.title('Neural Network Structure with Weights and Biases')
plt.show()
