import matplotlib
# --- THIS MUST BE THE FIRST MATPLOTLIB IMPORT ---
# It tells the script to not look for a pop-up window
matplotlib.use('Agg') 

import torch
import osmnx as ox
import networkx as nx
from torch_geometric.data import Data
import os
import geopandas as gpd
import matplotlib.pyplot

# --- Import the GNN model definition ---
try:
    # Assumes your training script is 'train_model.py'
    from train_model import GNN
except ImportError:
    # Fallback in case it's named '3_train_model.py'
    try:
        from three_train_model import GNN 
    except ImportError:
        # Fallback: define the class right here
        print("Could not import GNN class, defining it locally.")
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        class GNN(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(GNN, self).__init__()
                self.conv1 = SAGEConv(in_channels, 64)
                self.conv2 = SAGEConv(64, 32)
                self.out = torch.nn.Linear(32, out_channels)
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = self.out(x)
                return x

print("--- Phase 5: Find Cleanest Route ---")

# --- SET YOUR START AND END POINTS ---
START_POINT = (12.9746, 77.6562) # Example: Koramangala
END_POINT   = (12.9351, 77.5360)   # Example: Malleshwaram
# -------------------------------------------

try:
    print("Loading Bengaluru graph...")
    G = ox.load_graphml('./data/bengaluru_graph.graphml')
    
    print("Creating node ID map...")
    gdf_nodes, _ = ox.graph_to_gdfs(G)
    node_id_map = {osm_id: i for i, osm_id in enumerate(gdf_nodes.index)}

    print("Loading trained GNN model...")
    model = GNN(in_channels=2, out_channels=1) 
    model.load_state_dict(torch.load('gnn_model.pth'))
    model.eval() 

    graph_data = torch.load('./data/training_graph.pt')
    
except FileNotFoundError:
    print("Error: Model or graph file not found. Run previous steps first.")
    exit()

print(f"Predicting pollution for all {G.number_of_nodes()} nodes...")
with torch.no_grad():
    all_predictions = model(graph_data)
    # --- DEBUG: Print prediction stats ---
    valid_predictions = all_predictions[~torch.isnan(all_predictions)] # Ignore any potential NaNs  
    if len(valid_predictions) > 0:
        print(f"\n--- Predicted AQI Stats ---")
        print(f"Min predicted AQI: {valid_predictions.min().item():.2f}")
        print(f"Max predicted AQI: {valid_predictions.max().item():.2f}")
        print(f"Mean predicted AQI: {valid_predictions.mean().item():.2f}")
        print(f"Std Dev predicted AQI: {valid_predictions.std().item():.2f}")
        print(f"---------------------------\n")
    else:
        print("\n--- No valid predictions found --- \n")
    # --- END DEBUG ---

print(f"Assigning pollution cost to all {G.number_of_edges()} roads...")
for u_osm, v_osm, key, edge_data in G.edges(keys=True, data=True):
    try:
        u_idx = node_id_map[u_osm]
        v_idx = node_id_map[v_osm]
        
        aqi_u = all_predictions[u_idx].item()
        aqi_v = all_predictions[v_idx].item()
        
        avg_aqi = (aqi_u + aqi_v) / 2
        if avg_aqi < 0: avg_aqi = 0
        
        pollution_cost = edge_data['length'] * (1 + avg_aqi)
        nx.set_edge_attributes(G, {(u_osm, v_osm, key): pollution_cost}, 'pollution_cost')
    
    except KeyError:
        nx.set_edge_attributes(G, {(u_osm, v_osm, key): edge_data['length']}, 'pollution_cost')

print("Graph is weighted with pollution data.")

orig_node = ox.nearest_nodes(G, Y=START_POINT[0], X=START_POINT[1])
dest_node = ox.nearest_nodes(G, Y=END_POINT[0], X=END_POINT[1])

try:
    print("Calculating cleanest route...")
    clean_route = nx.shortest_path(G, orig_node, dest_node, weight='pollution_cost')
    print(f"Found clean route with {len(clean_route)} nodes.")
    
    print("Calculating fastest (shortest) route...")
    fastest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    print(f"Fastest (shortest) route has {len(fastest_route)} nodes.")

    # --- SIMPLIFIED PLOTTING ---
    print("Plotting routes...")
    
    # Plot the clean route (green) and get the axis
    fig, ax = ox.plot_graph_route(
        G, clean_route, route_color='g', route_linewidth=2,
        node_size=0, bgcolor='k', 
        show=False, close=False  # Do not show or close yet
    )
    
    # Plot the fastest route (red) on the *same axis*
    ox.plot_graph_route(
        G, fastest_route, route_color='r', route_linewidth=1,
        route_linestyle='--', node_size=0, ax=ax,
        show=False, close=False # Do not show or close yet
    )
    
    # Manually save the figure
    output_filename = 'clean_vs_fast_route.png'
    matplotlib.pyplot.savefig(output_filename, dpi=300, bbox_inches='tight')
    matplotlib.pyplot.close(fig)

    print(f"\nSuccess! Saved plot to '{output_filename}'")
    print("  Green = Cleanest Route (from your GNN)")
    print("  Red (dashed) = Fastest Route")

except nx.NetworkXNoPath:
    print(f"Error: No path found between {START_POINT} and {END_POINT}.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")
