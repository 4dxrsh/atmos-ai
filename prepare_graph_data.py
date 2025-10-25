import pandas as pd
import osmnx as ox
import torch
from torch_geometric.data import Data
import geopandas as gpd
import numpy as np
import os

print("--- Phase 3: Prepare Graph Data (FIXED) ---")

# --- CONFIGURATION ---
TRAINING_MONTH = 10 
# ---------------------

# 1. Load the consolidated training data
try:
    df = pd.read_csv("./data/final_training_data.csv", parse_dates=["timestamp"])
except FileNotFoundError:
    print("Error: 'final_training_data.csv' not found.")
    print("Please make sure it is in the 'data' folder.")
    exit()

df_month = df[df['month'] == TRAINING_MONTH]
if df_month.empty:
    print(f"Error: No data found for month {TRAINING_MONTH}. Check your CSV.")
    exit()

df_train = df_month.groupby(['latitude', 'longitude']).agg(
    station_aqi=('station_aqi', 'mean'),
    baseline_aqi=('baseline_aqi', 'mean')
).reset_index()
print(f"Created training set for month {TRAINING_MONTH} with {len(df_train)} stations.")

# 2. Download Bengaluru road graph
graph_file = "./data/bengaluru_graph.graphml"
if os.path.exists(graph_file):
    print("Loading existing graph from file...")
    G = ox.load_graphml(graph_file)
else:
    print("Downloading Bengaluru road network from OSMnx...")
    G = ox.graph_from_place('Bengaluru, India', network_type='drive')
    ox.save_graphml(G, graph_file)
    print("Graph saved to file.")

gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# --- THIS IS THE KEY MAPPING ---
# Create a mapping from OSM ID (like 11010097226) to tensor index (like 0, 1, 2...)
node_id_map = {osm_id: i for i, osm_id in enumerate(gdf_nodes.index)}
# -------------------------------

# 3. Create features for ALL nodes
node_features = []
for node_id, data in G.nodes(data=True):
    node_features.append([data['y'], data['x']]) # [lat, lon]
X = torch.tensor(node_features, dtype=torch.float)

# 4. Get graph edge structure --- THIS IS THE FIXED SECTION ---
print("Mapping graph edges...")
u_osm, v_osm, _ = zip(*gdf_edges.index)

# Convert OSM IDs to 0-based tensor indices
u_idx = [node_id_map[osm_id] for osm_id in u_osm]
v_idx = [node_id_map[osm_id] for osm_id in v_osm]
    
edge_index_np = np.array([u_idx, v_idx])
edge_index = torch.tensor(edge_index_np, dtype=torch.long)
# --- END OF FIX ---

# 5. Map station data to graph nodes
print("Mapping station data to graph...")
gdf_sensors = gpd.GeoDataFrame(
    df_train, 
    geometry=gpd.points_from_xy(df_train.longitude, df_train.latitude),
    crs="EPSG:4326"
)
gdf_sensors = gdf_sensors.to_crs(gdf_nodes.crs)
sensor_nodes_osm = ox.nearest_nodes(G, gdf_sensors.geometry.x, gdf_sensors.geometry.y)

# 6. Create labels (y) and training_mask
y = torch.full((G.number_of_nodes(), 1), float('nan'))
train_mask = torch.zeros(G.number_of_nodes(), dtype=torch.bool)

# Use the map to convert OSM ID to tensor index
for i, osm_id in enumerate(sensor_nodes_osm):
    tensor_index = node_id_map[osm_id]
    y[tensor_index] = df_train.iloc[i]['station_aqi']
    train_mask[tensor_index] = True

# 7. Create and save the PyG Data object
graph_data = Data(x=X, edge_index=edge_index.contiguous(), y=y, train_mask=train_mask)
print("\nPyG Data Object Created:")
print(graph_data)
print(f"Number of nodes: {graph_data.num_nodes}")
print(f"Number of edges: {graph_data.num_edges}")
print(f"Number of nodes to train on: {graph_data.train_mask.sum()}")

torch.save(graph_data, './data/training_graph.pt')
print("Success! Saved 'training_graph.pt'. Ready for model training.")
