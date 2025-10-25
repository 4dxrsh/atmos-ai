import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import os

print("--- Phase 4: Train GNN Model ---")

# 1. Define your GNN Model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # GraphSAGE is good for learning from node features (lat/lon)
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 32)
        self.out = torch.nn.Linear(32, out_channels) # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.out(x)
        return x

# 2. Define the main training function
def main():
    # Load the prepared graph data
    try:
        graph_data = torch.load('./data/training_graph.pt')
    except FileNotFoundError:
        print("Error: 'training_graph.pt' not found. Run '2_prepare_graph_data.py' first.")
        exit()

    # Initialize model and optimizer
    # in_channels=2 because our features are [lat, lon]
    model = GNN(in_channels=graph_data.num_node_features, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training Loop
    print("Training GNN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        
        out = model(graph_data)
        
        # Calculate loss ONLY on the nodes that have real station data
        loss = F.mse_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 199:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), 'gnn_model.pth')
    print("Success! Saved trained model to 'gnn_model.pth'.")
    print("You are ready for the final step.")

# 3. --- THIS IS THE FIX ---
# This line tells Python to ONLY run main()
# when you execute this file directly.
if __name__ == "__main__":
    main()
