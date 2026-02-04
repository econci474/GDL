"""GNN models with layer-wise embedding extraction capability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCNNet(nn.Module):
    """
    Graph Convolutional Network with K layers.
    
    Supports both standard forward pass and embedding extraction.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, K, dropout=None):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            K: Number of GCN layers
            dropout: Dropout probability (None = no dropout)
        """
        super(GCNNet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        
        # Build K GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(K - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:  # ReLU for all except last layer
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def forward_with_embeddings(self, data):
        """
        Forward pass that also returns intermediate embeddings.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            embeddings: List of [N, D] tensors, one per depth k=0..K
                        embeddings[0] = input features
                        embeddings[k] = output of layer k (before classifier)
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: input features
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits


class GATNet(nn.Module):
    """
    Graph Attention Network with K layers.
    
    Supports both standard forward pass and embedding extraction.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, K, heads=8, dropout=None):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension  
            num_classes: Number of output classes
            K: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability (None = no dropout)
        """
        super(GATNet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        self.heads = heads
        
        # Build K GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, concat=True))
        
        for _ in range(K - 1):
            # Input: heads * hidden_dim from previous layer
            self.convs.append(GATConv(heads * hidden_dim, hidden_dim, heads=heads, concat=True))
        
        # Final classifier (input: heads * hidden_dim)
        self.classifier = nn.Linear(heads * hidden_dim, num_classes)
        
    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.elu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def forward_with_embeddings(self, data):
        """
        Forward pass that also returns intermediate embeddings.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            embeddings: List of [N, D] tensors, one per depth k=0..K
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: input features
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.elu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits


class GraphSAGENet(nn.Module):
    """
    GraphSAGE (SAmple and aggreGatE) Network with K layers.
    
    Supports both standard forward pass and embedding extraction.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, K, dropout=None, aggr='mean'):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            K: Number of GraphSAGE layers
            dropout: Dropout probability (None = no dropout)
            aggr: Aggregation method ('mean', 'max', or 'lstm')
        """
        super(GraphSAGENet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        self.aggr = aggr
        
        # Build K GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
        for _ in range(K - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:  # ReLU for all except last layer
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def forward_with_embeddings(self, data):
        """
        Forward pass that also returns intermediate embeddings.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            embeddings: List of [N, D] tensors, one per depth k=0..K
                        embeddings[0] = input features
                        embeddings[k] = output of layer k (before classifier)
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: input features
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits


if __name__ == '__main__':
    # Test model instantiation
    import sys
    sys.path.append('..')
    from datasets import load_dataset
    
    print("Testing GCN model...")
    data = load_dataset('Cora')
    
    model = GCNNet(num_features=data.num_features, 
                   hidden_dim=64, 
                   num_classes=7,  # Cora has 7 classes
                   K=8,
                   dropout=None)
    
    # Test standard forward
    logits = model(data)
    print(f"✓ Forward pass: logits shape = {logits.shape}")
    
    # Test embedding extraction
    embeddings, logits = model.forward_with_embeddings(data)
    print(f"✓ Embedding extraction: {len(embeddings)} embeddings")
    for k, emb in enumerate(embeddings):
        print(f"  k={k}: shape {emb.shape}")
