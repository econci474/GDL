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
    
    def __init__(self, num_features, hidden_dim, num_classes, K, dropout=None, normalize=True):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            K: Number of GCN layers (K=0 means no GCN layers, just input features)
            dropout: Dropout probability (None = no dropout)
            normalize: Whether to add self-loops and apply symmetric normalization
        """
        super(GCNNet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        self.normalize = normalize
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Build K GCN layers with normalization
        self.convs = nn.ModuleList()
        
        if K == 0:
            # K=0: No GCN layers, just use input features directly
            pass
        elif K == 1:
            # K=1: Single GCN layer
            self.convs.append(GCNConv(num_features, hidden_dim, normalize=normalize))
        else:
            # K>=2: Multiple GCN layers
            self.convs.append(GCNConv(num_features, hidden_dim, normalize=normalize))
            for _ in range(K - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, normalize=normalize))
        
        # Final classifier (operates on input features if K=0, else hidden_dim)
        classifier_input_dim = num_features if K == 0 else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        # MULTIPLE classifiers (one per layer): one for input + one per conv layer
        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))  # For k=0 (input features)
        for _ in range(len(self.convs)):
            self.layer_classifiers.append(nn.Linear(hidden_dim, num_classes))  # For each conv layer
        
    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        # If K=0, no convolutions - use input features directly
        if self.K == 0:
            return self.classifier(x)
        
        # K >= 1: Apply conv layers
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
            embeddings: Dict mapping k -> [N, D] tensor, for k=0..K
                        embeddings[0] = input features
                        embeddings[k] = output of layer k (before classifier, k=1..K)
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: input features
        
        # If K=0, only return input features
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        # K >= 1: Apply conv layers and collect embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits
    
    def forward_with_classifier_head(self, data):
        """
        Returns per-layer logits and per-layer softmax probabilities.

        Returns:
            layer_logits: List of [N, C] tensors for k=0..K
            layer_probs:  List of [N, C] tensors for k=0..K (softmax over classes)
        """
        embeddings, _ = self.forward_with_embeddings(data)

        layer_logits = []
        layer_probs = []
        for k, emb in enumerate(embeddings):
            logits_k = self.layer_classifiers[k](emb)
            probs_k = F.softmax(logits_k, dim=-1)
            layer_logits.append(logits_k)
            layer_probs.append(probs_k)

        return layer_logits, layer_probs


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
            K: Number of GAT layers (K=0 means no GAT layers, just input features)
            heads: Number of attention heads
            dropout: Dropout probability (None = no dropout)
        """
        super(GATNet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        self.heads = heads
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Build K GAT layers
        self.convs = nn.ModuleList()
        
        if K == 0:
            # K=0: No GAT layers
            pass
        elif K == 1:
            # K=1: Single GAT layer
            self.convs.append(GATConv(num_features, hidden_dim, heads=heads, concat=True))
        else:
            # K>=2: Multiple GAT layers
            self.convs.append(GATConv(num_features, hidden_dim, heads=heads, concat=True))
            for _ in range(K - 1):
                # Input: heads * hidden_dim from previous layer
                self.convs.append(GATConv(heads * hidden_dim, hidden_dim, heads=heads, concat=True))
        
        # Final classifier (operates on input features if K=0, else heads * hidden_dim)
        classifier_input_dim = num_features if K == 0 else (heads * hidden_dim)
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        # MULTIPLE classifiers (one per layer): one for input + one per conv layer
        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))  # For k=0 (input features)
        for _ in range(len(self.convs)):
            self.layer_classifiers.append(nn.Linear(heads * hidden_dim, num_classes))  # For each conv layer
        
    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        # If K=0, no convolutions - use input features directly
        if self.K == 0:
            return self.classifier(x)
        
        # K >= 1: Apply GAT layers
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
        
        # If K=0, only return input features
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        # K >= 1: Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.elu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits

    def forward_with_classifier_head(self, data):
        """
        Returns per-layer logits and per-layer softmax probabilities.

        Returns:
            layer_logits: List of [N, C] tensors for k=0..K
            layer_probs:  List of [N, C] tensors for k=0..K (softmax over classes)
        """
        embeddings, _ = self.forward_with_embeddings(data)

        layer_logits = []
        layer_probs = []
        for k, emb in enumerate(embeddings):
            logits_k = self.layer_classifiers[k](emb)
            probs_k = F.softmax(logits_k, dim=-1)
            layer_logits.append(logits_k)
            layer_probs.append(probs_k)

        return layer_logits, layer_probs


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
            K: Number of GraphSAGE layers (K=0 means no GraphSAGE layers, just input features)
            dropout: Dropout probability (None = no dropout)
            aggr: Aggregation method ('mean', 'max', or 'lstm')
        """
        super(GraphSAGENet, self).__init__()
        
        self.K = K
        self.dropout = dropout
        self.aggr = aggr
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Build K GraphSAGE layers
        self.convs = nn.ModuleList()
        
        if K == 0:
            # K=0: No GraphSAGE layers
            pass
        elif K == 1:
            # K=1: Single GraphSAGE layer
            self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
        else:
            # K>=2: Multiple GraphSAGE layers
            self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
            for _ in range(K - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Final classifier (operates on input features if K=0, else hidden_dim)
        classifier_input_dim = num_features if K == 0 else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        # MULTIPLE classifiers (one per layer): one for input + one per conv layer
        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))  # For k=0 (input features)
        for _ in range(len(self.convs)):
            self.layer_classifiers.append(nn.Linear(hidden_dim, num_classes))  # For each conv layer

    def forward(self, data):
        """
        Standard forward pass returning final logits.
        
        Args:
            data: torch_geometric.data.Data object
            
        Returns:
            logits: [N, num_classes] final class logits
        """
        x, edge_index = data.x, data.edge_index
        
        # If K=0, no convolutions - use input features directly
        if self.K == 0:
            return self.classifier(x)
        
        # K >= 1: Apply GraphSAGE layers
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
        
        # If K=0, only return input features
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        # K >= 1: Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings.append(x.clone())  # k=1..K
        
        logits = self.classifier(x)
        return embeddings, logits
    
    def forward_with_classifier_head(self, data):
        """
        Returns per-layer logits and per-layer softmax probabilities.

        Returns:
            layer_logits: List of [N, C] tensors for k=0..K
            layer_probs:  List of [N, C] tensors for k=0..K (softmax over classes)
        """
        embeddings, _ = self.forward_with_embeddings(data)

        layer_logits = []
        layer_probs = []
        for k, emb in enumerate(embeddings):
            logits_k = self.layer_classifiers[k](emb)
            probs_k = F.softmax(logits_k, dim=-1)
            layer_logits.append(logits_k)
            layer_probs.append(probs_k)

        return layer_logits, layer_probs



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
                   dropout=None,
                   normalize=True)
    
    # Test standard forward
    logits = model(data)
    print(f"✓ Forward pass: logits shape = {logits.shape}")
    
    # Test embedding extraction
    embeddings, logits = model.forward_with_embeddings(data)
    print(f"✓ Embedding extraction: {len(embeddings)} embeddings")
    for k, emb in enumerate(embeddings):
        print(f"  k={k}: shape {emb.shape}")
