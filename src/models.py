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
    
    def __init__(self, num_features, hidden_dim, num_classes, K,
                 dropout=None, normalize=True,
                 dropout_input=None, dropout_middle=None):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            K: Number of GCN layers
            dropout: Legacy — sets both dropout_input and dropout_middle if provided
            normalize: Whether to add self-loops and apply symmetric normalization
            dropout_input: Dropout on raw input features before first conv
            dropout_middle: Dropout between intermediate conv layers (None = disabled)
        """
        super(GCNNet, self).__init__()
        
        self.K = K
        self.normalize = normalize
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        # Legacy compat: if only dropout provided, treat as dropout_input
        self.dropout_input  = dropout_input if dropout_input  is not None else dropout
        self.dropout_middle = dropout_middle if dropout_middle is not None else None
        
        # Build K GCN layers with normalization
        self.convs = nn.ModuleList()
        
        if K == 0:
            pass
        elif K == 1:
            self.convs.append(GCNConv(num_features, hidden_dim, normalize=normalize))
        else:
            self.convs.append(GCNConv(num_features, hidden_dim, normalize=normalize))
            for _ in range(K - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, normalize=normalize))
        
        classifier_input_dim = num_features if K == 0 else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))
        for _ in range(len(self.convs)):
            self.layer_classifiers.append(nn.Linear(hidden_dim, num_classes))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.K == 0:
            return self.classifier(x)
        
        # Input feature dropout
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def forward_with_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: raw input features (before dropout)
        
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        # Input feature dropout
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
            embeddings.append(x.clone())  # k=1..K (clean, no dropout)
        
        logits = self.classifier(x)
        return embeddings, logits
    
    def forward_with_classifier_head(self, data):
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
    
    def __init__(self, num_features, hidden_dim, num_classes, K, heads=8,
                 dropout=None, dropout_input=None, dropout_middle=None,
                 attention_dropout=0.6):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension  
            num_classes: Number of output classes
            K: Number of GAT layers
            heads: Number of attention heads
            dropout: Legacy — sets dropout_input if provided
            dropout_input: Dropout on raw input features before first conv
            dropout_middle: Dropout between intermediate conv layers (None = disabled)
            attention_dropout: Dropout on normalised attention coefficients
                               (default 0.6, per Veličković et al. ICLR 2018)
        """
        super(GATNet, self).__init__()
        
        self.K = K
        self.heads = heads
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_input     = dropout_input if dropout_input  is not None else dropout
        self.dropout_middle    = dropout_middle if dropout_middle is not None else None
        self.attention_dropout = attention_dropout
        
        self.convs = nn.ModuleList()
        
        if K == 0:
            # No GAT layers — fall back to linear classifier on raw features
            pass
        elif K == 1:
            # Single layer: maps directly to num_classes, averaging across heads
            self.convs.append(GATConv(num_features, num_classes,
                                      heads=heads, concat=False,
                                      dropout=attention_dropout))
        else:
            # First layer: num_features → heads*hidden_dim (concat=True)
            self.convs.append(GATConv(num_features, hidden_dim,
                                      heads=heads, concat=True,
                                      dropout=attention_dropout))
            # Intermediate layers: heads*hidden_dim → heads*hidden_dim
            for _ in range(K - 2):
                self.convs.append(GATConv(heads * hidden_dim, hidden_dim,
                                          heads=heads, concat=True,
                                          dropout=attention_dropout))
            # Final layer: concat=False → averages heads → [N, num_classes]
            self.convs.append(GATConv(heads * hidden_dim, num_classes,
                                      heads=heads, concat=False,
                                      dropout=attention_dropout))

        # Linear classifier only used for K==0 (no conv layers)
        self.classifier = nn.Linear(num_features, num_classes) if K == 0 else None

        # Per-layer linear classifiers for the classifier-head training path.
        # k=0: project raw features; k=1..K-1: project intermediate heads*hidden_dim embeddings.
        # k=K: final GATConv already outputs num_classes — no Linear needed.
        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))  # k=0
        num_intermediate = len(self.convs) - 1  # K-1 layers with concat=True
        for _ in range(num_intermediate):
            self.layer_classifiers.append(nn.Linear(heads * hidden_dim, num_classes))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.K == 0:
            return self.classifier(x)
        
        # Node feature dropout on raw input
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                # Intermediate layers: ELU + optional node feature dropout
                x = F.elu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
            # Final layer: no activation — outputs raw logits [N, num_classes]
        
        return x  # [N, num_classes] from GATConv(concat=False)
    
    def forward_with_embeddings(self, data):
        """
        Returns:
            embeddings: list of length K, where
                embeddings[0] = raw input [N, num_features]
                embeddings[k] = after intermediate layer k [N, heads*hidden_dim], k=1..K-1
                (the final layer output is returned as logits, not stored as embedding)
            logits: [N, num_classes] from the final GATConv(concat=False)
        """
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: raw input features (before dropout)
        
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.elu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
                embeddings.append(x.clone())  # intermediate: [N, heads*hidden_dim]
            # Final layer: no activation, not stored as embedding
        
        logits = x  # [N, num_classes]
        return embeddings, logits

    def forward_with_classifier_head(self, data):
        """
        Returns per-layer logits and probs for k=0..K:
          k=0..K-1: nn.Linear applied to intermediate embeddings
          k=K:      final GATConv output (already [N, num_classes])
        """
        embeddings, final_logits = self.forward_with_embeddings(data)

        layer_logits = []
        layer_probs = []
        # k=0..K-1: apply per-layer linear classifiers
        for k, emb in enumerate(embeddings):
            logits_k = self.layer_classifiers[k](emb)
            probs_k = F.softmax(logits_k, dim=-1)
            layer_logits.append(logits_k)
            layer_probs.append(probs_k)

        # k=K: final GATConv output — already num_classes, no Linear needed
        layer_logits.append(final_logits)
        layer_probs.append(F.softmax(final_logits, dim=-1))

        return layer_logits, layer_probs


class GraphSAGENet(nn.Module):
    """
    GraphSAGE (SAmple and aggreGatE) Network with K layers.
    
    Supports both standard forward pass and embedding extraction.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, K,
                 dropout=None, aggr='sum',
                 dropout_input=None, dropout_middle=None):
        """
        Args:
            num_features: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            K: Number of GraphSAGE layers
            dropout: Legacy, sets dropout_input if provided
            aggr: Aggregation method ('mean', 'max', or 'lstm')
            dropout_input: Dropout on raw input features before first conv
            dropout_middle: Dropout between intermediate conv layers (None = disabled)
        """
        super(GraphSAGENet, self).__init__()
        
        self.K = K
        self.aggr = aggr
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_input  = dropout_input if dropout_input  is not None else dropout
        self.dropout_middle = dropout_middle if dropout_middle is not None else None
        
        self.convs = nn.ModuleList()
        
        if K == 0:
            pass
        elif K == 1:
            self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
        else:
            self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggr))
            for _ in range(K - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        classifier_input_dim = num_features if K == 0 else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        self.layer_classifiers = nn.ModuleList()
        self.layer_classifiers.append(nn.Linear(num_features, num_classes))
        for _ in range(len(self.convs)):
            self.layer_classifiers.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        if self.K == 0:
            return self.classifier(x)
        
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def forward_with_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        
        embeddings = [x.clone()]  # k=0: raw input features (before dropout)
        
        if self.K == 0:
            logits = self.classifier(x)
            return embeddings, logits
        
        if self.dropout_input is not None:
            x = F.dropout(x, p=self.dropout_input, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.K - 1:
                x = F.relu(x)
                if self.dropout_middle is not None:
                    x = F.dropout(x, p=self.dropout_middle, training=self.training)
            embeddings.append(x.clone())
        
        logits = self.classifier(x)
        return embeddings, logits
    
    def forward_with_classifier_head(self, data):
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
