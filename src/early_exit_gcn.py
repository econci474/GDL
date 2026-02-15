
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def node_entropy(probs, eps=1e-12):
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)   # [N]

def per_class_node_entropy(probs, eps=1e-12):
    probs = probs.clamp_min(eps)
    return -(probs * probs.log())  # [N,C]


#Second finite difference across layers l, l-1, l-2
def entropy_accel(H_k, H_km1, H_km2):
    return H_k - 2*H_km1 + H_km2   # [N]

#Ricci curvature threshold
0 > tau > -1

#or learnable parameter
tau = nn.Parameter(torch.tensor(-1.0))

# early exit the node if acceleration of the Hessian is < -1 (could also check < -2)

class GCNNetEarlyExit(nn.Module):
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
    

    def forward_early_exit(self, data, tau=-1, warmup=2):
        """
        Node-wise early exit based on entropy acceleration.

        Args:
            tau: threshold for negative acceleration (-delta2H > tau)
            eps_entropy: entropy must be below this to allow exit
            warmup: require at least `warmup` layers before allowing exit
                    (need k>=2 anyway to compute delta2)

        Returns:
            final_logits: [N, C] logits from the exit layer per node (gathered)
            exit_layer:  [N] int64 layer index where node exited (0..K)
        """
        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        C = self.layer_classifiers[0].out_features
        device = x.device

        # Track per-node exit
        exited = torch.zeros(N, dtype=torch.bool, device=device)
        exit_layer = torch.full((N,), fill_value=self.K, dtype=torch.long, device=device)

        # Store logits per layer so we can output the exit layer logits per node
        logits_layers = []

        # For delta2 entropy
        H_km2 = None
        H_km1 = None

        # k=0 (input features)
        logits0 = self.layer_classifiers[0](x)
        probs0 = F.softmax(logits0, dim=-1)
        H0 = node_entropy(probs0)
        logits_layers.append(logits0)

        H_km2 = None
        H_km1 = H0

        # If K==0, everyone exits at 0
        if self.K == 0:
            exit_layer[:] = 0
            return logits0, exit_layer

        # Sequential conv layers k=1..K
        for k, conv in enumerate(self.convs, start=1):
            x_new = conv(x, edge_index)

            if k < self.K:
                x_new = F.relu(x_new)
                if self.dropout is not None:
                    x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Probe at this depth
            logits_k = self.layer_classifiers[k](x_new)
            probs_k = F.softmax(logits_k, dim=-1)
            Hk = node_entropy(probs_k)
            logits_layers.append(logits_k)

            # Decide exits when we can compute delta2
            if H_km2 is not None and k >= warmup:
                delta2 = Hk - 2*H_km1 + H_km2  # [N]

                # "collapse" = strong downward curvature + already confident e.g. H<0.2
                want_exit = (~exited) & (delta2 < tau) & (Hk < 0.2)

                exit_layer[want_exit] = k
                exited = exited | want_exit

            # Freeze exited nodes: gate=0 -> keep old x, gate=1 -> take new x_new
            # (Hard gate; for soft gate use sigmoid and convex combination)
            gate = (~exited).float().unsqueeze(-1)  # [N,1]
            x = gate * x_new + (1.0 - gate) * x

            # Update entropy history
            H_km2, H_km1 = H_km1, Hk

        # Build final logits per node by selecting the logits at its exit layer
        # sp that the final prediction for each node comes from the layer it exited at.
        # logits_layers is list length K+1 each [N,C]
        stacked = torch.stack(logits_layers, dim=0)     # [K+1, N, C]
        idx = exit_layer.view(1, N, 1).expand(1, N, C)  # [1,N,C]
        final_logits = stacked.gather(dim=0, index=idx).squeeze(0)  # [N,C]

        return final_logits, exit_layer



