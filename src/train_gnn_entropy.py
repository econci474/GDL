# Train model with classifier head with entropy regulariser

import torch
import math
import pandas as pd
from pathlib import Path
import argparse
import sys

# ADD to imports section:
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.utils import get_device, set_seed, to_device
from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
import torch.nn.functional as F

# Helper functions
def compute_class_weights(data):
    """
    Compute class weights for class-weighted cross-entropy.
    Returns tensor of shape (C,) where weights[c] = 1/n_c
    """
    num_classes = int(data.y.max()) + 1
    class_counts = torch.bincount(data.y[data.train_mask], minlength=num_classes).float()
    # Avoid division by zero
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
    return class_weights

def get_num_splits(data):
    return data.train_mask.size(1) if data.train_mask.dim() > 1 else 1

def select_split_masks(data, split_id: int):
    # Make a copy so we don't overwrite masks for later splits
    data_s = data.clone()
    if data_s.train_mask.dim() > 1:
        data_s.train_mask = data_s.train_mask[:, split_id]
        data_s.val_mask   = data_s.val_mask[:, split_id]
        data_s.test_mask  = data_s.test_mask[:, split_id]
    return data_s

def build_model(model_name: str, data, num_classes: int, K: int, config: dict):
    """Factory for models."""
    if model_name == "GCN":
        return GCNNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            dropout=None,
            normalize=True,
        )
    elif model_name == "GAT":
        return GATNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            heads=config.get("gat_heads", 8),
            dropout=None,
        )
    elif model_name == "GraphSAGE":
        return GraphSAGENet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            aggr=config.get("sage_aggr", "mean"),
            dropout=None,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use GCN, GAT, GraphSAGE.")

def node_entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12, normalize: bool = True):
    """
    probs: [N, C] (single layer) OR [K, N, C] (stacked layers)
    returns:
      H:   [N] or [K, N] predictive entropy per node
    """
    probs = probs.clamp_min(eps)
    H = -(probs * probs.log()).sum(dim=-1)  # sums over classes

    if normalize:
        C = probs.size(-1) #the column C
        H = H / math.log(C)  # now in [0,1]
    return H

def second_finite_diff(H: torch.Tensor):
    """
    H: [K, N] entropy across layers k=2..K-1
    returns delta2: [K-2, N] where delta2[k] = H[k] - 2H[k-1] + H[k-2]
    """
    return H[2:] - 2.0 * H[1:-1] + H[:-2]

def soft_hinge(x: torch.Tensor, alpha: float = 10.0):
    """
    Smooth approximation to relu(x):
      relu(x) ~= softplus(alpha*x)/alpha
    """
    return F.softplus(alpha * x) / alpha

def band_penalty(c: torch.Tensor, mode: str = "hard", alpha: float = 10.0, 
                 lower_bound: float = -1.0, upper_bound: float = 0.0):
    """
    Penalize c outside [lower_bound, upper_bound].
    c: any shape
    returns penalty with same shape.

    hard:
      relu(c - upper_bound)^2 + relu(lower_bound - c)^2
    smooth:
      soft_hinge(c - upper_bound)^2 + soft_hinge(lower_bound - c)^2
    """
    if mode == "hard":
        pos = F.relu(c - upper_bound)
        neg = F.relu(lower_bound - c)
    elif mode == "smooth":
        pos = soft_hinge(c - upper_bound, alpha=alpha)
        neg = soft_hinge(lower_bound - c, alpha=alpha)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pos.pow(2) + neg.pow(2)

def classwise_mean(values: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, num_classes: int, eps: float = 1e-12):
    """
    values: [N] per-node values
    y: [N] int labels
    mask: [N] bool mask (e.g., train_mask)
    returns means: [C] classwise means (0 if class has no masked nodes)
    """
    idx = mask.nonzero(as_tuple=False).view(-1)
    y_m = y[idx]               # [M]
    v_m = values[idx]          # [M]

    sums = torch.zeros(num_classes, device=values.device)
    cnts = torch.zeros(num_classes, device=values.device)

    sums.scatter_add_(0, y_m, v_m)
    cnts.scatter_add_(0, y_m, torch.ones_like(v_m))

    means = sums / (cnts + eps)
    return means

def train_epoch_multi_layer(
    model, data, optimizer, device, class_weights,
    *,
    loss_type="ce_plus_R",          # "ce_plus_R" (recommended)
    beta=0.5,                       # for exponential weighting (optional)
    use_exponential_weights=True,   # True = depth-decay weights on layer losses
    lambda_R=1.0,                   # weight of curvature regularizer
    R_mode="smooth",                # "hard" or "smooth"
    R_alpha=10.0,                   # softplus sharpness for smooth
    entropy_floor=None,             # e.g. 0.2 to discourage low entropy
    per_class_R=False,              # if True: average R by class first
    class_R_weights=None,           # optional tensor [C] summing to 1
    band_lower=-1.0,                # lower bound for curvature band penalty
    band_upper=0.0,                 # upper bound for curvature band penalty
):
    model.train()
    data = to_device(data, device)
    optimizer.zero_grad()

    layer_logits, layer_probs = model.forward_with_classifier_head(data)
    L = len(layer_logits)   # typically K+1 heads (k=0..K)

    # ---- Compute entropy per layer: H shape [L, N] in [0,1]
    probs_stack = torch.stack(layer_probs, dim=0)  # [L, N, C]
    H = node_entropy_from_probs(probs_stack, normalize=True)  # [L, N]

    # ---- Compute curvature c for interior layers only: shape [L-2, N]
    # delta2 has [-2,2] range when H in [0,1]
    delta2 = second_finite_diff(H)                  # [L-2, N]

    # ---- Per-node curvature penalty for each interior layer: [L-2, N]
    R_node = band_penalty(delta2, mode=R_mode, alpha=R_alpha, 
                         lower_bound=band_lower, upper_bound=band_upper)  # [L-2, N]

    # Optional: entropy floor penalty (also per interior layer)
    if entropy_floor is not None:
        # apply to interior layers aligned with c (layers 1..L-2)
        floor_pen = F.relu(entropy_floor - H[1:-1]).pow(2)  # [L-2, N]
        R_node = R_node + floor_pen

    # ---- Build per-layer losses (CE_k + lambda_R * R_k)
    layer_losses = []
    train_mask = data.train_mask
    y = data.y
    num_classes = int(y.max()) + 1

    for k, logits_k in enumerate(layer_logits):
        # Always use class-weighted CE
        ce_k = F.cross_entropy(
            logits_k[train_mask], y[train_mask],
            weight=class_weights
        )

        # Curvature regularizer aligned to this layer k:
        # c (and R_node) is defined for interior layers k=1..L-2
        if 2 <= k <= L - 1:
            Rk_per_node = R_node[k-2]  # [N] corresponds to curvature centered at layer k

            if per_class_R:
                # Compute classwise means of R, then average across classes
                class_means = classwise_mean(Rk_per_node, y, train_mask, num_classes)  # [C]
                if class_R_weights is None:
                    R_k = class_means.mean()
                else:
                    # ensure class_R_weights sums to 1
                    R_k = (class_R_weights.to(device) * class_means).sum()
            else:
                # Plain mean over training nodes
                R_k = Rk_per_node[train_mask].mean()
        else:
            R_k = torch.zeros((), device=device)  # scalar 0

        if loss_type == "ce_plus_R":
            loss_k = ce_k + lambda_R * R_k
        elif loss_type == "ce_only":
            loss_k = ce_k
        elif loss_type == "R_only":
            loss_k = lambda_R * R_k
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        layer_losses.append(loss_k)

    # ---- Combine layer losses (depth-decay weights or simple sum)
    if use_exponential_weights:
        # weight final head = 1, earlier heads get exp(-beta*(K-k)) style
        # Here L = K+1 heads indexed 0..K. Let K_head = L-1.
        K_head = L - 1
        total_loss = layer_losses[-1]
        for k in range(0, K_head):  # only earlier heads
            alpha_k = math.exp(-beta * (K_head - k))
            total_loss = total_loss + alpha_k * layer_losses[k]
    else:
        total_loss = torch.stack(layer_losses).sum()

    total_loss.backward()
    optimizer.step()

    # training accuracy using final head
    pred = layer_logits[-1][train_mask].argmax(dim=1)
    acc = (pred == y[train_mask]).float().mean()

    return total_loss.item(), acc.item(), [l.item() for l in layer_losses]


@torch.no_grad()
def evaluate_multi_layer_with_R(
    model, data_split, device, class_weights,
    *,
    beta=0.5,
    lambda_R=1.0,
    use_exponential_weights=True,
    R_mode="smooth",
    R_alpha=10.0,
    entropy_floor=None,
    band_lower=-1.0,
    band_upper=0.0,
):
    model.eval()
    data = to_device(data_split, device)

    layer_logits, layer_probs = model.forward_with_classifier_head(data)
    L = len(layer_logits)
    val_mask = data.val_mask
    y = data.y

    # per-layer CE on val (always class-weighted)
    ce_layers = []
    
    for logits_k in layer_logits:
        ce = F.cross_entropy(
            logits_k[val_mask], y[val_mask],
            weight=class_weights
        )
        ce_layers.append(ce)

    # entropy H: [L, N] (normalized)
    probs_stack = torch.stack(layer_probs, dim=0)             # [L, N, C]
    H = node_entropy_from_probs(probs_stack, normalize=True)  # [L, N]

    # backward second diff aligns to layers k=2..L-1 : shape [L-2, N]
    a_back = H[2:] - 2.0 * H[1:-1] + H[:-2]                   # [L-2, N]

    # band penalty on acceleration
    R_node = band_penalty(a_back, mode=R_mode, alpha=R_alpha,
                         lower_bound=band_lower, upper_bound=band_upper)  # [L-2, N]

    if entropy_floor is not None:
        R_node = R_node + F.relu(entropy_floor - H[2:]).pow(2)  # note: aligns w/ k=2..L-1

    # per-layer R_k on val (defined only for k>=2)
    R_layers = []
    for k in range(L):
        if k >= 2:
            Rk = R_node[k-2][val_mask].mean()
        else:
            Rk = torch.zeros((), device=device)
        R_layers.append(Rk)

    # combine losses
    layer_losses = [ce_layers[k] + lambda_R * R_layers[k] for k in range(L)]

    if use_exponential_weights:
        K_head = L - 1
        total = layer_losses[-1]
        for k in range(0, K_head):
            alpha_k = math.exp(-beta * (K_head - k))
            total = total + alpha_k * layer_losses[k]
        val_obj = total
    else:
        val_obj = torch.stack(layer_losses).sum()

    # final-head acc on val
    val_pred = layer_logits[-1][val_mask].argmax(dim=1)
    val_acc = (val_pred == y[val_mask]).float().mean()

    return float(val_obj.item()), float(val_acc.item())


def run_one_split(
    *,
    dataset_name: str,
    model_name: str,
    K: int,
    seed: int,
    config: dict,
    data_split,
    num_classes: int,
    output_dir: Path,
    loss_type: str,
    beta: float,
):
    device = get_device()
    set_seed(seed)

    model = build_model(model_name, data_split, num_classes, K, config).to(device)

    # Always use class-weighted CE
    class_weights = compute_class_weights(data_split)
    class_weights = class_weights.to(device)
    print(f"Using class weights: {class_weights}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_log = []

    output_dir.mkdir(parents=True, exist_ok=True)


    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_acc, layer_losses = train_epoch_multi_layer(
            model, data_split, optimizer, device, class_weights,
            loss_type=config.get("train_loss_type", "ce_plus_R"),
            beta=config.get("beta", 0.5),
            use_exponential_weights=config.get("use_exponential_weights", True),
            lambda_R=config.get("lambda_R", 1.0),
            R_mode=config.get("R_mode", "hard"),
            R_alpha=config.get("R_alpha", 10.0),
            entropy_floor=config.get("entropy_floor", None),
            per_class_R=config.get("per_class_R", False),
            band_lower=config.get("band_lower", -1.0),
            band_upper=config.get("band_upper", 0.0)
        )

        val_loss, val_acc = evaluate_multi_layer_with_R(
            model, data_split, device, class_weights,
            band_lower=config.get("band_lower", -1.0),
            band_upper=config.get("band_upper", 0.0)
        )

        scheduler.step(val_loss)

        log_entry = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            # Hyperparameter metadata
            lr=config["lr"],
            patience=config["patience"],
            max_epochs=config["max_epochs"],
            K=K,
            loss_type=config.get("train_loss_type", "ce_plus_R"),
            lambda_R=config.get("lambda_R", 1.0),
            R_mode=config.get("R_mode", "hard"),
            # Entropy regularization specific parameters
            entropy_floor=config.get("entropy_floor", None),
            per_class_R=config.get("per_class_R", False),
            exponential_decay=config.get("exponential_decay", True),
            beta=config.get("beta", 0.5),
            band_lower=config.get("band_lower", -1.5),
            band_upper=config.get("band_upper", 0.5),
        )
        for k, loss_k in enumerate(layer_losses):
            log_entry[f"train_loss_layer_{k}"] = loss_k

        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    val_loss=val_loss,
                    val_acc=val_acc,
                ),
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)

    print(f"\n✓ Split complete: {output_dir}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val acc:  {best_val_acc:.4f}")

    return dict(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )

def train_gnn_classifier_heads(
    dataset_name: str,
    model_name: str,
    K: int,
    seed: int,
    args,
    config: dict
):
    # Load dataset
    data, num_classes, dataset_kind = load_dataset(
        dataset_name,
        root_dir=args.root_dir,
        planetoid_normalize=args.normalize_planetoid,
        planetoid_split=args.planetoid_split,
    )

    # Decide split behavior
    num_splits = get_num_splits(data)

    if args.split_mode == "auto":
        split_ids = list(range(num_splits)) if num_splits > 1 else [0]
    elif args.split_mode == "all":
        split_ids = list(range(num_splits))
    elif args.split_mode == "first":
        split_ids = [0]
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")

    # Base directory with loss_type and regularization config
    # Create descriptive subdirectory based on regularization settings
    if args.loss_type in ['ce_plus_R', 'R_only']:
        # Include R configuration in directory name
        R_config_parts = []
        R_config_parts.append(f"R{config.get('lambda_R', 1.0):.1f}")
        R_config_parts.append(config.get('R_mode', 'hard'))
        
        if config.get('entropy_floor') is not None:
            R_config_parts.append(f"floor{config.get('entropy_floor'):.2f}")
        
        if config.get('per_class_R', False):
            R_config_parts.append("perclass")
        
        # Add band bounds if non-default (default: -1.0 to 0.0)
        band_lower = config.get('band_lower', -1.0)
        band_upper = config.get('band_upper', 0.0)
        if band_lower != -1.0 or band_upper != 0.0:
            R_config_parts.append(f"band{band_lower:.1f}to{band_upper:.1f}")
        
        loss_dir = f"{args.loss_type}_{'_'.join(R_config_parts)}"
    else:
        loss_dir = args.loss_type
    
    base_dir = Path(cfg.classifier_heads_dir) / loss_dir / dataset_name / model_name / f"seed_{seed}" / f"K_{K}"

    # Print header
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {num_classes}")
    print(f"  Train nodes: {data.train_mask[:, 0].sum() if data.train_mask.dim() > 1 else data.train_mask.sum()}")
    if num_splits > 1:
        print(f"  Val nodes:   {data.val_mask[:, 0].sum()}")
        print(f"  Test nodes:  {data.test_mask[:, 0].sum()}")
    else:
        print(f"  Val nodes:   {data.val_mask.sum()}")
        print(f"  Test nodes:  {data.test_mask.sum()}")
    print(f"\nModel: {model_name} | K={K} | seed={seed}")
    print(f"Loss type: {args.loss_type} | Beta: {args.beta}")
    print(f"Dataset kind: {dataset_kind} | Planetoid split: {args.planetoid_split}")
    print(f"Splits to train: {len(split_ids)}")
    print(f"{'='*60}\n")

    all_results = []

    for split_id in split_ids:
        data_split = select_split_masks(data, split_id)

        if num_splits > 1:
            output_dir = base_dir / f"split_{split_id}"
        else:
            output_dir = base_dir

        result = run_one_split(
            dataset_name=dataset_name,
            model_name=model_name,
            K=K,
            seed=seed,
            config=config,
            data_split=data_split,
            num_classes=num_classes,
            output_dir=output_dir,
            loss_type=args.loss_type,
            beta=args.beta,
        )
        all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training complete for {len(split_ids)} split(s)")
    if len(all_results) > 1:
        avg_test_acc = sum(r['best_test_acc'] for r in all_results) / len(all_results)
        print(f"Average test accuracy: {avg_test_acc:.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--loss-type', type=str, default='ce_plus_R',
                       choices=['ce_plus_R', 'ce_only', 'R_only']) 
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--root-dir', type=str, default='data')
    parser.add_argument('--normalize-planetoid', action='store_true')
    parser.add_argument('--planetoid-split', type=str, default='public')
    parser.add_argument('--split-mode', type=str, default='auto',
                       choices=['auto', 'all', 'first'])
    
    args = parser.parse_args()
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    train_gnn_classifier_heads(
        args.dataset, args.model, args.K, args.seed, args, config
    )

if __name__ == '__main__':
    main()
