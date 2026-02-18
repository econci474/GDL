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
    dropout_input  = config.get("dropout_input")
    dropout_middle = config.get("dropout_middle")
    if model_name == "GCN":
        return GCNNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
            normalize=True,
        )
    elif model_name == "GAT":
        return GATNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            heads=config.get("gat_heads", 8),
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
        )
    elif model_name == "GraphSAGE":
        return GraphSAGENet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            aggr=config.get("sage_aggr", "mean"),
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
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
    loss_type="weighted_ce_plus_R",  # see taxonomy in config.py
    beta=0.5,                        # for exponential weighting
    use_exponential_weights=True,    # True = depth-decay weights on layer losses
    lambda_R=1.0,                    # weight of curvature regularizer
    R_mode="smooth",                 # "hard" or "smooth"
    R_alpha=10.0,                    # softplus sharpness for smooth
    entropy_floor=None,              # e.g. 0.2 to discourage low entropy
    per_class_R=False,               # if True: average R by class first
    class_R_weights=None,            # optional tensor [C] summing to 1
    band_lower=-1.0,                 # lower bound for curvature band penalty
    band_upper=0.0,                  # upper bound for curvature band penalty
):
    """
    loss_type options:
      'ce_only'            - unweighted CE, no regulariser
      'weighted_ce'        - class-weighted CE, no regulariser
      'ce_plus_R'          - unweighted CE + curvature regulariser
      'weighted_ce_plus_R' - class-weighted CE + curvature regulariser
      'R_only'             - curvature regulariser only (ablation)
    """
    model.train()
    data = to_device(data, device)
    optimizer.zero_grad()

    layer_logits, layer_probs = model.forward_with_classifier_head(data)
    L = len(layer_logits)   # typically K+1 heads (k=0..K)

    # ---- Compute entropy per layer: H shape [L, N] in [0,1]
    probs_stack = torch.stack(layer_probs, dim=0)  # [L, N, C]
    H = node_entropy_from_probs(probs_stack, normalize=True)  # [L, N]

    # ---- Compute curvature c for interior layers only: shape [L-2, N]
    delta2 = second_finite_diff(H)                  # [L-2, N]

    # ---- Per-node curvature penalty for each interior layer: [L-2, N]
    R_node = band_penalty(delta2, mode=R_mode, alpha=R_alpha,
                          lower_bound=band_lower, upper_bound=band_upper)  # [L-2, N]

    # Optional: entropy floor penalty (also per interior layer)
    if entropy_floor is not None:
        floor_pen = F.relu(entropy_floor - H[1:-1]).pow(2)  # [L-2, N]
        R_node = R_node + floor_pen

    # ---- Determine which components to use
    use_weights = loss_type in ('weighted_ce', 'weighted_ce_plus_R')
    use_R       = loss_type in ('ce_plus_R', 'weighted_ce_plus_R', 'R_only')

    # ---- Build per-layer losses
    layer_losses = []
    train_mask = data.train_mask
    y = data.y
    num_classes = int(y.max()) + 1

    for k, logits_k in enumerate(layer_logits):
        # CE term
        if loss_type != 'R_only':
            ce_k = F.cross_entropy(
                logits_k[train_mask], y[train_mask],
                weight=class_weights if use_weights else None
            )
        else:
            ce_k = torch.zeros((), device=logits_k.device)

        # Curvature regulariser (only for interior layers k>=2)
        if use_R and 2 <= k <= L - 1:
            Rk_per_node = R_node[k - 2]  # [N]
            if per_class_R:
                class_means = classwise_mean(Rk_per_node, y, train_mask, num_classes)
                R_k = class_means.mean() if class_R_weights is None else \
                      (class_R_weights.to(logits_k.device) * class_means).sum()
            else:
                R_k = Rk_per_node[train_mask].mean()
        else:
            R_k = torch.zeros((), device=logits_k.device)

        if loss_type == 'R_only':
            loss_k = lambda_R * R_k
        elif use_R:
            loss_k = ce_k + lambda_R * R_k
        else:  # ce_only or weighted_ce
            loss_k = ce_k

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
    loss_type="weighted_ce_plus_R",  # MUST match training loss_type
    beta=0.5,
    lambda_R=1.0,
    use_exponential_weights=True,
    R_mode="smooth",
    R_alpha=10.0,
    entropy_floor=None,
    band_lower=-1.0,
    band_upper=0.0,
):
    """Evaluate val loss using the same loss_type as training."""
    model.eval()
    data = to_device(data_split, device)

    layer_logits, layer_probs = model.forward_with_classifier_head(data)
    L = len(layer_logits)
    val_mask = data.val_mask
    y = data.y

    # Determine which components to use (mirrors train_epoch_multi_layer)
    use_weights = loss_type in ('weighted_ce', 'weighted_ce_plus_R')
    use_R       = loss_type in ('ce_plus_R', 'weighted_ce_plus_R', 'R_only')

    # per-layer CE on val
    ce_layers = []
    for logits_k in layer_logits:
        if loss_type != 'R_only':
            ce = F.cross_entropy(
                logits_k[val_mask], y[val_mask],
                weight=class_weights if use_weights else None
            )
        else:
            ce = torch.zeros((), device=device)
        ce_layers.append(ce)

    # entropy H: [L, N] (normalized)
    probs_stack = torch.stack(layer_probs, dim=0)             # [L, N, C]
    H = node_entropy_from_probs(probs_stack, normalize=True)  # [L, N]

    # curvature: shape [L-2, N]
    a_back = H[2:] - 2.0 * H[1:-1] + H[:-2]

    # band penalty
    R_node = band_penalty(a_back, mode=R_mode, alpha=R_alpha,
                          lower_bound=band_lower, upper_bound=band_upper)

    if entropy_floor is not None:
        R_node = R_node + F.relu(entropy_floor - H[2:]).pow(2)

    # per-layer R_k on val (defined only for k>=2)
    R_layers = []
    for k in range(L):
        if use_R and k >= 2:
            Rk = R_node[k - 2][val_mask].mean()
        else:
            Rk = torch.zeros((), device=device)
        R_layers.append(Rk)

    # combine losses (same logic as train)
    if loss_type == 'R_only':
        layer_losses = [lambda_R * R_layers[k] for k in range(L)]
    elif use_R:
        layer_losses = [ce_layers[k] + lambda_R * R_layers[k] for k in range(L)]
    else:  # ce_only or weighted_ce
        layer_losses = ce_layers

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
    split_id: int,
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

    # Compute class weights (used when loss_type requires them)
    class_weights = compute_class_weights(data_split).to(device)
    print(f"Class weights: {class_weights}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_counter = 0
    train_log = []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared kwargs for train and eval — ensures they use the same loss
    loss_kwargs = dict(
        loss_type=loss_type,
        beta=config.get("beta", 0.5),
        use_exponential_weights=config.get("exponential_decay", True),
        lambda_R=config.get("lambda_R", 1.0),
        R_mode=config.get("R_mode", "smooth"),
        R_alpha=config.get("R_alpha", 10.0),
        entropy_floor=config.get("entropy_floor", None),
        band_lower=config.get("band_lower", -1.5),
        band_upper=config.get("band_upper", 0.5),
    )

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_acc, layer_losses = train_epoch_multi_layer(
            model, data_split, optimizer, device, class_weights,
            per_class_R=config.get("per_class_R", False),
            **loss_kwargs,
        )

        val_loss, val_acc = evaluate_multi_layer_with_R(
            model, data_split, device, class_weights,
            **loss_kwargs,
        )

        scheduler.step(val_loss)

        log_entry = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            # Hyperparameter metadata
            loss_type=loss_type,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            patience=config["patience"],
            max_epochs=config["max_epochs"],
            hidden_dim=config["hidden_dim"],
            K=K,
            beta=config.get("beta", 0.5),
            lambda_R=config.get("lambda_R", 1.0),
            R_mode=config.get("R_mode", "smooth"),
            entropy_floor=config.get("entropy_floor", None),
            per_class_R=config.get("per_class_R", False),
            exponential_decay=config.get("exponential_decay", True),
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
            best_val_acc  = val_acc
            best_epoch    = epoch
            patience_counter = 0

            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    val_loss=val_loss,
                    val_acc=val_acc,
                    hyperparams=dict(
                        loss_type=loss_type,
                        lr=config["lr"],
                        weight_decay=config["weight_decay"],
                        patience=config["patience"],
                        max_epochs=config["max_epochs"],
                        hidden_dim=config["hidden_dim"],
                        K=K,
                        beta=config.get("beta", 0.5),
                        lambda_R=config.get("lambda_R", 1.0),
                        R_mode=config.get("R_mode", "smooth"),
                        entropy_floor=config.get("entropy_floor", None),
                        per_class_R=config.get("per_class_R", False),
                        exponential_decay=config.get("exponential_decay", True),
                        band_lower=config.get("band_lower", -1.5),
                        band_upper=config.get("band_upper", 0.5),
                    ),
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
        split_id=split_id,
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
            split_id=split_id,
            config=config,
            data_split=data_split,
            num_classes=num_classes,
            output_dir=output_dir,
            loss_type=args.loss_type,
            beta=config.get("beta", args.beta),
        )
        all_results.append(result)

        # Append to sweep results tracker
        try:
            from src.results_tracker import append_result
            append_result(dict(
                dataset=dataset_name, model=model_name,
                loss_type=args.loss_type, K=K, seed=seed, split=split_id,
                lr=config["lr"], weight_decay=config["weight_decay"],
                patience=config["patience"], max_epochs=config["max_epochs"],
                hidden_dim=config["hidden_dim"],
                beta=config.get("beta", 0.5),
                lambda_r=config.get("lambda_R", 1.0),
                entropy_floor=config.get("entropy_floor", None),
                per_class_r=config.get("per_class_R", False),
                band_lower=config.get("band_lower", -1.5),
                band_upper=config.get("band_upper", 0.5),
                best_epoch=result["best_epoch"],
                best_val_loss=result["best_val_loss"],
                best_val_acc=result["best_val_acc"],
            ))
        except ImportError:
            pass  # results_tracker optional

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training complete for {len(split_ids)} split(s)")
    if len(all_results) > 1:
        avg_val_acc = sum(r['best_val_acc'] for r in all_results) / len(all_results)
        print(f"Average val accuracy: {avg_val_acc:.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Train GNN with classifier heads')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--loss-type', type=str, default='weighted_ce_plus_R',
                        choices=['ce_only', 'weighted_ce', 'ce_plus_R',
                                 'weighted_ce_plus_R', 'R_only'])
    parser.add_argument('--root-dir', type=str, default='data')
    parser.add_argument('--normalize-planetoid', action='store_true', default=True)
    parser.add_argument('--planetoid-split', type=str, default='public')
    parser.add_argument('--split-mode', type=str, default='auto',
                        choices=['auto', 'all', 'first'])

    # Hyperparameter overrides (override config defaults)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--lambda-r', type=float, default=None)
    parser.add_argument('--entropy-floor', type=float, default=None)
    parser.add_argument('--per-class-r', action='store_true', default=None)
    parser.add_argument('--band-lower', type=float, default=None)
    parser.add_argument('--band-upper', type=float, default=None)

    args = parser.parse_args()

    # Build config from module, then apply dataset-type defaults, then CLI overrides
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}

    # Apply dataset-type defaults
    if args.dataset in cfg.homophilous_datasets:
        config.update(cfg.defaults_homophilous)
    elif args.dataset in cfg.heterophilous_datasets:
        config.update(cfg.defaults_heterophilous)

    # Apply CLI overrides (only if explicitly provided)
    cli_overrides = {
        'lr':           args.lr,
        'weight_decay': args.weight_decay,
        'patience':     args.patience,
        'max_epochs':   args.max_epochs,
        'hidden_dim':   args.hidden_dim,
        'beta':         args.beta,
        'lambda_R':     args.lambda_r,
        'entropy_floor':args.entropy_floor,
        'per_class_R':  args.per_class_r,
        'band_lower':   args.band_lower,
        'band_upper':   args.band_upper,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            config[key] = val

    train_gnn_classifier_heads(
        args.dataset, args.model, args.K, args.seed, args, config
    )

if __name__ == '__main__':
    main()
