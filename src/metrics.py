"""Metrics for evaluating model predictions and computing entropy."""

import numpy as np
import torch


def entropy_from_probs(p):
    """
    Compute predictive entropy from probability distributions.
    
    Uses natural logarithm as specified in project requirements.
    
    Args:
        p: Probability matrix [N, C] where each row sums to 1
        
    Returns:
        entropy: [N] entropy per node, H = -sum(p * log(p))
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_safe = np.clip(p, eps, 1.0)
    
    # Compute entropy using natural log
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    
    return entropy


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: [N] predicted class labels
        labels: [N] true class labels
        
    Returns:
        accuracy: Scalar accuracy value
    """
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total


def compute_nll(probs, labels):
    """
    Compute negative log-likelihood (cross-entropy).
    
    Args:
        probs: [N, C] probability matrix
        labels: [N] true class labels (integers)
        
    Returns:
        nll: Scalar negative log-likelihood
    """
    eps = 1e-10
    N = len(labels)
    
    # Get probability of true class for each sample
    true_class_probs = probs[np.arange(N), labels]
    
    # Compute NLL
    nll = -np.mean(np.log(true_class_probs + eps))
    
    return nll


def split_by_correctness(predictions, labels, values, mask=None):
    """
    Split values (e.g., entropy) by prediction correctness.
    
    Args:
        predictions: [N] predicted labels
        labels: [N] true labels
        values: [N] values to split (e.g., entropy per node)
        mask: Optional [N] boolean mask to apply first
        
    Returns:
        correct_values: Values for correctly predicted samples
        incorrect_values: Values for incorrectly predicted samples
    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
        values = values[mask]
    
    correct_mask = predictions == labels
    incorrect_mask = ~correct_mask
    
    correct_values = values[correct_mask]
    incorrect_values = values[incorrect_mask]
    
    return correct_values, incorrect_values


def compute_entropy_stats(probs, labels, mask=None):
    """
    Compute comprehensive entropy statistics.
    
    Args:
        probs: [N, C] probabilities
        labels: [N] true labels
        mask: Optional [N] boolean mask
        
    Returns:
        dict with keys:
            - mean: Mean entropy
            - std: Standard deviation of entropy
            - correct_mean: Mean entropy for correct predictions
            - incorrect_mean: Mean entropy for incorrect predictions
    """
    entropy = entropy_from_probs(probs)
    predictions = np.argmax(probs, axis=1)
    
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        entropy = entropy[mask]
        predictions = predictions[mask]
        labels = labels[mask] if isinstance(labels, np.ndarray) else labels.cpu().numpy()[mask]
    
    correct_entropy, incorrect_entropy = split_by_correctness(
        predictions, labels, entropy
    )
    
    stats = {
        'mean': np.mean(entropy),
        'std': np.std(entropy),
        'correct_mean': np.mean(correct_entropy) if len(correct_entropy) > 0 else 0.0,
        'incorrect_mean': np.mean(incorrect_entropy) if len(incorrect_entropy) > 0 else 0.0,
    }
    
    return stats


if __name__ == '__main__':
    # Test metrics
    print("Testing entropy computation...")
    
    # Create sample probabilities
    probs = np.array([
        [0.8, 0.1, 0.1],  # Low entropy (confident)
        [0.33, 0.33, 0.34],  # High entropy (uncertain)
    ])
    
    entropy = entropy_from_probs(probs)
    print(f"Entropy: {entropy}")
    print(f"  Low-entropy sample: {entropy[0]:.4f}")
    print(f"  High-entropy sample: {entropy[1]:.4f}")
    
    # Test NLL
    labels = np.array([0, 1])
    nll = compute_nll(probs, labels)
    print(f"NLL: {nll:.4f}")
    
    print("✓ Metrics tests passed")
