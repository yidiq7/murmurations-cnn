"""Saliency map computation for CNN interpretability."""

import torch
import numpy as np


def compute_saliency(model, X_test, n_samples=3000):
    """Averaged absolute saliency scores w_p (eq. saliencyscore in the paper).

    Averages |w_p^v(E)| over a random subset of test samples and their
    predicted classes.

    Returns:
        np.ndarray of shape (n_features,) — averaged saliency per prime.
    """
    model.eval()
    indices = torch.randperm(X_test.size(0))[:n_samples]
    input_data = X_test[indices].clone().detach().requires_grad_(True)

    output = model(input_data)
    _, predicted_class = torch.max(output, 1)

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[torch.arange(output.size(0)), predicted_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    saliency = input_data.grad.abs().mean(dim=0).squeeze().detach().cpu().numpy()
    return saliency


def compute_class_saliency(model, X_test, num_classes, n_samples=3000):
    """Per-class signed saliency W_p^v (eq. bigW in the paper).

    For each predicted class v, averages the signed gradient w_p^v(E) over
    samples predicted as class v, then normalizes by the max absolute value.

    Returns:
        dict mapping class_index -> np.ndarray of shape (n_features,).
        Values are normalized to [-1, 1].
        Classes with no predictions are omitted.
    """
    model.eval()
    indices = torch.randperm(X_test.size(0))[:n_samples]
    input_data = X_test[indices].clone().detach().requires_grad_(True)

    output = model(input_data)
    _, predicted_class = torch.max(output, 1)

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[torch.arange(output.size(0)), predicted_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    saliency = input_data.grad  # (n_samples, 1, n_features)

    result = {}
    for v in range(num_classes):
        mask = predicted_class == v
        if mask.sum() == 0:
            continue
        class_sal = saliency[mask].mean(dim=0).squeeze().detach().cpu().numpy()
        max_abs = np.max(np.abs(class_sal))
        if max_abs > 0:
            class_sal = class_sal / max_abs
        result[v] = class_sal

    return result
