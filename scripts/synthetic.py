"""Synthetic data generation and classification for the murmuration experiment.

Sato-Tate sampling adapted from generate_fake_ap.py.
Classification logic adapted from classify_and_plot_fake_ranks.py.
"""

import numpy as np
import torch


def sample_sato_tate_angles(n):
    """Sample n angles from the Sato-Tate distribution, PDF: (2/pi)*sin^2(theta).

    Vectorized version of sample_sato_tate_angle() from generate_fake_ap.py.
    Uses rejection sampling: propose (theta, y) uniformly, accept if y <= PDF(theta).
    """
    max_pdf_val = 2 / np.pi
    results = np.empty(n)
    filled = 0
    while filled < n:
        batch_size = (n - filled) * 2  # oversample (acceptance rate ~50%)
        theta = np.random.uniform(0, np.pi, batch_size)
        y = np.random.uniform(0, max_pdf_val, batch_size)
        pdf_vals = (2 / np.pi) * np.sin(theta) ** 2
        accepted = theta[y <= pdf_vals]
        take = min(len(accepted), n - filled)
        results[filled:filled + take] = accepted[:take]
        filled += take
    return results


def generate_fake_ap(n_sequences, prime_columns, seed=42):
    """Generate synthetic a_p sequences from the Sato-Tate distribution.

    For each prime p, samples theta ~ Sato-Tate, computes cos(theta),
    scales by 2*sqrt(p), and rounds to integer — same procedure as
    generate_fake_ap.py.

    Returns:
        fake_ap_raw: np.ndarray of shape (n_sequences, n_primes), unnormalized
            integer a_p values.
    """
    primes = np.array([float(p) for p in prime_columns])
    n_primes = len(primes)
    max_bounds = 2 * np.sqrt(primes)

    np.random.seed(seed)
    total_samples = n_sequences * n_primes
    angles = sample_sato_tate_angles(total_samples)
    x_p = np.cos(angles).reshape(n_sequences, n_primes)
    fake_ap_raw = np.round(x_p * max_bounds[np.newaxis, :]).astype(np.float32)

    return fake_ap_raw


def classify_fake_ap(model, scaler, fake_ap_raw, prime_columns, device,
                     batch_size=10000):
    """Classify synthetic a_p sequences using a trained model.

    Normalizes to a_tilde_p = a_p / (2*sqrt(p)) to match ECQ data format,
    applies the same StandardScaler, and predicts in batches.

    Returns:
        predictions: np.ndarray of shape (n_sequences,), predicted rank per sample.
    """
    primes = np.array([float(p) for p in prime_columns])
    max_bounds = 2 * np.sqrt(primes)
    n_primes = len(primes)

    fake_normalized = fake_ap_raw / max_bounds[np.newaxis, :]
    fake_scaled = scaler.transform(fake_normalized)
    fake_tensor = torch.tensor(
        fake_scaled.reshape(-1, 1, n_primes), dtype=torch.float32,
    ).to(device)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(fake_tensor), batch_size):
            batch = fake_tensor[i:i + batch_size]
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())

    return np.concatenate(all_preds)
