"""Data loading and preprocessing for ECQ experiments."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


# Row ranges in ECQ6_DF_ap.csv for each conductor interval.
# Each interval contains ~36-40k curves from the ~1.5M total.
CONDUCTOR_RANGES = {
    "[0, 10000]": (0, 36838),
    "[100000, 110000]": (402856, 442523),
    "[200000, 210000]": (791985, 830890),
    "[300000, 310000]": (1168364, 1205106),
}


def get_prime_columns(num_ans=10000):
    """Return list of string column names for primes up to num_ans.

    These correspond to the a_p columns in ECQ6_DF_ap.csv.
    For num_ans=10000, returns 1229 primes.
    """
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    return [str(n + 1) for n in range(num_ans) if is_prime(n + 1)]


def load_ecq_data(csv_path, row_start, row_end, feature_columns):
    """Load a slice of ECQ6_DF_ap.csv by row range.

    Args:
        csv_path: Path to ECQ6_DF_ap.csv.
        row_start: First row to read (0-indexed, excluding header).
        row_end: Last row (exclusive).
        feature_columns: List of prime column names to use as features.

    Returns:
        X: np.ndarray of shape (n_samples, n_features) — a_p values.
        y: np.ndarray of shape (n_samples,) — order of vanishing (rank).
    """
    header = pd.read_csv(csv_path, nrows=0).columns
    nrows = row_end - row_start
    df = pd.read_csv(
        csv_path, skiprows=row_start + 1, nrows=nrows,
        low_memory=False, header=None, names=header,
    )
    X = df[feature_columns].values
    y = df['order_of_vanishing'].values
    return X, y


def prepare_tensors(X, y, test_size=0.2, random_state=None, device=None):
    """Split, scale, reshape, and convert to PyTorch tensors.

    Applies StandardScaler and reshapes to (batch, 1, seq_len) for Conv1d.

    Returns:
        X_train, X_test, y_train, y_test: tensors on device.
        scaler: fitted StandardScaler (needed for inference on new data).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN: (batch, 1_channel, seq_len)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    return X_train, X_test, y_train, y_test, scaler
