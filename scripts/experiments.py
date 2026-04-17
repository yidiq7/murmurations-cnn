"""Experiment orchestration: training across conductor ranges, prime sweeps, saliency evolution."""

import os
import time
import gc

import numpy as np
import torch

from .model import CNN
from .data import load_ecq_data, prepare_tensors, CONDUCTOR_RANGES
from .train import train_model
from .saliency import compute_saliency, compute_class_saliency


def train_all_ranges(csv_path, prime_columns, device,
                     max_epochs=100, batch_size=3000, verbose=True):
    """Train the CNN on each conductor range and compute averaged saliency.

    Returns:
        dict mapping range label -> dict with keys:
            'train_acc', 'test_acc', 'model', 'saliency', 'scaler',
            'X_test', 'y_test', 'num_classes'.
    """
    results = {}
    for label, (row_start, row_end) in CONDUCTOR_RANGES.items():
        if verbose:
            print(f'\n=== Training on {label} ===')
        X, y = load_ecq_data(csv_path, row_start, row_end, prime_columns)
        X_train, X_test, y_train, y_test, scaler = prepare_tensors(
            X, y, device=device,
        )
        del X
        gc.collect()

        num_classes = len(set(y))
        model = CNN(
            input_length=X_train.shape[-1], num_classes=num_classes,
        ).to(device)

        result = train_model(
            model, X_train, X_test, y_train, y_test,
            batch_size=batch_size, max_epochs=max_epochs, verbose=verbose,
        )

        saliency = compute_saliency(model, X_test, n_samples=3000)
        result['saliency'] = saliency
        result['X_test'] = X_test
        result['y_test'] = y_test
        result['scaler'] = scaler
        result['num_classes'] = num_classes
        results[label] = result

    return results


# Mapping from conductor range label to save-file name.
_SAVE_FILENAMES = {
    '[0, 10000]': '0',
    '[100000, 110000]': '100000',
    '[200000, 210000]': '200000',
    '[300000, 310000]': '300000',
}


def sweep_n_primes(csv_path, prime_columns, device,
                   save_dir='test_accuracies', batch_size=3000,
                   max_epochs=400, patience=10, verbose=True):
    """Sweep number of primes used for training and record best accuracy.

    Loads pre-computed results from save_dir if available, otherwise trains
    from scratch (slow).

    Returns:
        dict mapping range label -> (primes_array, accuracy_array).
    """
    prime_range = list(range(5, 1229, 10))
    accuracy_vs_primes = {}

    for label, (row_start, row_end) in CONDUCTOR_RANGES.items():
        save_path = os.path.join(save_dir, f'{_SAVE_FILENAMES[label]}.txt')

        if os.path.exists(save_path):
            data = np.loadtxt(save_path)
            accuracy_vs_primes[label] = (data[:, 0], data[:, 1])
            if verbose:
                print(f'Loaded {save_path}')
            continue

        if verbose:
            print(f'\n=== Sweeping number of primes for {label} ===')
        X_full, y = load_ecq_data(csv_path, row_start, row_end, prime_columns)
        best_accs = []

        for n_primes in prime_range:
            start = time.time()
            X = X_full[:, :n_primes]
            X_train, X_test, y_train, y_test, _ = prepare_tensors(
                X, y, test_size=0.2, random_state=1042, device=device,
            )

            num_classes = len(set(y))
            model = CNN(
                input_length=n_primes, num_classes=num_classes,
            ).to(device)
            result = train_model(
                model, X_train, X_test, y_train, y_test,
                batch_size=batch_size, max_epochs=max_epochs,
                patience=patience, verbose=False,
            )
            best_accs.append(result['best_accuracy'])
            if verbose:
                elapsed = time.time() - start
                print(
                    f'  n_primes={n_primes}, '
                    f'best_acc={result["best_accuracy"]:.4f}, '
                    f'time={elapsed:.1f}s'
                )

        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w') as f:
            for p, a in zip(prime_range, best_accs):
                f.write(f'{p} {a}\n')

        accuracy_vs_primes[label] = (prime_range, best_accs)

    return accuracy_vs_primes


def train_saliency_evolution(csv_path, label, row_start, row_end,
                             prime_columns, device,
                             max_epochs=30, batch_size=3000, verbose=True):
    """Train with per-step checkpoints and compute per-class saliency at step 0.

    Returns:
        list of dicts, one per epoch, each with keys:
            'epoch', 'accuracy', 'class_saliency'.
    """
    X, y = load_ecq_data(csv_path, row_start, row_end, prime_columns)
    X_train, X_test, y_train, y_test, scaler = prepare_tensors(
        X, y, device=device,
    )
    del X
    gc.collect()

    num_classes = len(set(y))
    checkpoint_dir = f'Conductor_models/{label}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = CNN(
        input_length=X_train.shape[-1], num_classes=num_classes,
    ).to(device)
    result = train_model(
        model, X_train, X_test, y_train, y_test,
        batch_size=batch_size, max_epochs=max_epochs,
        save_checkpoints=True, checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    saliency_grids = []
    for epoch in range(max_epochs):
        ckpt_path = f'{checkpoint_dir}/{epoch}_0.pth'
        if not os.path.exists(ckpt_path):
            continue

        m = CNN(
            input_length=X_train.shape[-1], num_classes=num_classes,
        ).to(device)
        m.load_state_dict(torch.load(ckpt_path, weights_only=True))
        class_sal = compute_class_saliency(
            m, X_test, num_classes, n_samples=3000,
        )

        acc = (
            result['step_test_acc'][epoch][0]
            if epoch < len(result['step_test_acc'])
            else None
        )
        saliency_grids.append({
            'epoch': epoch,
            'accuracy': acc,
            'class_saliency': class_sal,
        })

    return saliency_grids
