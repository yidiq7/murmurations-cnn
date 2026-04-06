"""Training utilities for the CNN."""

import torch
import torch.nn as nn
import torch.optim as optim


def calculate_accuracy(y_true, y_pred):
    """Classification accuracy from model output logits."""
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)


def train_model(model, X_train, X_test, y_train, y_test,
                batch_size=3000, max_epochs=100, lr=0.001,
                patience=None, verbose=True,
                save_checkpoints=False, checkpoint_dir=None):
    """Train the CNN model.

    Args:
        model: CNN instance, already on the target device.
        X_train, X_test: tensors of shape (n, 1, seq_len).
        y_train, y_test: label tensors.
        batch_size: Mini-batch size (default 3000).
        max_epochs: Maximum training epochs.
        lr: Learning rate for Adam.
        patience: If set, stop after this many epochs without test accuracy gain.
        verbose: Print per-epoch stats.
        save_checkpoints: Save model state at each (epoch, step).
        checkpoint_dir: Directory for checkpoints.

    Returns:
        dict with 'train_acc', 'test_acc' (lists), 'model'.
        If patience is set: also 'best_accuracy'.
        If save_checkpoints: also 'step_test_acc' (list of lists).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    train_accuracies = []
    test_accuracies = []
    step_test_accuracies = []
    best_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        step_accs = []

        for step, (inputs, labels) in enumerate(train_loader):
            if save_checkpoints and checkpoint_dir:
                model.eval()
                with torch.no_grad():
                    acc = calculate_accuracy(y_test, model(X_test))
                    step_accs.append(acc)
                torch.save(
                    model.state_dict(),
                    f'{checkpoint_dir}/{epoch}_{step}.pth',
                )
                model.train()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if save_checkpoints:
            step_test_accuracies.append(step_accs)

        model.eval()
        with torch.no_grad():
            train_acc = calculate_accuracy(y_train, model(X_train))
            test_acc = calculate_accuracy(y_test, model(X_test))

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if verbose:
            avg_loss = running_loss / len(train_loader)
            print(
                f'Epoch [{epoch + 1}/{max_epochs}], Loss: {avg_loss:.4f}, '
                f'Train: {train_acc:.4f}, Test: {test_acc:.4f}'
            )

        if patience is not None:
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    result = {
        'train_acc': train_accuracies,
        'test_acc': test_accuracies,
        'model': model,
    }
    if patience is not None:
        result['best_accuracy'] = best_accuracy
    if save_checkpoints:
        result['step_test_acc'] = step_test_accuracies
    return result
