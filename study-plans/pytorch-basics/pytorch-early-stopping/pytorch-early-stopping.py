import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses = []
    val_losses = []
    best_val = float("inf")
    patience_counter = 0
    
    for epoch in range(1,max_epochs+1):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train += 1

        train_losses.append(total_train_loss/n_train)

        model.eval()
        total_eval_loss = 0.0
        n_eval = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                preds = model(inputs)
                loss = criterion(preds, targets)
                total_eval_loss += loss.item()
                n_eval += 1

            val_losses.append(total_eval_loss/ n_eval)

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                return {"train_losses" : train_losses, "val_losses": val_losses, "stopped_epoch": epoch}

    return {"train_losses" : train_losses, "val_losses": val_losses, "stopped_epoch": epoch}
