import torch
from tools.losses import NPPLoss

class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')

    def __call__(self, epoch, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping after {epoch} epochs.")
                return True  # Stop training
        return False  # Continue training
    
    
def train_model(model, train_dataloader, val_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, criterion, optimizer, device, early_stopping):
    train_losses = []  # To track train loss for plotting
    val_losses = []    # To track validation loss for plotting
    best_val_loss = float('inf') 

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            x_train = batch['image'][:, :input_channel, :, :].to(device)
            p_train = [tensor.to(device) for tensor in batch['pins']]
            y_train = [tensor.to(device) for tensor in batch['outputs']] 

            # Forward pass
            outputs = model(x_train.float())
            loss = criterion(y_train, outputs, p_train)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_dataloader.dataset)
        # Print train loss
        train_losses.append(total_loss)

        # Check validation loss every val_every_epoch epochs
        if (epoch) % val_every_epoch == 0:
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    x_val = val_batch['image'][:, :input_channel, :, :].to(device)
                    p_val = [tensor.to(device) for tensor in val_batch['pins']]
                    y_val = [tensor.to(device) for tensor in val_batch['outputs']]

                    val_outputs = model(x_val.float())
                    val_loss += criterion(y_val, val_outputs, p_val).item()

            val_loss /= len(val_dataloader.dataset)  # Average validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model
                torch.save(model.state_dict(), './history/best_model.pth')

            if early_stopping(epoch, val_loss):
                break  # Stop training early
            print(f'Validation Loss: {val_loss:.4f}')
            val_losses.append(val_loss)

    # Reload the best model after training
    model.load_state_dict(torch.load('./history/best_model.pth'))

    return model, train_losses, val_losses


def evaluate_model(model, dataloader, input_channel, device):
    model.eval()
    total_loss = 0.0
    criterion = NPPLoss(identity=True).to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            x_test = batch['image'][:, :input_channel, :, :].to(device)
            p_test = [tensor.to(device) for tensor in batch['pins']]
            y_test = [tensor.to(device) for tensor in batch['outputs']]

            test_outputs = model(x_test.float())
            loss = criterion(y_test, test_outputs, p_test)

            total_loss += loss.item()

    total_loss /= len(dataloader.dataset)
    return total_loss