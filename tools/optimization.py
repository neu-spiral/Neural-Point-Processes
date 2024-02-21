import torch
from tools.losses import NPPLoss, gaussian_kernel_matrix
import scipy

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
    
    
def train_model(model, train_dataloader, val_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, criterion, optimizer, device, early_stopping, experiment_id):
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
        total_loss /= len(train_dataloader)
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

            val_loss /= len(val_dataloader)  # Average validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model
                torch.save(model.state_dict(), f'./history/best_model{experiment_id}.pth')

            if early_stopping(epoch, val_loss):
                break  # Stop training early
            print(f'Validation Loss: {val_loss:.4f}')
            val_losses.append(val_loss)

    # Reload the best model after training
    model.load_state_dict(torch.load(f'./history/best_model{experiment_id}.pth'))

    return model, train_losses, val_losses


def GP_prediction(P1, y1, mu1, P2, mu2, kernel_func, sigma):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input P2, the observations (y1, P1), 
    and the prior kernel function.
    P1 P2 shape: nx2
    y mu shape: n,
    """
    P1 = P1.float()
    P2 = P2.float()
    # Kernel of the observations
    Cov11 = kernel_func(P1, P1, sigma)
    # Kernel of observations vs to-predict
    Cov12 = kernel_func(P1, P2, sigma)
    Cov22 = kernel_func(P2, P2, sigma)
    # Solve

    solved = torch.linalg.solve(Cov11, Cov12).T
    # Compute posterior mean
    mu2_new = solved @ (y1-mu1)+mu2
    
    # Compute the posterior covariance
    Cov22_new = Cov22 - (solved @ Cov12)
    return mu2_new, Cov22_new


def evaluate_model(model, dataloader, input_channel, device, sigma=1, partial_label_GP=False, partial_percent=0, kernel_func=gaussian_kernel_matrix):
    model.eval()
    total_loss = 0.0
    criterion = NPPLoss(identity=True).to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            x_test = batch['image'][:, :input_channel, :, :].to(device)
            p_test = [tensor.to(device) for tensor in batch['pins']]
            y_test = [tensor.to(device) for tensor in batch['outputs']]
            test_outputs = model(x_test.float())
            
            if partial_percent > 0 and partial_percent < 1:
                # Determine the number of samples to keep based on partial_percent
                
                for i in range(len(x_test)):            
                    num_samples = int(len(p_test[i]) * partial_percent)
                    x_sample = x_test[i]
                    p_sample = p_test[i][num_samples:]
                    y_sample = y_test[i][num_samples:]
                    mu_sample = (test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]]
                        
                    if partial_label_GP:
                        x_ref = x_test[i]
                        p_ref = p_test[i][:num_samples]
                        y_ref = y_test[i][:num_samples]
                        mu_ref = (test_outputs[i].squeeze())[p_ref[:, 0], p_ref[:, 1]]
                        test_outputs_new, cov = GP_prediction(p_ref, y_ref, mu_ref, p_sample, mu_sample, gaussian_kernel_matrix, sigma)
                        test_outputs[i][:, p_sample[:, 0], p_sample[:, 1]] = test_outputs_new
                        
                    y_test[i] = y_sample
                    p_test[i] = p_sample
                        
            
            loss = criterion(y_test, test_outputs, p_test)

            total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss
