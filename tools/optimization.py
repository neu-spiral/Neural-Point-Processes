import torch
from tools.losses import NPPLoss, gaussian_kernel_matrix
# from sklearn.metrics import r2_score
from torcheval.metrics.functional import r2_score
import torch.optim.lr_scheduler as lr_scheduler
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


def train_model(model, train_dataloader, val_dataloader, input_channel, epochs, val_every_epoch, learning_rate,
                criterion, optimizer, device, early_stopping, experiment_id, exp_name, global_best_val_loss, manual_lr, sigma=0):
    train_losses = []  # To track train loss for plotting
    val_losses = []  # To track validation loss for plotting
    best_val_loss = float('inf')
    if manual_lr:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)
        current_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(epochs):
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
        if manual_lr:
            scheduler.step(total_loss)
            if (current_lr != optimizer.param_groups[0]["lr"]):
                current_lr = optimizer.param_groups[0]["lr"]
                print("New LR: ", current_lr)
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
                if (sigma == 0):
                    torch.save(model.state_dict(), f'./history{exp_name}/{experiment_id}/current_model_MSE.pth')
                else:
                    torch.save(model.state_dict(), f'./history{exp_name}/{experiment_id}/current_model_NPP.pth')
            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                # Save the model
                if (sigma == 0):
                    torch.save(model.state_dict(), f'./history{exp_name}/{experiment_id}/best_model_MSE.pth')
                else:
                    torch.save(model.state_dict(), f'./history{exp_name}/{experiment_id}/best_model_NPP.pth')

            if early_stopping(epoch, val_loss):
                break  # Stop training early
            print(f'Validation Loss: {val_loss:.4f}')
            val_losses.append(val_loss)

    # Reload the best model after training
    if (sigma == 0):
        model.load_state_dict(torch.load(f'./history{exp_name}/{experiment_id}/current_model_MSE.pth'))
    else:
        model.load_state_dict(torch.load(f'./history{exp_name}/{experiment_id}/current_model_NPP.pth'))

    return model, train_losses, val_losses, global_best_val_loss


def GP_prediction(x1, y1, mu1, x2, mu2, kernel_func, sigma, noise=1e-5):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input x2, the observations (y1, x1), 
    and the prior kernel function.
    x1 x2 shape: nx2
    y mu shape: n,
    """
    x1 = x1.float()
    x2 = x2.float()
    # Kernel of the observations
    Cov11 = kernel_func(x1, x1, sigma) + (noise*torch.eye(len(x1))).to(x1.device)
    # Kernel of observations vs to-predict
    Cov12 = kernel_func(x1, x2, sigma)
    Cov22 = kernel_func(x2, x2, sigma)

    solved = torch.linalg.solve(Cov11, Cov12).T
    # Compute posterior mean
    mu2_new = solved @ (y1 - mu1) + mu2

    # Compute the posterior covariance
    Cov22_new = Cov22 - (solved @ Cov12)
    return mu2_new, Cov22_new


def NP_prediction(NP_model, x1, y1, x2):
    y2 = NP_model(x1, y1, x2)
    return y2


def evaluate_model(model, dataloader, input_channel, device, sigma=1, partial_label_GP=False, partial_percent=0,
                   kernel_func=gaussian_kernel_matrix, hidden_samples=0.5):
    # Partial percent is the percentage of hidden labels you want to reveal
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    criterion = NPPLoss(identity=True).to(device)

    with torch.no_grad():
        for batch in dataloader:
            x_test = batch['image'][:, :input_channel, :, :].to(device)  # batch of image tensors (batch * ch * h * w)
            p_test = [tensor.to(device) for tensor in batch['pins']]  # list of pin tensors
            y_test = [tensor.to(device) for tensor in batch['outputs']]  # list of output tensors
            test_outputs = model(x_test.float())  # batch of 2D predictions [batch, 1, h, w]
            r2 = 0
            for i in range(len(x_test)):  # iterates over each image on the batch
                num_samples = int(len(p_test[i]) * hidden_samples)  # test on half of the samples
                p_sample = p_test[i][num_samples:]
                y_sample = y_test[i][num_samples:]

                # below we extract the outputs on test_outputs by pin locations
                mu_sample = (test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]]

                if partial_percent > 0 and partial_label_GP:
                    reveal_samples = int(num_samples * partial_percent)
                    p_ref = p_test[i][:reveal_samples]
                    y_ref = y_test[i][:reveal_samples]
                    mu_ref = (test_outputs[i].squeeze())[p_ref[:, 0], p_ref[:, 1]]
                    test_outputs_new, cov = GP_prediction(p_ref, y_ref, mu_ref, p_sample, mu_sample,
                                                          gaussian_kernel_matrix, sigma)
                    test_outputs[i][:, p_sample[:, 0], p_sample[:, 1]] = test_outputs_new

                y_test[i] = y_sample
                p_test[i] = p_sample
                if torch.allclose(y_test[i], torch.zeros_like(y_test[i]), atol=1e-10):
                    # If target is constant r2 should return 0.0 if pred is different or 1.0 if pred == target
                    if torch.all(torch.eq((test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]], y_test[i])):
                        r2 += 1.0
                else:
                    r2 += r2_score((test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]], y_test[i]).item()
            loss = criterion(y_test, test_outputs, p_test)
            total_loss += loss.item()
            total_r2 += r2/len(x_test)

    total_loss /= len(dataloader)
    total_r2 /= len(dataloader)
    return total_loss, total_r2