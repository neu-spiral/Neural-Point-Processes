import torch
from random import randint
from tools.NPmodels import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence
import numpy as np
from torcheval.metrics.functional import r2_score

class NPLRFinder:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {'lr': [], 'loss': []}

    def find_lr(self, train_loader, input_channel, start_lr=1e-6, end_lr=1, num_iter=20,smooth_f=0.05):
        model = self.model.to(self.device)
        optimizer = self.optimizer
        device = self.device
        model.train()

        lr_step = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr

        for iteration in range(num_iter):
            optimizer.param_groups[0]['lr'] = lr

            total_loss = 0.0
            for batch in train_loader:
                x_context, y_context, x_target, y_target = process_batch(batch, self.device)
                optimizer.zero_grad()
                p_y_pred, q_target, q_context = model(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)

            lr *= lr_step
            
    def plot_lr_finder(self):
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')  # Use a logarithmic scale for better visualization
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder Curve')
        plt.show()
        
    def find_best_lr(self, skip_start=3, skip_end=3):
        # Find the index of the minimum loss in the specified range
        min_loss_index = skip_start + np.argmin(self.history['loss'][skip_start:-skip_end])
        # Output the learning rate corresponding to the minimum loss
        best_lr = self.history['lr'][min_loss_index]
        return best_lr
    
    def _loss(self, p_y_pred, y_target, q_target, q_context):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
    
    
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


def batch_r2(pred, target):
    # Calculate R2 score for each pair and store the results
    r2_scores = 0
    for i in range(len(pred)):
        if torch.allclose(target[i], target[i][0], atol=1e-10):
            if torch.allclose(pred[i], target[i][0], atol=1e-10):
                r2 = 1
            else:
                r2 = 0
        else:
            r2 = r2_score(pred[i], target[i])
        r2_scores += r2

    # Calculate the average R2 score
    avg_r2 = r2_scores / len(pred)
    return avg_r2


def process_batch(batch, device):
    pins = [tensor.to(device) for tensor in batch['pins']]
    outputs = [tensor.to(device) for tensor in batch['outputs']]
    # get np arrays from lists
    torch_pins = torch.dstack(pins).permute(2, 0, 1)  # stack into (batch, n_pins, 2)
    torch_outputs = torch.dstack(outputs).permute(2, 1, 0)  # stack into (batch, n_pins, 1)
    # get half context and half target for training
    num_context = len(pins[0]) // 2
    x_context = torch_pins[:, :num_context, :]
    y_context = torch_outputs[:, :num_context, :]
    x_target = torch_pins
    y_target = torch_outputs
    return x_context, y_context, x_target, y_target


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """

    def __init__(self, device, neural_process, optimizer, early_stopping, experiment_id, print_freq=10):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.experiment_id = experiment_id
        self.early_stopping = early_stopping
        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            total_loss = 0.
            for i, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                # Create context and target points and apply neural process
                x_context, y_context, x_target, y_target = process_batch(batch, self.device)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_dataloader)
            self.train_losses.append(total_loss)
            print("Epoch: {}, Avg_loss: {:.3f}".format(epoch, total_loss))

            if epoch % self.print_freq == 0:
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        x_context_val, y_context_val, x_target_val, y_target_val = process_batch(val_batch, self.device)
                        p_y_pred_val, q_target_val, q_context_val = \
                            self.neural_process(x_context_val, y_context_val, x_target_val, y_target_val)
                        loss_val = self._loss(p_y_pred, y_target, q_target, q_context)
                        val_loss += loss_val.item()
                val_loss /= len(val_dataloader)
                self.val_losses.append(val_loss)
                print("Epoch: {}, Val_loss {:.3f}".format(epoch, val_loss))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # Save the model
                    torch.save(self.neural_process.state_dict(), f'./history/{self.experiment_id}/model_NP.pth')
                if self.early_stopping(epoch, val_loss):
                    break  # Stop training early

        self.neural_process.load_state_dict(torch.load(f'./history/{self.experiment_id}/model_NP.pth'))

        # return self.neural_process, self.train_losses, self.val_losses

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl


def NP_prediction(model, x_context, y_context, x_target):
    model.eval()
    y2 = model(x_context, y_context, x_target, y_target=None)
    return y2


def evaluate_np(model, dataloader, device, partial_percent=0, hidden_samples=0.5):
    # Partial percent is the percentage of hidden labels you want to rebeal
    model.eval()
    total_loss = 0.0
    total_r2 = 0.0
    if partial_percent == 0:
        print("Error: partial_percent cannot be 0.")
        return None
    else:
        with torch.no_grad():
            for batch in dataloader:
                # convert batch into neural process input format
                x_context, y_context, x_target, y_target = process_batch(batch, device)
                # divide batch into test and partial label (hidden)
                total_hidden_samples = int(y_target.shape[1] * hidden_samples)
                revealed_samples = int(total_hidden_samples * partial_percent)
                # context data is the partial revealed label
                x_context = x_context[:, :revealed_samples, :]
                y_context = y_context[:, :revealed_samples, :]
                # target data includes both the revealed and the to-be-tested data
                x_target = torch.cat((x_target[:, :revealed_samples, :], x_target[:, total_hidden_samples:, :]), dim=1)
                y_true = y_target[:, total_hidden_samples:, :]
                NP_outputs = NP_prediction(model, x_context, y_context, x_target)
                y_pred = NP_outputs.mean[:, revealed_samples::]
                # per image per pin MSE loss
                mse_error = (y_true - y_pred)  # get the mean of predictions
                total_r2 += batch_r2(y_pred, y_true)
                total_loss += torch.sum((mse_error) ** 2) / (y_pred.shape[0] * y_pred.shape[1])
                
        total_r2 /= len(dataloader)
        total_loss /= len(dataloader)
        return total_loss, total_r2