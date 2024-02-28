import torch
from random import randint
from tools.NPmodels import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence


def process_batch(batch):
    x_train = batch['image'][:, :input_channel, :, :].to(self.device)
    p_train = [tensor.to(self.device) for tensor in batch['pins']]
    y_train = [tensor.to(self.device) for tensor in batch['outputs']] 
    # get np arrays from lists
    np_pins = np.dstack(pins).transpose((2, 0, 1))
    np_outputs = np.dstack(outputs).transpose((2, 1, 0))
    # get half context and half target for training
    num_context = len(x_train)//2
    x_context = np_pins[:,:num_context, :]
    y_context = np_outputs[:, :num_context, :]
    x_target = np_pins
    y_target = np_outputs
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
    def __init__(self, device, neural_process, optimizer, experiment_id, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.experiment_id = experiment_id

        # Check if neural process is for images
        self.is_img = isinstance(self.neural_process, NeuralProcessImg)
        self.train_losses = []
        self.val_losses = []

    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        best_val_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0.
            for i, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                # Create context and target points and apply neural process
                x_context, y_context, x_target, y_target = process_batch(batch)
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
                        x_context_val, y_context_val, x_target_val, y_target_val = process_batch(val_batch)
                        p_y_pred_val, q_target_val, q_context_val = \
                            self.neural_process(x_context_val, y_context_val, x_target_val, y_target_val)
                        loss_val = self._loss(p_y_pred, y_target, q_target, q_context)
                        val_loss += loss_val.item()
                val_loss /= len(val_dataloader)
                self.val_losses.append(val_loss)
                print("Epoch: {}, Val_loss {:.3f}".format(epoch, val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save the model
                    torch.save(model.state_dict(), f'./history/{self.experiment_id}/model_NP.pth')
                if early_stopping(epoch, val_loss):
                    break  # Stop training early

        self.neural_process.load_state_dict(torch.load(f'./history/{self.experiment_id}/model_NP.pth'))

    return model, train_losses, val_losses, global_best_val_loss

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