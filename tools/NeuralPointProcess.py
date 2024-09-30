import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score
from tools.models import Autoencoder
from tools.kernels import RBFKernel, SMKernel
from tools.losses import NPPLoss
from tools.optimization import EarlyStoppingCallback


class NeuralPointProcesses(nn.Module):
    def __init__(self, identity=False, kernel="RBF", kernel_mode="fixed", D=2, num_encoder=[64, 32], num_decoder=[64], input_shape=32, input_channel=3, deeper=False, kernel_param=1, lr=0.0001, noise=1e-4, device='cuda'):
        super(NeuralPointProcesses, self).__init__()
        self.identity = identity
        self.kernel_mode = kernel_mode
        self.input_channel = input_channel
        self.device = device
        self.noise = noise
        # Initialize the appropriate kernel
        if kernel == "RBF":
            self.kernel = RBFKernel(length_scale=kernel_param, kernel_mode=self.kernel_mode)
            param_size = 1
        elif kernel == "SM":
            D = 2 # the x and y axes
            num_mixture=int(kernel_param)
            self.kernel = SMKernel(num_mixture, D=D, kernel_mode=self.kernel_mode)
            param_size = num_mixture * (2 * D + 1)
        
        # Initialize the Autoencoder model
        if self.kernel_mode == "predicted":
            self.model = Autoencoder(num_encoder, num_decoder, input_shape, input_channel, deeper, param_size).to(device)  
        else:
            self.model = Autoencoder(num_encoder, num_decoder, input_shape, input_channel, deeper).to(device)
        self.criterion = NPPLoss(identity=identity, kernel=self.kernel, noise=noise).to(device)
        
        if kernel_mode == "learned":
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.kernel.parameters()), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
    
    def train_model(self, train_dataloader, val_dataloader, epochs, experiment_id, exp_name, global_best_val_loss, val_every_epoch, early_stopping=True):
        train_losses = []
        val_losses = []

        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)
        current_lr = self.optimizer.param_groups[0]["lr"]
        early_stopping = EarlyStoppingCallback(patience=15, min_delta=0.001)
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                x_train = batch['image'][:, :self.input_channel, :, :].to(self.device)
                p_train = [tensor.to(self.device) for tensor in batch['pins']]
                y_train = [tensor.to(self.device) for tensor in batch['outputs']]

                outputs, kernel_param = self.model(x_train.float())
                loss = self.criterion(y_train, outputs, p_train, kernel_param)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_dataloader)
            scheduler.step(total_loss)
            if (current_lr != self.optimizer.param_groups[0]["lr"]):
                current_lr = self.optimizer.param_groups[0]["lr"]
                print("New LR: ", current_lr)
            train_losses.append(total_loss)

            if (epoch) % val_every_epoch == 0:
                val_loss = self._validate(val_dataloader)
                if not os.path.exists(f'./history/{exp_name}/{experiment_id}'):
                    os.makedirs(f'./history/{exp_name}/{experiment_id}')

                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    self._save_model(experiment_id, exp_name)
                    print(f"new model saved at {epoch} epoch with global_best_val_loss: {global_best_val_loss}")

                if early_stopping(epoch, val_loss):
                    break
                print(f'Validation Loss: {val_loss:.4f}')
                val_losses.append(val_loss)

        self.load_best_model(experiment_id, exp_name)
        return train_losses, val_losses, global_best_val_loss

    def _validate(self, val_dataloader):
        val_criterion = NPPLoss(identity=True, kernel=self.kernel, noise=self.noise).to(self.device)
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                x_val = val_batch['image'][:, :self.input_channel, :, :].to(self.device)
                p_val = [tensor.to(self.device) for tensor in val_batch['pins']]
                y_val = [tensor.to(self.device) for tensor in val_batch['outputs']]

                val_outputs, kernel_param = self.model(x_val.float())
                val_loss += val_criterion(y_val, val_outputs, p_val, kernel_param).item()

        return val_loss / len(val_dataloader)

    def _save_model(self, experiment_id, exp_name):
        if self.identity:
            loss = "MSE"
        else:
            loss = "NPP"
            
        model_file_name = f'./history/{exp_name}/{experiment_id}/best_model_{loss}.pth'
        torch.save(self.model.state_dict(), model_file_name)
        
        if self.kernel_mode == "learned":
            kernel_file_name = f'./history/{exp_name}/{experiment_id}/best_kernel_{loss}.pth'
            torch.save(self.kernel.state_dict(), kernel_file_name)

    def load_best_model(self, experiment_id, exp_name):
        if self.identity:
            loss = "MSE"
        else:
            loss = "NPP"
        model_file_name = f'./history/{exp_name}/{experiment_id}/best_model_{loss}.pth'
        self.model.load_state_dict(torch.load(model_file_name))
        
        if self.kernel_mode == "learned":
            kernel_file_name = f'./history/{exp_name}/{experiment_id}/best_kernel_{loss}.pth'
            self.kernel.load_state_dict(torch.load(kernel_file_name))

    def evaluate_model(self, dataloader, partial_percent=0, hidden_samples=0.5):
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        test_criterion = NPPLoss(identity=True, kernel=self.kernel, noise=self.noise).to(self.device)
        with torch.no_grad():
            for batch in dataloader:
                x_test = batch['image'][:, :self.input_channel, :, :].to(self.device)
                p_test = [tensor.to(self.device) for tensor in batch['pins']]
                y_test = [tensor.to(self.device) for tensor in batch['outputs']]
                test_outputs, kernel_param = self.model(x_test.float())
                r2 = 0
                for i in range(len(x_test)):
                    num_samples = int(len(p_test[i]) * hidden_samples)
                    p_sample = p_test[i][num_samples:]
                    y_sample = y_test[i][num_samples:]

                    mu_sample = (test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]]

                    if partial_percent > 0:
                        reveal_samples = int(num_samples * partial_percent)
                        p_ref = p_test[i][:reveal_samples]
                        y_ref = y_test[i][:reveal_samples]
                        mu_ref = (test_outputs[i].squeeze())[p_ref[:, 0], p_ref[:, 1]]
                        kernel_input = kernel_param[i] if kernel_param is not None else None     
                        test_outputs_new, cov = self.GP_prediction(p_ref, y_ref, mu_ref, p_sample, mu_sample, kernel_input)
                        test_outputs[i][:, p_sample[:, 0], p_sample[:, 1]] = test_outputs_new

                    y_test[i] = y_sample
                    p_test[i] = p_sample
                    if torch.allclose(y_test[i], y_test[i][0], atol=1e-10):
                        if torch.all(torch.eq((test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]], y_test[i])):
                            r2 += 1.0
                    else:
                        r2 += r2_score((test_outputs[i].squeeze())[p_sample[:, 0], p_sample[:, 1]].cpu().numpy(), y_test[i].cpu().numpy()).item()
                loss = test_criterion(y_test, test_outputs, p_test, kernel_param)
                total_loss += loss.item()
                total_r2 += r2 / len(x_test)

        total_loss /= len(dataloader)
        total_r2 /= len(dataloader)
        return total_loss, total_r2
    
    def GP_prediction(self, x1, y1, mu1, x2, mu2, kernel_param):
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
        Cov11 = self.kernel(x1, x1, kernel_param) + (self.noise*torch.eye(len(x1))).to(x1.device)
        # Kernel of observations vs to-predict
        Cov12 = self.kernel(x1, x2, kernel_param)
        Cov22 = self.kernel(x2, x2, kernel_param)

        solved = torch.linalg.solve(Cov11, Cov12).T
        # Compute posterior mean
        # print(f'c11:{Cov11.shape}, c12:{Cov12.shape}, c22:{Cov22.shape}, solved:{solved.shape}')
        mu2_new = solved @ (y1 - mu1) + mu2

        # Compute the posterior covariance
        Cov22_new = Cov22 - (solved @ Cov12)
        return mu2_new, Cov22_new
