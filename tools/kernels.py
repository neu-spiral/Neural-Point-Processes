import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import exit

class RBFKernel(nn.Module):
    def __init__(self, length_scale=1, kernel_mode="fixed"):
        super(RBFKernel, self).__init__()
        self.kernel_mode = kernel_mode
        
        # Compute the log bounds for length_scale in the range [0.1, 10]
        self.clip_min = torch.log(torch.tensor(1.0))
        self.clip_max = torch.log(torch.tensor(2.0))
        
        # Store the length scale in logarithmic form
        if self.kernel_mode == "learned":
            self.log_length_scale = nn.Parameter(torch.tensor(float(length_scale)).log())
        elif self.kernel_mode == "fixed":
            self.log_length_scale = torch.tensor(float(length_scale)).log()

    def forward(self, X, X_prime, kernel_params=None):
        if self.kernel_mode == "predicted":
            log_length_scale = torch.tensor(kernel_params)
            # print(f"kernel params: {kernel_params[:5]}")
        else:
            self.clip_min = self.clip_min.to(self.log_length_scale.device)
            self.clip_max = self.clip_max.to(self.log_length_scale.device)
            # Clip the log_length_scale to keep the length_scale within [0.1, 10]
            log_length_scale = torch.clamp(self.log_length_scale, self.clip_min, self.clip_max)
        
        # Convert back to length scale
        length_scale = torch.exp(log_length_scale)
        # print(f"ls: {length_scale}")
        # Compute the squared Euclidean distance
        dist_sq = torch.cdist(X, X_prime, p=2)**2
        
        # Compute the RBF kernel
        # print(dist_sq.shape, length_scale.shape)
        K = torch.exp(-dist_sq / (2.0 * length_scale**2))
        return K
    
    
class SMKernel(nn.Module):
    def __init__(self, num_mixture, D=2, params=None, kernel_mode="fixed"):
        """
        Initialize the MixtureKernel.

        Parameters:
        num_mixture (int): Number of mixture components.
        D (int): Dimensionality of the input space.
        kernel_mode (str): The mode of operation for the kernel ('fixed', 'learned', 'predicted').
        """
        super(SMKernel, self).__init__()
        self.num_mixture = num_mixture
        self.D = D
        self.kernel_mode = kernel_mode
        
        
        if kernel_mode == "fixed":
            # Initialize weights (w_q)
            self.log_weights = torch.zeros(num_mixture)

            # Initialize means (mu_q)
            self.means = torch.randn(num_mixture, D)

            # Initialize diagonal elements of covariance matrices (v_qd)
            self.log_covariances = torch.randn(num_mixture, D)
            
        elif self.kernel_mode == "learned":
            # Initialize weights (w_q)
            self.log_weights = nn.Parameter(torch.zeros(num_mixture))

            # Initialize means (mu_q)
            self.means = nn.Parameter(torch.randn(num_mixture, D))

            # Initialize diagonal elements of covariance matrices (v_qd)
            self.log_covariances = nn.Parameter(torch.randn(num_mixture, D))

    def forward(self, x, x_prime, kernel_params):
        device = x.device
        if self.kernel_mode == "learned" or self.kernel_mode == "fixed":
            log_weights = self.log_weights.to(device)
            means = self.means.to(device)
            log_covariances = self.log_covariances.to(device)

        elif self.kernel_mode == "predicted" and kernel_params is not None:
            log_weights, means, log_covariances = (
                kernel_params[:self.num_mixture].to(device), 
                (kernel_params[self.num_mixture:self.num_mixture*(self.D+1)]).reshape((-1, self.D)).to(device), 
                (kernel_params[self.num_mixture*(self.D+1):]).reshape((-1, self.D)).to(device)
            )
        #     print(kernel_params.shape)
        # print(self.num_mixture)
        # print(log_weights.shape, means.shape, log_covariances.shape)
        # print(log_covariances[0], log_covariances[0].shape)
        # exit(0)
            
        # Number of samples in the batch
        n = x.shape[0]
        m = x_prime.shape[0]

        # Reshape x and x_prime for broadcasting
        x = x.unsqueeze(1)  # Shape: (n, 1, D)
        x_prime = x_prime.unsqueeze(0)  # Shape: (1, m, D)

        # Compute pairwise differences
        diff = x - x_prime  # Shape: (n, m, D)
        
        kernel_matrix = torch.zeros((n, m), device=device)
        total_weights = sum(torch.exp(log_weights))
        for q in range(self.num_mixture):
            w_q = torch.exp(log_weights[q]).to(device)
            mu_q = means[q].to(device)
            sigma_q_diag = torch.exp(log_covariances[q]).to(device)
            Sigma_q = torch.diag(sigma_q_diag).to(device)
            
            # Compute determinant of Σ_q (product of diagonal elements)
            det_Sigma_q = torch.prod(sigma_q_diag)
            
            # Compute the normalization factor
            norm_factor = (det_Sigma_q**0.5) / ( (2 * torch.pi)**(self.D / 2))
            
            # Compute the exponent term
            exponent = -0.5 * torch.einsum('ijD,D,ijD->ij', diff, sigma_q_diag, diff)  # Shape: (n, n)
            
            # Compute the cosine term
            cosine_term = torch.cos(2 * torch.pi * torch.einsum('ijD,D->ij', diff, mu_q))  # Shape: (n, n)
            
            # Add the weighted component to the kernel matrix
            kernel_matrix += (w_q/total_weights) * norm_factor * torch.exp(exponent) * cosine_term
        return kernel_matrix