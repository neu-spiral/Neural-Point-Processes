import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKernel(nn.Module):
    def __init__(self, length_scale=1, kernel_mode="fixed"):
        super(RBFKernel, self).__init__()
        self.kernel_mode = kernel_mode
        
        # Store the length scale in logarithmic form
        if self.kernel_mode == "learned":
            self.log_length_scale = nn.Parameter(torch.tensor(float(length_scale)).log())
        elif self.kernel_mode == "fixed":
            
            self.log_length_scale = torch.tensor(float(length_scale)).log()

    def forward(self, X, X_prime, kernel_params=None):
        if self.kernel_mode == "predicted":
            log_length_scale = torch.tensor(kernel_params)
        else:
            log_length_scale = self.log_length_scale

        length_scale = torch.exp(log_length_scale)
        dist_sq = torch.cdist(X, X_prime, p=2)**2
        K = torch.exp(-dist_sq / (2.0 * length_scale**2))
        return K
    
    
class SMKernel(nn.Module):
    def __init__(self, Q, D=2, params=None, kernel_mode="fixed"):
        """
        Initialize the MixtureKernel.

        Parameters:
        Q (int): Number of mixture components.
        D (int): Dimensionality of the input space.
        kernel_mode (str): The mode of operation for the kernel ('fixed', 'learned', 'predicted').
        """
        super(Batch_SMKernel, self).__init__()
        self.Q = Q
        self.D = D
        self.kernel_mode = kernel_mode
        
        
        if kernel_mode == "fixed":
            # Initialize weights (w_q)
            self.log_weights = torch.zeros(Q)

            # Initialize means (mu_q)
            self.means = torch.randn(Q, D)

            # Initialize diagonal elements of covariance matrices (v_qd)
            self.log_covariances = torch.randn(Q, D)
            
        elif self.kernel_mode == "learned":
            # Initialize weights (w_q)
            self.log_weights = nn.Parameter(torch.zeros(Q))

            # Initialize means (mu_q)
            self.means = nn.Parameter(torch.randn(Q, D))

            # Initialize diagonal elements of covariance matrices (v_qd)
            self.log_covariances = nn.Parameter(torch.randn(Q, D))

    def forward(self, x, x_prime, kernel_params):
        if self.kernel_mode == "learned" or self.kernel_mode == "fixed":
            log_weights = self.log_weights
            means = self.means
            log_covariances = self.log_covariances

        elif self.kernel_mode == "predicted":
            log_weights, means, log_covariances = kernel_params
            
        # Number of samples in the batch
        n = x.shape[0]

        # Reshape x and x_prime for broadcasting
        x = x.unsqueeze(1)  # Shape: (n, 1, D)
        x_prime = x_prime.unsqueeze(0)  # Shape: (1, n, D)

        # Compute pairwise differences
        diff = x - x_prime  # Shape: (n, n, D)
        
        kernel_matrix = torch.zeros(n, n)
        total_weights = sum(torch.exp(log_weights))
        for q in range(self.Q):
            w_q = torch.exp(log_weights[q])
            mu_q = self.means[q]
            sigma_q_diag = torch.exp(log_covariances[q])
            Sigma_q = torch.diag(sigma_q_diag)
            
            # Compute determinant of Σ_q (product of diagonal elements)
            det_Sigma_q = torch.prod(sigma_q_diag)
            
            # Compute the normalization factor
            norm_factor = 1 / (det_Sigma_q**0.5 * (2 * torch.pi)**(self.D / 2))
            
            # Compute the exponent term
            exponent = -0.5 * torch.einsum('ijD,D,ijD->ij', diff, sigma_q_diag, diff)  # Shape: (n, n)
            
            # Compute the cosine term
            cosine_term = torch.cos(2 * torch.pi * torch.einsum('ijD,D->ij', diff, mu_q))  # Shape: (n, n)
            
            # Add the weighted component to the kernel matrix
            kernel_matrix += w_q/total_weights * norm_factor * torch.exp(exponent) * cosine_term

        return kernel_matrix