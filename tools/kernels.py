import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKernel(nn.Module):
    def __init__(self, D, length_scale=1.0):
        """
        Initialize the RBFKernel.

        Parameters:
        D (int): Dimensionality of the input space.
        length_scale (float): Initial length scale parameter.
        """
        super(RBFKernel, self).__init__()
        self.D = D
        self.log_length_scale = nn.Parameter(torch.tensor(float(length_scale)).log())

    def forward(self, X, X_prime):
        """
        Compute the RBF kernel matrix.

        Parameters:
        X (torch.Tensor): Input matrix of shape (n, D).
        X_prime (torch.Tensor): Input matrix of shape (m, D).

        Returns:
        torch.Tensor: Kernel matrix of shape (n, m).
        """
        length_scale = torch.exp(self.log_length_scale)
        
        # Compute the squared Euclidean distance
        X_norm = X.pow(2).sum(1).view(-1, 1)
        X_prime_norm = X_prime.pow(2).sum(1).view(1, -1)
        dist_sq = X_norm + X_prime_norm - 2.0 * torch.mm(X, X_prime.t())
        
        # Compute the RBF kernel matrix
        K = torch.exp(-dist_sq / (2.0 * length_scale**2))
        
        return K
    
    
class SMKernel(nn.Module):
    def __init__(self, Q, D):
        """
        Initialize the MixtureKernel.

        Parameters:
        Q (int): Number of mixture components.
        D (int): Dimensionality of the input space.
        """
        super(Batch_SMKernel, self).__init__()
        self.Q = Q
        self.D = D
        
        # Initialize weights (w_q)
        self.weights = nn.Parameter(torch.ones(Q) / Q)
        
        # Initialize means (mu_q)
        self.means = nn.Parameter(torch.randn(Q, D))
        
        # Initialize diagonal elements of covariance matrices (v_qd)
        self.log_covariances = nn.Parameter(torch.randn(Q, D))

    def forward(self, x, x_prime):
        """
        Compute the kernel matrix for batched inputs.

        Parameters:
        x (torch.Tensor): Input tensor of shape (n, D)
        x_prime (torch.Tensor): Input tensor of shape (n, D)

        Returns:
        torch.Tensor: The kernel matrix of shape (n, n)
        """
        # Number of samples in the batch
        n = x.shape[0]

        # Reshape x and x_prime for broadcasting
        x = x.unsqueeze(1)  # Shape: (n, 1, D)
        x_prime = x_prime.unsqueeze(0)  # Shape: (1, n, D)

        # Compute pairwise differences
        diff = x - x_prime  # Shape: (n, n, D)
        
        kernel_matrix = torch.zeros(n, n)

        for q in range(self.Q):
            w_q = torch.exp(self.weights[q])
            mu_q = self.means[q]
            sigma_q_diag = torch.exp(self.log_covariances[q])
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
            kernel_matrix += w_q * norm_factor * torch.exp(exponent) * cosine_term

        return kernel_matrix