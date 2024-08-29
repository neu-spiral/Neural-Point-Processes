import torch
import torch.nn as nn
from functools import lru_cache
from tools.kernels import RBFKernel, SMKernel

def gaussian_kernel_matrix(X, Y, sigma):
    """
    Calculate the Gaussian kernel matrix between two sets of PyTorch tensors X and Y.

    Parameters:
    X (torch.Tensor): First set of tensors with shape (n, d), where n is the number of vectors and d is the dimensionality.
    Y (torch.Tensor): Second set of tensors with shape (m, d), where m is the number of vectors and d is the dimensionality.
    sigma (float): The kernel bandwidth parameter.

    Returns:
    torch.Tensor: The Gaussian kernel matrix of shape (n, m).
    """
    if X.size(1) != Y.size(1):
        raise ValueError("Input tensors must have the same dimension")

    n, m = X.size(0), Y.size(0)
    X = X.unsqueeze(1)  # Shape (n, 1, d)
    Y = Y.unsqueeze(0)  # Shape (1, m, d)

    diff = torch.norm(X - Y, dim=2)  # Pairwise Euclidean distances between vectors
    return torch.exp(- (diff ** 2) / (2 * sigma ** 2))


def pseudo_inverse(kernel_matrix, epsilon=1e-5):
    """
    Calculate the pseudo-inverse of a matrix using Singular Value Decomposition (SVD).

    Parameters:
    kernel_matrix (torch.Tensor): The matrix for which to compute the pseudo-inverse.
    epsilon (float): A small value to avoid division by zero.

    Returns:
    torch.Tensor: The pseudo-inverse of the input matrix.
    """
    U, S, V = torch.svd(kernel_matrix)
    S_inv = 1.0 / (S + epsilon)
    pseudo_inv = torch.mm(V, torch.mm(torch.diag(S_inv), U.t()))
    return pseudo_inv


class NPPLoss(nn.Module):
    def __init__(self, identity, kernel, noise=0):
        super(NPPLoss, self).__init__()
        self.identity = identity
        self.kernel = kernel
        self.noise = noise
    
    def forward(self, y_true, y_pred, pins):       
        loss = 0
        if self.identity:
            for i in range(len(y_true)):
                loss += 1 / len(y_true[i]) * torch.matmul(
                    (y_true[i] - y_pred[i].squeeze()[pins[i][:, 0], pins[i][:, 1]]).t(),
                    (y_true[i] - y_pred[i].squeeze()[pins[i][:, 0], pins[i][:, 1]]))
                
        else:
            for i in range(len(y_true)):
                diff = y_true[i] - y_pred[i].squeeze()[pins[i][:, 0], pins[i][:, 1]]
                kernel_matrix = self.kernel(pins[i].float(), pins[i].float(), kernel_params[i] if kernel_params is not None else None)
                noise_kernel = kernel_matrix+ self.noise * torch.eye(len(X), device=X.device)
                pseudo_inv = pseudo_inverse(noise_kernel) 
                loss += 1 / len(y_true[i]) * (torch.matmul(diff.t(), torch.matmul(pseudo_inv, diff))+torch.linalg.logdet(pseudo_inv))

        loss /= len(y_true)
        return loss