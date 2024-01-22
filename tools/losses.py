import torch
import torch.nn as nn
from functools import lru_cache

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
    def __init__(self, identity, sigma=1.0):
        super(NPPLoss, self).__init__()
        self.identity = identity
        self.sigma = sigma  # Add sigma as an instance variable
    
    @lru_cache(maxsize=128)
    def compute_kernel(self, pins):
        matrix_list = []
        for i in range(len(pins)):
            X = Y = pins[i].float()
            kernel_matrix = gaussian_kernel_matrix(X, Y, self.sigma)  # Use self.sigma
            pseudo_inv_matrix = pseudo_inverse(kernel_matrix)
            matrix_list.append(pseudo_inv_matrix)
        return matrix_list
    
    def forward(self, y_true, y_pred, pins):
        loss = 0
        if self.identity:
            for i in range(len(y_true)):
                loss += 1/len(y_true[i]) * torch.matmul(
                    (y_true[i] - y_pred[i].squeeze()[pins[i][:,0], pins[i][:,1]]).t(),
                    (y_true[i] - y_pred[i].squeeze()[pins[i][:,0], pins[i][:,1]])
                )
        else:
            matrix_list = self.compute_kernel(tuple(pins))
            for i in range(len(y_true)):
                loss += 1/len(y_true[i]) * torch.matmul(
                    (y_true[i] - y_pred[i].squeeze()[pins[i][:,0], pins[i][:,1]]).t(),
                    torch.matmul(matrix_list[i], y_true[i] - y_pred[i].squeeze()[pins[i][:,0], pins[i][:,1]])
                )
            matrix_list = []
        loss /= len(y_true)
        return loss