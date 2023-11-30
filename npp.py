import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage import io, transform
from tools.plot_utils import visualize_pins, plot_label_pin, plot_all, plot_and_save, plot_loss
from tools.data_utils import *
from tools.losses import NPPLoss
from tools.models import Autoencoder
from tools.optimization import EarlyStoppingCallback, train_model, evaluate_model
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder

    
def run_pipeline_ci(sigmas, num_kernels_encoder, num_kernels_decoder, train_dataloader, val_dataloader, test_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, device, num_runs=1):
    test_losses_npp_true = []
    test_losses_npp_false= []

    for run in range(num_runs):
        test_losses_vs_sigma_npp_true = []
        test_loss_npp_false = None

        # Run NPP=False once and collect the test loss
        early_stopping = EarlyStoppingCallback(patience=5, min_delta=0.001)
        criterion = NPPLoss(identity=True).to(device)

        autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)
        
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        model, train_losses, val_losses = train_model(autoencoder, train_dataloader, val_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, criterion, optimizer, device, early_stopping)

        test_loss_npp_false = evaluate_model(autoencoder, test_dataloader, input_channel, device)
        print(f"Test loss npp_f:{test_loss_npp_false}")
        test_losses_npp_false.append(test_loss_npp_false)

        # Run LR Finder for different sigma values
        for sigma in sigmas:
            early_stopping = EarlyStoppingCallback(patience=5, min_delta=0.001)
            criterion = NPPLoss(identity=False, sigma=sigma).to(device)
            autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)
            optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
            model, train_losses, val_losses = train_model(autoencoder, train_dataloader, val_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, criterion, optimizer, device, early_stopping)
            test_loss = evaluate_model(autoencoder, test_dataloader, input_channel, device)
            print(f"Test loss npp_t:{test_loss}")
            test_losses_vs_sigma_npp_true.append(test_loss)

        test_losses_npp_true.append(test_losses_vs_sigma_npp_true)
    return test_losses_npp_true, test_losses_npp_false

    
# Function to run the pipeline and save data
def run_and_save_pipeline(sigmas, num_kernels_encoder, num_kernels_decoder, train_dataloader, val_dataloader, test_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, device):
    # Run the pipeline
    test_loss_npp_true, test_loss_npp_false= run_pipeline_ci(sigmas, num_kernels_encoder, num_kernels_decoder, train_dataloader, val_dataloader, test_dataloader, input_channel, num_epochs, val_every_epoch, learning_rate, device)
    print("start saving!")
    # Save the data
    save_data(test_loss_npp_true, './history/test_loss_npp_true.npy')
    save_data(test_loss_npp_false, './history/test_loss_npp_false.npy')
    print("saved")
    return test_loss_npp_true, test_loss_npp_false


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="Synthetic", help="Dataset name")
    parser.add_argument("--n", type=int, default=100, help="Value for 'n'")
    parser.add_argument("--mesh", type=bool, default=True, help="Value for 'mesh'")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--fixed_pins", type=bool, default=True, help="Value for 'fixed_pins'")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")
    parser.add_argument("--d1", type=int, default=28, help="Value for 'd1'")
    parser.add_argument("--d2", type=int, default=28, help="Value for 'd2'")

    # Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    # List of sigma values
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 0.2, 0.5, 1, 2, 5], help="List of sigma values to test")

    # Kernel sizes
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[16, 8], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[16], help="List of decoder kernel sizes")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set a random seed for PyTorch
    seed = 4  # You can use any integer value as the seed
    torch.manual_seed(seed)
    # Set a random seed for NumPy (if you're using NumPy operations)
    np.random.seed(seed)

    # Choose datasets
    dataset = args.dataset 
    n = args.n
    mesh = args.mesh
    d = args.d
    n_pins = args.n_pins
    fixed_pins = args.fixed_pins
    r = args.r
    d1,d2 = args.d1, args.d2

    # Set your hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    sigmas = args.sigmas  # Set the sigma values you want to test
    num_kernels_encoder = args.num_encoder
    num_kernels_decoder = args.num_decoder
    learning_rate = args.learning_rate
    
    input_channel = 1 if dataset == "MNIST" else 3
    val_every_epoch = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data
    if dataset == "PinMNIST":
        if mesh:
            data_folder = f"./data/MNIST_{n}images_mesh_{d}step_{28}by{28}pixels_{r}radius_{seed}seed"
        else:
            data_folder = f"./data/MNIST_{n}images_random_fixed{fixed_pins}_{n_pins}pins_{28}by{28}pixels_{r}radius_{seed}seed"

    if dataset == "Synthetic":
        if mesh:
            data_folder = f"./data/Synthetic_{n}images_{d1}by{d2}pixels_{d}_distanced_grid_pins_{seed}seed/"
        else:
            data_folder = f"./data/Synthetic_{n}images_{d1}by{d2}pixels_upto{n_pins}pins_{seed}seed/"


    # Create a transform pipeline that includes the resizing step
    transform = transforms.Compose([
        ToTensor(),         # Convert to tensor (as you were doing)
        Resize()  # Resize to 100x100
    ])

    transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                          root_dir=f"{data_folder}/images/",
                                          transform=transform)

    dataset_size = len(transformed_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        transformed_dataset, [train_size, val_size, test_size])

    # Create your DataLoader with the custom_collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Run and save the pipeline data
    loss_vs_sigma_data = run_and_save_pipeline(sigmas, num_kernels_encoder, num_kernels_decoder, train_dataloader, val_dataloader, test_dataloader,\
                                               input_channel, num_epochs, val_every_epoch, learning_rate, device)


    # Plot and save the plot using the saved data
    plot_and_save(loss_vs_sigma_data, sigmas, dataset, learning_rate)
    # plot_loss(train_losses, val_losses, val_every_epoch, NPP, sigma, dataset, learning_rate, num_kernels_encoder, num_kernels_decoder) # used to look into a single sigma


if __name__ == "__main__":
    main()