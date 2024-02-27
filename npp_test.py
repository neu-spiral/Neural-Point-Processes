import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tools.plot_utils import plot_and_save
from tools.data_utils import *
from tools.losses import NPPLoss
from tools.models import Autoencoder
from tools.optimization import EarlyStoppingCallback, evaluate_model
import matplotlib.pyplot as plt
import argparse
import time
from tools.models import *

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="PinMNIST", help="Dataset name")
    parser.add_argument("--feature", type=str, default="AE", help="feature from 'DDPM' or 'DDPM'")
    parser.add_argument("--mode", type=str, default="mesh", help="mode for 'mesh' or 'random'")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--partial_percent", type=float, default=1.00, help="Value for partially showing the labels (0 to 1 range)")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Kernel sizes
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[32, 16], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[32], help="List of decoder kernel sizes")

    # Evaluation mode
    parser.add_argument("--experiment_id", type=int, default=0, help="Provide an experiment id to test the produced models")

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
    feature_extracted = True if args.feature == "DDPM" else False
    mesh = True if args.mode == "mesh" else False
    d = args.d
    n_pins = args.n_pins
    r = args.r
    partial_percent = args.partial_percent

    batch_size = args.batch_size
    num_kernels_encoder = args.num_encoder
    num_kernels_decoder = args.num_decoder
    
    experiment_id = args.experiment_id
    
    input_channel = 1 if dataset == "PinMNIST" else 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extracted:
        folder = f"{dataset}_ddpm"
    else:
        folder = f"{dataset}"
    
    if dataset == "PinMNIST":
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_{28}by{28}pixels_{r}radius_{seed}seed"
        else:
            data_folder = f"./data/{folder}/random_fixedTrue_{n_pins}pins_{28}by{28}pixels_{r}radius_{seed}seed"
    
    if dataset == "Synthetic":
        if mesh:
            data_folder = f"./data/{folder}/{d}_distanced_grid_pins_{seed}seed/"
        else:
            data_folder = f"./data/{folder}/upto{n_pins}pins_{seed}seed/"

    transform = transforms.Compose([
        ToTensor(),         # Convert to tensor (as you were doing)
        Resize()  # Resize to 100x100
    ])
    
    transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                          root_dir=f"./data/{folder}/images/",
                                          transform=transform)
    
    dataset_size = len(transformed_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        transformed_dataset, [train_size, val_size, test_size]
    )

    # Create your DataLoader with the custom_collate_fn
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Testing
    if not os.path.exists(f'./history/{experiment_id}'):
        raise Exception(f"Could not find experiment with id: {experiment_id}")
    else:
        autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)
        # MSE
        try:
            autoencoder.load_state_dict(torch.load(f'./history/{experiment_id}/best_model_MSE.pth'))
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
        best_MSE_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device, partial_label_GP=False, partial_percent=partial_percent)
        # NPP
        try:
            autoencoder.load_state_dict(torch.load(f'./history/{experiment_id}/best_model_NPP.pth'))
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
        best_NPP_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device, partial_label_GP=False, partial_percent=partial_percent)
        GP_best_NPP_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device, partial_label_GP=True, partial_percent=partial_percent)
        # Write output into file
        filename = f"test_{folder}_{partial_percent}"
        with open(f"./history/{experiment_id}/{filename}", "w") as f:
            f.write(f"MSE {best_MSE_test_loss}; NPP {best_NPP_test_loss}, {GP_best_NPP_test_loss} (GP)")

if __name__ == "__main__":
    main()