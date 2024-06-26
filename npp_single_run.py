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
from tools.optimization import EarlyStoppingCallback, evaluate_model, train_model
import matplotlib.pyplot as plt
import argparse
import time
from tools.models import *
from torch.utils.data import Subset
import time


def custom_collate_fn(batch):
    images = [sample['image'] for sample in batch]
    pins = [sample['pins'] for sample in batch]
    outputs = [sample['outputs'] for sample in batch]

    return {
        'image': torch.stack(images, dim=0),
        'pins': pins,
        'outputs': outputs}


def run_pipeline_ci(sigma, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader,
                    test_loader, input_channel, epochs, val_every_epoch, config, device, num_runs=3, exp_name=""):
    test_losses_npp_true = []
    test_losses_npp_false = []
    r2_losses_npp_true = []
    r2_losses_npp_false = []
    experiment_id = int(time.time())
    best_val_loss_MSE = float('inf')
    best_val_loss_NPP = float('inf')
    config['experiment_id'] = experiment_id
    deeper = config['deeper']
    manual_lr = config['manual_lr']
    losses = {}
    lr = config["best_lr"]
    print(f"The current LR is: {lr}")
    # Create storage directory and store the experiment configuration
    if not os.path.exists(f'./history/{exp_name}/{experiment_id}'):
        os.makedirs(f'./history/{exp_name}/{experiment_id}')
    with open(f"./history/{exp_name}/{experiment_id}/config.json", "w") as outfile:
        json.dump(config, outfile)

    for run in range(num_runs):
        count = 0
        test_losses_vs_sigma_npp_true = []
        R2_losses_vs_sigma_npp_true = []

        # Run NPP=False once and collect the test loss
        early_stopping = EarlyStoppingCallback(patience=15, min_delta=0.001)

        if sigma == 0:
            criterion = NPPLoss(identity=True).to(device)
            autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)

            optimizer = optim.Adam(autoencoder.parameters(), lr=config["best_lr"])
            model, train_losses, val_losses, best_val_loss = train_model(autoencoder, train_loader, val_loader,
                                                                         input_channel, epochs, \
                                                                         val_every_epoch, config["best_lr"],
                                                                         criterion, optimizer, device, early_stopping,
                                                                         experiment_id, exp_name, best_val_loss_MSE, manual_lr=manual_lr, sigma=0)
            losses[f"MSE_run{run}_train"] = train_losses
            losses[f"MSE_run{run}_val"] = val_losses
            if best_val_loss < best_val_loss_MSE:
                best_val_loss_MSE = best_val_loss

            test_loss_npp_false, r2_loss_npp_false = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                                    partial_label_GP=False, partial_percent=0)
            print(f"MSE Test loss:{test_loss_npp_false:.3f}")
            print(f"R2 Test loss:{r2_loss_npp_false:.3f}")
            test_losses_npp_false.append(test_loss_npp_false)
            r2_losses_npp_false.append(r2_loss_npp_false)

        else:
            early_stopping = EarlyStoppingCallback(patience=12, min_delta=0.001)
            criterion = NPPLoss(identity=False, sigma=sigma).to(device)
            autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
            optimizer = optim.Adam(autoencoder.parameters(), lr=config["best_lr"])
            model, train_losses, val_losses, best_val_loss = train_model(autoencoder, train_loader, val_loader,
                                                                         input_channel, epochs, \
                                                                         val_every_epoch, config["best_lr"],
                                                                         criterion, optimizer, device, early_stopping,
                                                                         experiment_id, exp_name, best_val_loss_NPP, manual_lr=manual_lr, sigma=sigma)
            losses[f"NPP_run{run}_sigma{sigma}_train"] = train_losses
            losses[f"NPP_run{run}_sigma{sigma}_val"] = val_losses
            if best_val_loss < best_val_loss_NPP:
                best_val_loss_NPP = best_val_loss

            test_loss, r2_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                partial_label_GP=False, partial_percent=0)
            GP_test_loss, GP_r2_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                      partial_label_GP=True, partial_percent=0)
            print(f"NPP sigma={sigma} Test loss:{test_loss:.3f}, R2 loss:{r2_loss:.3f}, GP Test loss:{GP_test_loss:.3f}"
                  f", GP R2 loss:{GP_r2_loss:.3f}")
            test_losses_vs_sigma_npp_true.append(test_loss)
            R2_losses_vs_sigma_npp_true.append(r2_loss)
            count += 1

        test_losses_npp_true.append(test_losses_vs_sigma_npp_true)
        r2_losses_npp_true.append(R2_losses_vs_sigma_npp_true)
    with open(f"./history/{exp_name}/{experiment_id}/losses.json", "w") as outfile:
        json.dump(losses, outfile)
    return test_losses_npp_true, test_losses_npp_false, r2_losses_npp_false, r2_losses_npp_true, experiment_id


# Function to run the pipeline and save data
def run_and_save_pipeline(sigma, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader,
                          input_channel, epochs, val_every_epoch, config, num_runs, exp_name, device):
    # Run the pipeline
    test_loss_npp_true, test_loss_npp_false, r2_loss_npp_false, r2_losses_npp_true, experiment_id = run_pipeline_ci(
        sigma, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader, input_channel, epochs,
        val_every_epoch, config, device, num_runs, exp_name)
    # Run final testing
    autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=config["deeper"]).to(device)
    # MSE
    if sigma == 0:
        autoencoder.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_MSE.pth'))
        best_MSE_test_loss, best_R2_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                               partial_label_GP=False,
                                                               partial_percent=0)
    else:
    # NPP
        autoencoder.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_NPP.pth'))
        best_NPP_test_loss, best_NPP_R2_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                                   partial_label_GP=False,
                                                                   partial_percent=0)
    print("start saving!")
    f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "w")
    if sigma == 0:
        f.write(
            f"Results {experiment_id}:\n MSE: {best_MSE_test_loss}, R2: {best_R2_test_loss} ")
        f.close()
        print(f"MSE saved: experiment id {experiment_id}")
    else:
        f.write(
            f"Results {experiment_id}:\n | NPP (sigma {sigma}): {best_NPP_test_loss}, R2: {best_NPP_R2_test_loss}")
        f.close()
        print(f"NPP saved: experiment id {experiment_id}")
    return (test_loss_npp_true, test_loss_npp_false), (r2_loss_npp_false, r2_losses_npp_true), experiment_id


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="Building", help="Dataset name")
    parser.add_argument("--feature", type=str, default="DDPM", help="feature from 'DDPM' or 'DDPM'")
    parser.add_argument("--mode", type=str, default="mesh", help="mode for 'mesh' or 'random'")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_every_epoch", type=int, default=5, help="Number of epochs in between validations")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of different trainings to do per model and sigma")
    parser.add_argument("--manual_lr", action='store_true', default=False, help="Do not use Custom LR Finder")

    # List of sigma values

    parser.add_argument("--sigma", type=float, default=0, help="a single sigma values to test (0 indicates MSE)")

    # Model configuration
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[32, 16], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[32], help="List of decoder kernel sizes")
    parser.add_argument("--deeper", action='store_true', default=False, help="Add extra convolutional layer for the model")

    # Evaluation mode
    parser.add_argument("--experiment_id", type=int, default=0,
                        help="Provide an experiment id to test the produced models")
    
    # Experiment title
    parser.add_argument("--experiment_name", type=str, default="single_run", help="Define if you want to save the generated experiments in an specific folder")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
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
    exp_name = '/' + args.experiment_name if args.experiment_name is not None else ""

    # Set your hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    sigma = args.sigma  # Set the sigma values you want to test
    num_kernels_encoder = args.num_encoder
    num_kernels_decoder = args.num_decoder
    deeper = args.deeper
    learning_rate = args.learning_rate
    val_every_epoch = args.val_every_epoch
    num_runs = args.num_runs
    manual_lr = args.manual_lr

    config = vars(args)
    config['seed'] = seed

    if dataset == "Synthetic":
        if feature_extracted:
            input_channel = 74
        else:
            input_channel = 3
    elif dataset == "PinMNIST":
        if feature_extracted:
            input_channel = 71
        else:
            input_channel = 1
    elif dataset == "Building":
        if feature_extracted:
            input_channel = 3584
        else:
            input_channel = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extracted:
        folder = f"{dataset}_ddpm"
    else:
        folder = f"{dataset}"

    if dataset == "PinMNIST":
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_{28}by{28}pixels_{r}radius_{seed}seed"
            config['n_pins'] = (28 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/random_fixedTrue_{n_pins}pins_{28}by{28}pixels_{r}radius_{seed}seed"
    elif dataset == "Synthetic":
        folder += "/28by28pixels_1000images_123456seed"
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_pins"
            config['n_pins'] = (28 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/random_{n_pins}pins"
    elif dataset == "Building":
        if mesh:
            data_folder = f"./data/{folder}/processed/mesh_{d}_step"
            config['n_pins'] = (100 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/processed/random_n_pins_{n_pins}"
    
    if dataset == "Building":
        transform = transforms.Compose([
        ToTensor(),  # Convert to tensor (as you were doing)
        Resize100(),  # Resize to 100x100
    ])
    else:
        transform = transforms.Compose([
        ToTensor(),  # Convert to tensor (as you were doing)
        Resize()  # Resize to 100x100
    ])

    if dataset == "Building" and feature_extracted:
        root_dir = "/work/DNAL/Datasets/Building_ddpm/images/"
        if mesh:
            data_folder = f"/work/DNAL/Datasets/Building_ddpm/processed/mesh_{d}_step"
            config['n_pins'] = (100 // d + 1) ** 2
        else:
            data_folder = f"/work/DNAL/Datasets/Building_ddpm/processed/random_n_pins_{n_pins}"
    else:
        root_dir = f"/work/DNAL/Datasets/Building_ddpm/images/"
        
    transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                     root_dir=root_dir,
                                     transform=transform)

    dataset_size = len(transformed_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = dataset_size - train_size - val_size


    if os.path.exists(f"./data/{dataset}/train_indices.npy"):
        train_indices = np.load(f'./data/{dataset}/train_indices.npy')
        val_indices = np.load(f'./data/{dataset}/val_indices.npy')
        test_indices = np.load(f'./data/{dataset}/test_indices.npy')
        # Use the indices to create new datasets
        train_dataset = Subset(transformed_dataset, train_indices)
        val_dataset = Subset(transformed_dataset, val_indices)
        test_dataset = Subset(transformed_dataset, test_indices)
    else:
        # Split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(
            transformed_dataset, [train_size, val_size, test_size]
        )
        np.save(f'./data/{dataset}/train_indices.npy', train_dataset.indices)
        np.save(f'./data/{dataset}/val_indices.npy', val_dataset.indices)
        np.save(f'./data/{dataset}/test_indices.npy', test_dataset.indices)

    # Create your DataLoader with the custom_collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


    best_lr = 0.0001 if sigma==0 else 0.001
    config['best_lr'] = best_lr
    # Run and save the pipeline data
    loss_vs_sigma_data, _, experiment_id = run_and_save_pipeline(sigma, num_kernels_encoder, num_kernels_decoder,
                                                              train_loader, val_loader, test_loader, \
                                                              input_channel, epochs, val_every_epoch, config,
                                                              num_runs, exp_name, device)
    # Testing
    if sigma == 0:
        autoencoder_MSE = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
        autoencoder_MSE.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_MSE.pth'))
        filename = f"test_{folder.split('/')[0]}_MSE.txt"
        MSE_loss, MSE_R2 = evaluate_model(autoencoder_MSE, test_loader, input_channel, device,
                                                                   partial_label_GP=False, partial_percent=0)
        print(f"saving to /history{exp_name}/{experiment_id}/{filename}")
        with open(f"./history/{exp_name}/{experiment_id}/{filename}", "w") as f:
            f.write(f"MSE: {MSE_loss}, R2: {MSE_R2} ")

    else:
        autoencoder_NPP = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel,).to(device)
        autoencoder_NPP.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_NPP.pth'))

        for percent in [0.25, 0.50, 0.75, 1.00]:
            print(f'Percent testing {percent}')
            best_NPP_test_loss, best_R2_test_loss_NPP = evaluate_model(autoencoder_NPP, test_loader, input_channel, device,
                                                   partial_label_GP=False, partial_percent=percent)
            GP_best_NPP_test_loss, best_R2_test_loss_GP = evaluate_model(autoencoder_NPP, test_loader, input_channel, device,
                                                      partial_label_GP=True, partial_percent=percent)
            # Write output into file
            filename = f"test_{folder.split('/')[0]}_{percent}.txt"
            print(f"saving to /history/{exp_name}/{experiment_id}/{filename}")
            with open(f"./history/{exp_name}/{experiment_id}/{filename}", "w") as f:
                f.write(
                    f"| NPP: {best_NPP_test_loss}, R2: {best_R2_test_loss_NPP}; percent: {percent} -- GP: {GP_best_NPP_test_loss}, R2: {best_R2_test_loss_GP}")

    end_time = time.time()
    print(f"Time Elapsed: {(end_time - start_time) / 3600} hours")
if __name__ == "__main__":
    main()
