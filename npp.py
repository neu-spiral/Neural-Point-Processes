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
from tools.NeuralPointProcess import NeuralPointProcesses


def custom_collate_fn(batch):
    images = [sample['image'] for sample in batch]
    pins = [sample['pins'] for sample in batch]
    outputs = [sample['outputs'] for sample in batch]

    return {
        'image': torch.stack(images, dim=0),
        'pins': pins,
        'outputs': outputs}


def data_prepare(config, seed):
    dataset = config['dataset']
    mesh = True if config['mode'] == "mesh" else False
    feature_extracted = True if config['feature'] == "DDPM" else False
    modality = config['modality']
    batch_size = config['batch_size']
    n_pins = config['n_pins']
    d = config['d']
    r = config['r']

    if dataset == "Synthetic":
        input_shape = 28
        if feature_extracted:
            input_channel = 74
        else:
            input_channel = 3
    elif dataset == "PinMNIST":
        input_shape = 28
        if feature_extracted:
            input_channel = 71
        else:
            input_channel = 1
    elif dataset == "Building":
        input_shape = 100
        if feature_extracted:
            input_channel = 3584
        else:
            if modality == "PS-RGBNIR":
                input_channel = 4
            elif modality == "PS-RGB":
                input_channel = 3
            elif modality == "PS-RGBNIR-SAR":
                input_channel = 8
    elif dataset == "Cars":
        if feature_extracted:
            # TO DO: Check how many features does the DDPM version has
            print('DDPM is still not available for this dataset')
        else:
            input_channel = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extracted:
        folder = f"{dataset}_ddpm"
    else:
        folder = f"{dataset}"

    if dataset == "PinMNIST":
        test_data_folder = f"./data/{folder}/random_fixedTrue_{n_pins}pins_{28}by{28}pixels_{r}radius_4seed"
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_{28}by{28}pixels_{r}radius_4seed"
            config['n_pins'] = (28 // d + 1) ** 2
        else: # Random pins 
            data_folder = test_data_folder

    elif dataset == "Synthetic":
        folder += "/28by28pixels_1000images_123456seed"
        test_data_folder = f"./data/{folder}/random_{n_pins}pins"
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_pins"
            config['n_pins'] = (28 // d + 1) ** 2
        else:

            data_folder = test_data_folder
    elif dataset == "Building":
        test_data_folder = f"./data/{folder}/random_n_pins_{n_pins}"
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}_step"
            config['n_pins'] = (100 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/random_n_pins_{n_pins}"
            data_folder = test_data_folder
            
    elif dataset == "Cars":
        r = 100
        test_data_folder = f"./data/{folder}/test/random_fixedTrue_{n_pins}pins_{800}by{800}pixels_{r}radius_{seed}seed"
        if mesh:
            train_data_folder = f"./data/{folder}/train/mesh_{d}step_{800}by{800}pixels_{r}radius_{seed}seed"
            val_data_folder = f"./data/{folder}/val/mesh_{d}step_{800}by{800}pixels_{r}radius_{seed}seed"
            config['n_pins'] = (800 // d + 1) ** 2
        else: # Random pins 
            data_folder = test_data_folder

    if dataset == "Building":
        transform = transforms.Compose([
        ToTensor(),  # Convert to tensor (as you were doing)
        Resize100(),  # Resize to 100x100
    ])
    elif dataset == "Cars":
        transform = transforms.Compose([
        ExtractImage(), # Get image from image and mask combination
        ToTensor(),  # Convert to tensor (as you were doing)
        Resize200(),  # Resize to 200x200
    ])
    else:
        transform = transforms.Compose([
        ToTensor(),  # Convert to tensor (as you were doing)
        Resize()  # Resize to 100x100
    ])        
    # As DDPM does not work well with Rotterdam Building dataset, we have not explored this dataset with different modalities with DDPM
    if dataset == "Building":
        root_dir = f"./data/Building/{modality}/"
        transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                     root_dir=root_dir, modality=modality,
                                     transform=transform)
        test_dataset = PinDataset(csv_file=f"{test_data_folder}/pins.csv",
                                     root_dir=root_dir, modality=modality,
                                     transform=transform)
    elif dataset == "Cars":
        root_dir=f"./data/{folder}/images/"
        train_dataset = PinDataset(csv_file=f"{train_data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)
        val_dataset = PinDataset(csv_file=f"{val_data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)
        eval_dataset = PinDataset(csv_file=f"{test_data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)
    else:
        root_dir=f"./data/{folder}/images/"
        transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)
        test_dataset = PinDataset(csv_file=f"{test_data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)

    dataset_size = len(transformed_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = dataset_size - train_size - val_size

    if dataset != "Cars":
        if os.path.exists(f"./data/{dataset}/train_indices.npy"):
            train_indices = np.load(f'./data/{dataset}/train_indices.npy')
            val_indices = np.load(f'./data/{dataset}/val_indices.npy')
            test_indices = np.load(f'./data/{dataset}/test_indices.npy')
            # Use the indices to create new datasets
            train_dataset = Subset(transformed_dataset, train_indices)
            val_dataset = Subset(transformed_dataset, val_indices)
            # test_dataset = Subset(transformed_dataset, test_indices)
             # Use the indices to create new test datasets
            eval_dataset = Subset(test_dataset, test_indices)
        else:
            # Split the dataset into train, validation, and test sets
            train_dataset, val_dataset, test_dataset = random_split(
                transformed_dataset, [train_size, val_size, test_size]
            )
            np.save(f'./data/{dataset}/train_indices.npy', train_dataset.indices)
            np.save(f'./data/{dataset}/val_indices.npy', val_dataset.indices)
            np.save(f'./data/{dataset}/test_indices.npy', test_dataset.indices)
            # Use the indices to create new test datasets
            eval_dataset = Subset(test_dataset, test_dataset.indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return input_channel, input_shape, train_loader, val_loader, eval_loader
    

def run_experiments(config, train_loader, val_loader,
                    eval_loader, input_channel, input_shape, device):  
    best_val_loss_MSE = float('inf')
    best_val_loss_NPP = float('inf')
    deeper = config['deeper']
    kernel = config['kernel']
    manual_lr = config['manual_lr']
    # learning_rates = config['best_lrs']   
    num_runs = config['num_runs']
    val_every_epoch = config['val_every_epoch']
    epochs = config['epochs']
    exp_name = config['experiment_name']
    dataset = config['dataset']
    lr = 1e-4 if dataset == "PinMNIST" else 1e-2
    num_encoder = config['num_encoder']
    num_decoder = config['num_decoder']
    kernel = config['kernel']
    kernel_mode = config['kernel_mode']
    kernel_param = config['kernel_param']
    timestamp = int(time.time())
    experiment_id = f"{config['kernel']}_{config['kernel_mode']}_{config['kernel_param']}_{timestamp}"
    if dataset == "COWC":
        input_shape = 200
    elif dataset == "Building":
        input_shape = 100
    else:
        input_shape = 28
        
    losses = {}
    # Create storage directory and store the experiment configuration
    path = os.path.join(".", "history", exp_name, str(experiment_id))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as outfile:
        json.dump(config, outfile)

    # run the experiments for num_rums times and save the best-performing model
    for run in range(num_runs):
        if kernel_param == 0:
            # run MSE train
            NPP = NeuralPointProcesses(identity=True, num_encoder=num_encoder, num_decoder=num_decoder, input_shape=input_shape, input_channel=input_channel, deeper=deeper, lr=lr)
            train_losses, val_losses, best_val_loss = NPP.train_model(train_loader, val_loader,
                                                                         epochs, experiment_id, exp_name, 
                                                                         best_val_loss_MSE, val_every_epoch)
            losses[f"MSE_run{run}_train"] = train_losses
            losses[f"MSE_run{run}_val"] = val_losses
            if best_val_loss < best_val_loss_MSE:
                best_val_loss_MSE = best_val_loss
            MSE_test_loss, MSE_test_R2 = NPP.evaluate_model(eval_loader)
            print(f"MSE Loss| Loss: {MSE_test_loss}, R2: {MSE_test_R2} ")
        
        else:
            # run NPP train
            NPP = NeuralPointProcesses(kernel=kernel, kernel_mode=kernel_mode, num_encoder=num_encoder, num_decoder=num_decoder, input_shape=input_shape, input_channel=input_channel, deeper=deeper, kernel_param=kernel_param, lr=lr)
            train_losses, val_losses, best_val_loss = NPP.train_model(train_loader, val_loader,
                                                                         epochs, experiment_id, exp_name, 
                                                                         best_val_loss_NPP, val_every_epoch)
            
            losses[f"NPP_run{run}_train"] = train_losses
            losses[f"NPP_run{run}_val"] = val_losses
            if best_val_loss < best_val_loss_NPP:
                best_val_loss_NPP = best_val_loss
               
            for percent in [0.00, 0.25, 0.50, 0.75, 1.00]:
                NPP_test_loss, NPP_test_R2 = NPP.evaluate_model(eval_loader, partial_percent=percent)
                print(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2}")
   
    if kernel_param == 0:
        # MSE test
        NPP.load_best_model(experiment_id, exp_name)
        MSE_test_loss, MSE_test_R2 = NPP.evaluate_model(eval_loader)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "w")
        f.write(f"Results {experiment_id}: Best Val: {best_val_loss_MSE} \n MSE: {MSE_test_loss}, R2: {MSE_test_R2} ")
        f.close()
        print("metrics saved")
    else:
        # NPP test
        NPP.load_best_model(experiment_id, exp_name)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "a")
        f.write(f"Results {experiment_id}: Best Val: {best_val_loss_NPP} \n")
        for percent in [0.00, 0.25, 0.50, 0.75, 1.00]:
            print(f'Percent testing {percent}')  
            NPP_test_loss, NPP_test_R2 = NPP.evaluate_model(eval_loader, partial_percent=percent)
            print(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2}")
            # Write output into file
            
            f.write(f"Percent: {percent}| Loss: {NPP_test_loss}, R2: {NPP_test_R2} \n")
        f.close()
        print("metrics saved")
        
    print("start saving losses!")
    # Save losses
    save_loss(losses, f'./history/{exp_name}/{experiment_id}/losses.npy')


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="PinMNIST", help="Dataset name")
    parser.add_argument("--modality", type=str, default="PS-RGBNIR", help="Building dataset modality")
    parser.add_argument("--feature", type=str, default="AE", help="feature from 'AE' or 'DDPM'")
    parser.add_argument("--mode", type=str, default="mesh", help="mode for 'mesh' or 'random'")
    parser.add_argument("--n", type=int, default=100, help="Value for 'n'")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--val_every_epoch", type=int, default=5, help="Number of epochs in between validations")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of different trainings to do per model and sigma")
    parser.add_argument("--manual_lr", action='store_true', default=False, help="Do not use Custom LR Finder")

    # List of kernel_param values
    parser.add_argument("--kernel_param", type=float, default=[1],
                        help="a kernel param: Q=1 (SMK) or sigma =1 (RBF)")

    # Model configuration
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[64, 32], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[64], help="List of decoder kernel sizes")
    parser.add_argument("--deeper", action='store_true', default=False, help="Add extra convolutional layer for the model")
    parser.add_argument("--kernel_mode", type=str, default="fixed", help="fixed or learned, or predicted")
    parser.add_argument("--kernel", type=str, default="RBF", help="RBF or SM")
    
    # Experiment title
    parser.add_argument("--experiment_name", type=str, default=None, help="Define if you want to save the generated experiments in an specific folder")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    # Set a random seed for PyTorch
    seed = 0  # You can use any integer value as the seed
    torch.manual_seed(seed)
    # Set a random seed for NumPy (if you're using NumPy operations)
    np.random.seed(seed)

    config = vars(args)
    config['seed'] = seed
    dataset = config['dataset']
    input_channel, input_shape, train_loader, val_loader, eval_loader = data_prepare(config, seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run and save the pipeline data
    run_experiments(config, train_loader, val_loader, eval_loader, input_channel, input_shape, device)

    end_time = time.time()
    print(f"Time Elapsed: {(end_time - start_time) / 3600} hours")
    
    
if __name__ == "__main__":
    main()
