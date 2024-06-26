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


class CustomLRFinder:
    def __init__(self, model, criterion, optimizer, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {'lr': [], 'loss': []}

    def find_lr(self, train_loader, input_channel, start_lr=1e-4, end_lr=1, num_iter=20, smooth_f=0.05):
        model = self.model.to(self.device)
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device
        model.train()

        lr_step = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr

        for iteration in range(num_iter):
            optimizer.param_groups[0]['lr'] = lr

            total_loss = 0.0
            for batch in train_loader:
                x_train = batch['image'][:, :input_channel, :, :].to(device)
                p_train = [tensor.to(device) for tensor in batch['pins']]
                y_train = [tensor.to(device) for tensor in batch['outputs']]
                optimizer.zero_grad()
                outputs = model(x_train.float())
                loss = criterion(y_train, outputs, p_train)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history['lr'].append(lr)
            self.history['loss'].append(avg_loss)

            lr *= lr_step

    def plot_lr_finder(self):
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')  # Use a logarithmic scale for better visualization
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder Curve')
        plt.show()

    def find_best_lr(self, skip_start=3, skip_end=3):
        # Find the index of the minimum loss in the specified range
        min_loss_index = skip_start + np.argmin(self.history['loss'][skip_start:-skip_end])

        # Output the learning rate corresponding to the minimum loss
        best_lr = self.history['lr'][min_loss_index]
        return best_lr


def custom_collate_fn(batch):
    images = [sample['image'] for sample in batch]
    pins = [sample['pins'] for sample in batch]
    outputs = [sample['outputs'] for sample in batch]

    return {
        'image': torch.stack(images, dim=0),
        'pins': pins,
        'outputs': outputs}


def run_pipeline_ci(sigmas, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader,
                    test_loader, input_channel, epochs, val_every_epoch, config, device, num_runs=3, exp_name=""):
    GP_test_losses_npp_true = []
    test_losses_npp_true = []
    test_losses_npp_false = []
    GP_r2_losses_npp_true = []
    r2_losses_npp_true = []
    r2_losses_npp_false = []
    partial_percent = config['partial_percent']
    experiment_id = int(time.time())
    best_val_loss_MSE = float('inf')
    best_val_loss_NPP = float('inf')
    best_sigma_NPP = float('inf')
    config['experiment_id'] = experiment_id
    deeper = config['deeper']
    manual_lr = config['manual_lr']
    learning_rates = config['best_lrs']
    losses = {}

    # Create storage directory and store the experiment configuration
    if not os.path.exists(f'./history/{exp_name}/{experiment_id}'):
        os.makedirs(f'./history/{exp_name}/{experiment_id}')
    with open(f"./history/{exp_name}/{experiment_id}/config.json", "w") as outfile:
        json.dump(config, outfile)

    for run in range(num_runs):
        count = 0
        test_losses_vs_sigma_npp_true = []
        R2_losses_vs_sigma_npp_true = []
        GP_test_losses_vs_sigma_npp_true = []
        GP_R2_losses_vs_sigma_npp_true = []
        for sigma in sigmas:
            early_stopping = EarlyStoppingCallback(patience=15, min_delta=0.001)
            autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
            lr = learning_rates[count][1]
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            print(f"training start for sigma:{sigma}")
            if sigma == 0:
                # run plain
                
                criterion = NPPLoss(identity=True).to(device)
                model, train_losses, val_losses, best_val_loss = train_model(autoencoder, train_loader, val_loader,
                                                                             input_channel, epochs, \
                                                                             val_every_epoch, lr,
                                                                             criterion, optimizer, device, early_stopping,
                                                                             experiment_id, exp_name, best_val_loss_MSE, manual_lr, sigma=0)
                losses[f"MSE_run{run}_train"] = train_losses
                losses[f"MSE_run{run}_val"] = val_losses
                if best_val_loss < best_val_loss_MSE:
                    best_val_loss_MSE = best_val_loss

                test_loss_npp_false, r2_loss_npp_false = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                                        partial_label_GP=False, partial_percent=partial_percent)
                print(f"MSE Test loss:{test_loss_npp_false:.3f}")
                print(f"R2 Test loss:{r2_loss_npp_false:.3f}")
                test_losses_npp_false.append(test_loss_npp_false)
                r2_losses_npp_false.append(r2_loss_npp_false)        
            else:
                # run NPP
                criterion = NPPLoss(identity=False, sigma=sigma).to(device)               
                
                model, train_losses, val_losses, best_val_loss = train_model(autoencoder, train_loader, val_loader,
                                                                             input_channel, epochs, \
                                                                             val_every_epoch, lr,
                                                                             criterion, optimizer, device, early_stopping,
                                                                             experiment_id, exp_name, best_val_loss_NPP, manual_lr, sigma=sigma)
                losses[f"NPP_run{run}_sigma{sigma}_train"] = train_losses
                losses[f"NPP_run{run}_sigma{sigma}_val"] = val_losses
                if best_val_loss < best_val_loss_NPP:
                    best_val_loss_NPP = best_val_loss
                    best_sigma_NPP = sigma

                test_loss, r2_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                    partial_label_GP=False, partial_percent=partial_percent)
                GP_test_loss, GP_r2_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                          partial_label_GP=True, partial_percent=partial_percent)
                print(f"NPP sigma={sigma} Test loss:{test_loss:.3f}, R2 loss:{r2_loss:.3f}, GP Test loss:{GP_test_loss:.3f}"
                      f", GP R2 loss:{GP_r2_loss:.3f}")
                test_losses_vs_sigma_npp_true.append(test_loss)
                R2_losses_vs_sigma_npp_true.append(r2_loss)
                GP_test_losses_vs_sigma_npp_true.append(GP_test_loss)
                GP_R2_losses_vs_sigma_npp_true.append(GP_r2_loss)
            count += 1

        test_losses_npp_true.append(test_losses_vs_sigma_npp_true)
        GP_test_losses_npp_true.append(GP_test_losses_vs_sigma_npp_true)
        r2_losses_npp_true.append(R2_losses_vs_sigma_npp_true)
        GP_r2_losses_npp_true.append(GP_R2_losses_vs_sigma_npp_true)
    with open(f"./history/{exp_name}/{experiment_id}/losses.json", "w") as outfile:
        json.dump(losses, outfile)
    return GP_test_losses_npp_true, test_losses_npp_true, test_losses_npp_false, r2_losses_npp_false, r2_losses_npp_true, GP_r2_losses_npp_true, best_sigma_NPP, experiment_id


# Function to run the pipeline and save data
def run_and_save_pipeline(sigmas, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader,
                          input_channel, epochs, val_every_epoch, config, num_runs, exp_name, device):
    # Run the pipeline
    GP_test_loss_npp_true, test_loss_npp_true, test_loss_npp_false, r2_loss_npp_false, r2_losses_npp_true, GP_r2_losses_npp_true, best_sigma_NPP, experiment_id = run_pipeline_ci(
        sigmas, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader, input_channel, epochs,
        val_every_epoch, config, device, num_runs, exp_name)
    partial_percent = config['partial_percent']
    # Run final testing
    autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=config["deeper"]).to(device)
    if sigmas[0] == 0:
        # MSE
        autoencoder.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_MSE.pth'))
        best_MSE_test_loss, best_R2_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                               partial_label_GP=False,
                                                               partial_percent=partial_percent)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "w")
        f.write(f"Results {experiment_id}:\n MSE: {best_MSE_test_loss}, R2: {best_R2_test_loss} ")
        f.close()
        print("metrics saved")
    else:
        # NPP
        autoencoder.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_NPP.pth'))
        best_NPP_test_loss, best_NPP_R2_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                                   partial_label_GP=False,
                                                                   partial_percent=partial_percent)
        GP_best_NPP_test_loss, GP_best_NPP_r2_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device,
                                                                         partial_label_GP=True,
                                                                         partial_percent=partial_percent)
        f = open(f"./history/{exp_name}/{experiment_id}/results.txt", "w")
        f.write(f"| NPP (sigma {best_sigma_NPP}): {best_NPP_test_loss}, R2: {best_NPP_R2_test_loss}; GP: {GP_best_NPP_test_loss}, R2: {GP_best_NPP_r2_test_loss}")
        f.close()
        print("metrics saved")
        
    print("start saving losses!")
    # Save losses
    save_loss(test_loss_npp_true, f'./history/{exp_name}/{experiment_id}/test_loss_npp_true.npy')
    save_loss(test_loss_npp_false, f'./history/{exp_name}/{experiment_id}/test_loss_npp_false.npy')
    save_loss(GP_test_loss_npp_true, f'./history/{exp_name}/{experiment_id}/GP_test_loss_npp_true.npy')
    # Save r2 scores
    save_loss(r2_loss_npp_false, f'./history/{exp_name}/{experiment_id}/r2_loss_npp_false.npy')
    save_loss(r2_losses_npp_true, f'./history/{exp_name}/{experiment_id}/r2_losses_npp_true.npy')
    save_loss(GP_r2_losses_npp_true, f'./history/{exp_name}/{experiment_id}/GP_r2_losses_npp_true.npy')
    
    return (GP_test_loss_npp_true, test_loss_npp_true, test_loss_npp_false), (r2_loss_npp_false, r2_losses_npp_true, GP_r2_losses_npp_true), experiment_id


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
    parser.add_argument("--partial_percent", type=float, default=0.00,
                        help="Value for partially showing the labels (0 to 1 range)")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--val_every_epoch", type=int, default=5, help="Number of epochs in between validations")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of different trainings to do per model and sigma")
    parser.add_argument("--manual_lr", action='store_true', default=False, help="Do not use Custom LR Finder")

    # List of sigma values
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0],
                        help="List of sigma values to test")

    # Model configuration
    parser.add_argument("--num_encoder", nargs="+", type=int, default=[64, 32], help="List of encoder kernel sizes")
    parser.add_argument("--num_decoder", nargs="+", type=int, default=[64], help="List of decoder kernel sizes")
    parser.add_argument("--deeper", action='store_true', default=False, help="Add extra convolutional layer for the model")
    
    # Experiment title
    parser.add_argument("--experiment_name", type=str, default=None, help="Define if you want to save the generated experiments in an specific folder")

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
    n = args.n
    mesh = True if args.mode == "mesh" else False
    d = args.d
    n_pins = args.n_pins
    r = args.r
    partial_percent = args.partial_percent
    exp_name = args.experiment_name if args.experiment_name is not None else ""

    # Set your hyperparameters
    epochs = args.epochs
    modality = args.modality
    batch_size = args.batch_size
    sigmas = args.sigmas  # Set the sigma values you want to test
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
            if modality == "PS-RGBNIR":
                input_channel = 4
            elif modality == "PS-RGB":
                input_channel = 3
            elif modality == "PS-RGBNIR-SAR":
                input_channel = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extracted:
        folder = f"{dataset}_ddpm"
    else:
        folder = f"{dataset}"

    if dataset == "PinMNIST":
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step"
            config['n_pins'] = (28 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/random_{n_pins}pins"
    elif dataset == "Synthetic":
        if mesh:
            data_folder = f"./data/{folder}/28by28pixels_1000images_123456seed/mesh_{d}step_pins"
            config['n_pins'] = (28 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/28by28pixels_1000images_123456seed/random_{n_pins}pins"
    elif dataset == "Building":
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}_step"
            config['n_pins'] = (100 // d + 1) ** 2
        else:
            data_folder = f"./data/{folder}/random_n_pins_{n_pins}"
    
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
    # As DDPM does not work well with Rotterdam Building dataset, we have not explored this dataset with different modalities with DDPM
    if dataset == "Building":
        root_dir = f"./data/Building/{modality}/"
        transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                     root_dir=root_dir, modality=modality,
                                     transform=transform)
    else:
        root_dir=f"./data/{folder}/images/"
        transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                     root_dir=root_dir, transform=transform)

    

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

    if not manual_lr:
        model = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        best_lrs = []
        for sigma in sigmas:
            if sigma == 0:
                criterion_MSE = NPPLoss(identity=True).to(device)
                lr_finder_MSE = CustomLRFinder(model, criterion_MSE, optimizer, device=device)
                lr_finder_MSE.find_lr(train_loader, input_channel=input_channel, start_lr=1e-5, end_lr=1, num_iter=20)
                best_lr_MSE = lr_finder_MSE.find_best_lr()
                print(f"Best Learning Rate for MSE: {best_lr_MSE}")
                best_lrs.append((0, best_lr_MSE))
            else:
                criterion_NPP = NPPLoss(identity=False, sigma=sigma).to(device)
                lr_finder_NPP = CustomLRFinder(model, criterion_NPP, optimizer, device=device)
                lr_finder_NPP.find_lr(train_loader, input_channel=input_channel, start_lr=1e-4, end_lr=1, num_iter=10)
                best_lr_NPP = lr_finder_NPP.find_best_lr()
                best_lrs.append((sigma, best_lr_NPP))
                print(f"Best Learning Rate for NPP sigma={sigma}: {best_lr_NPP}")
    else:
        best_lrs = [(sigma, 1e-2) for sigma in sigmas] # Initial LR for MSE and each sigma
    config['best_lrs'] = best_lrs
    # Run and save the pipeline data
    loss_vs_sigma_data, _, experiment_id = run_and_save_pipeline(sigmas, num_kernels_encoder, num_kernels_decoder,
                                                              train_loader, val_loader, test_loader, \
                                                              input_channel, epochs, val_every_epoch, config,
                                                              num_runs, exp_name, device)

    # Plot and save the plot using the saved data
    # plot_and_save(loss_vs_sigma_data, sigmas, dataset, learning_rate, results_dir=f'./history/{exp_name}/{experiment_id}')

    # Testing
    if not os.path.exists(f'./history/{exp_name}/{experiment_id}'):
        raise Exception(f"Could not find experiment with id: {experiment_id}")
    else:
        if sigmas[0] == 0:
            autoencoder_MSE = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
            try:
                autoencoder_MSE.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_MSE.pth'))
            except:
                raise Exception(
                    "The model you provided does not correspond with the selected architecture. Please revise and try again.")
            best_MSE_test_loss, best_R2_test_loss_MSE =  evaluate_model(autoencoder_MSE, test_loader, input_channel, device,
                                                                       partial_label_GP=False, partial_percent=0)
            filename = f"test_{folder.split('/')[0]}.txt"
            with open(f"./history/{exp_name}/{experiment_id}/{filename}", "w") as f:
                    f.write(
                        f"MSE: {best_MSE_test_loss}, R2: {best_R2_test_loss_MSE} ")

            
        else:
            autoencoder_NPP = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel, deeper=deeper).to(device)
            # Load models
            try:
                autoencoder_NPP.load_state_dict(torch.load(f'./history/{exp_name}/{experiment_id}/best_model_NPP.pth'))
            except:
                raise Exception(
                    "The model you provided does not correspond with the selected architecture. Please revise and try again.")
            # NPP
            for percent in [0.25, 0.50, 0.75, 1.00]:
                print(f'Percent testing {percent}')  
                best_NPP_test_loss, best_R2_test_loss_NPP = evaluate_model(autoencoder_NPP, test_loader, input_channel, device,
                                                       partial_label_GP=False, partial_percent=percent)
                GP_best_NPP_test_loss, best_R2_test_loss_GP = evaluate_model(autoencoder_NPP, test_loader, input_channel, device,
                                                          partial_label_GP=True, partial_percent=percent)
                # Write output into file
                filename = f"test_{folder.split('/')[0]}_{percent}.txt"
                with open(f"./history/{exp_name}/{experiment_id}/{filename}", "w") as f:
                    f.write(
                        f"| NPP: {best_NPP_test_loss}, R2: {best_R2_test_loss_NPP}; GP: {GP_best_NPP_test_loss}, R2: {best_R2_test_loss_GP}")

    end_time = time.time()
    print(f"Time Elapsed: {(end_time - start_time) / 3600} hours")
if __name__ == "__main__":
    main()
