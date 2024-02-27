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
from tools.NPmodels import *
from tools.NPtrain import process_batch, NeuralProcessTrainer

class NPLRFinder:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {'lr': [], 'loss': []}

    def find_lr(self, train_loader, input_channel, start_lr=1e-4, end_lr=1, num_iter=20,smooth_f=0.05):
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
                x_context, y_context, x_target, y_target = process_batch(batch)

                optimizer.zero_grad()
                p_y_pred, q_target, q_context = model(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()
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
    
    def _loss(self, p_y_pred, y_target, q_target, q_context):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl

def custom_collate_fn(batch):
    images = [sample['image'] for sample in batch]
    pins = [sample['pins'] for sample in batch]
    outputs = [sample['outputs'] for sample in batch]

    return {
        'image': torch.stack(images, dim=0),
        'pins': pins,
        'outputs': outputs}

def run_pipeline_ci(train_loader, val_loader, 
                    test_loader, input_channel, epochs, val_every_epoch, learning_rates, config, device, num_runs=3):
    GP_test_losses_np = []
    test_losses_np = []
    partial_percent = config['partial_percent']
    experiment_id = int(time.time())
    best_val_loss_NP = float('inf')
    config['experiment_id'] = experiment_id
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    r_dim = config["r_dim"]
    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    lr = config["best_lr"]

    # Create storage directory and store the experiment configuration
    if not os.path.exists(f'./history/{experiment_id}'):
        os.makedirs(f'./history/{experiment_id}')
    with open(f"./history/{experiment_id}/config_np.json", "w") as outfile: 
        json.dump(config, outfile)
    
    for run in range(num_runs):
        count = 0
        test_losses_vs_sigma_np = []
        GP_test_losses_vs_sigma_np= []
        
        early_stopping = EarlyStoppingCallback(patience=10, min_delta=0.001)
        model = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        np_trainer = NeuralProcessTrainer(device, model, optimizer, print_freq=100)
        
        if best_val_loss < best_val_loss_NPP:
            best_val_loss_NP = best_val_loss
            

        test_loss = evaluate_model(autoencoder, test_loader, input_channel, device, partial_label_GP=False, partial_percent=partial_percent)
        GP_test_loss = evaluate_model(autoencoder, test_loader, input_channel, device, partial_label_GP=True, partial_percent=partial_percent)
        print(f"NPP sigma={sigma} Test loss:{test_loss:.3f}, GP Test loss:{GP_test_loss:.3f}")
        test_losses_vs_sigma_npp_true.append(test_loss)
        GP_test_losses_vs_sigma_npp_true.append(GP_test_loss)
        count += 1

        test_losses_npp_true.append(test_losses_vs_sigma_npp_true)
        GP_test_losses_npp_true.append(GP_test_losses_vs_sigma_npp_true)
    return GP_test_losses_npp_true, test_losses_npp_true, experiment_id

    
# Function to run the pipeline and save data
def run_and_save_pipeline(sigmas, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader, input_channel, epochs, val_every_epoch, learning_rates, config, num_runs, device):
    # Run the pipeline
    GP_test_loss_npp_true, test_loss_npp_true, experiment_id = run_pipeline_ci(sigmas, num_kernels_encoder, num_kernels_decoder, train_loader, val_loader, test_loader, input_channel, epochs, val_every_epoch, learning_rates, config, device, num_runs)
    partial_percent = config['partial_percent']
    # Run final testing
    model = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)
    # MSE
    model.load_state_dict(torch.load(f'./history/{experiment_id}/model_NP.pth'))
    best_NP_test_loss = evaluate_model(model, test_loader, input_channel, device, partial_label_GP=False, partial_percent=partial_percent)
    GP_best_NP_test_loss = evaluate_model(model, test_loader, input_channel, device, partial_label_GP=True, partial_percent=partial_percent)
    print("start saving!")
    # Save the data
    save_loss(test_loss_npp_true, f'./history/{experiment_id}/test_loss_npp_true.npy')
    save_loss(test_loss_npp_false, f'./history/{experiment_id}/test_loss_npp_false.npy')
    save_loss(GP_test_loss_npp_true, f'./history/{experiment_id}/GP_test_loss_npp_true.npy')
    f = open(f"./history/{experiment_id}/results.txt", "w")
    f.write(f"Results {experiment_id}:\n MSE: {best_MSE_test_loss} | NPP (sigma {best_sigma_NPP}): {best_NPP_test_loss} , GP: {GP_best_NPP_test_loss}")
    f.close()
    print("saved")
    return (GP_test_loss_npp_true, test_loss_npp_true, test_loss_npp_false), experiment_id


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Datasets and hyperparameters
    parser.add_argument("--dataset", type=str, default="PinMNIST", help="Dataset name")
    parser.add_argument("feature", type=str, default="AE", help="feature from 'DDPM' or 'DDPM'")
    parser.add_argument("--mode", type=str, default="mesh", help="mode for 'mesh' or 'random'")
    parser.add_argument("--n", type=int, default=100, help="Value for 'n'")
    parser.add_argument("--d", type=int, default=10, help="Value for 'd'")
    parser.add_argument("--n_pins", type=int, default=500, help="Value for 'n_pins'")
    parser.add_argument("--partial_percent", type=float, default=1.00, help="Value for partially showing the labels (0 to 1 range)")
    parser.add_argument("--r", type=int, default=3, help="Value for 'r'")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--val_every_epoch", type=int, default=5, help="Number of epochs in between validations")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of different trainings to do per model and sigma")

    # List of sigma values
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 0.2, 0.5, 1, 2, 5], help="List of sigma values to test")

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
    feature_extracted = args.features_extracted
    n = args.n
    mesh = args.mesh
    d = args.d
    n_pins = args.n_pins
    r = args.r
    partial_percent = args.partial_percent

    # Set your hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    sigmas = args.sigmas  # Set the sigma values you want to test
    num_kernels_encoder = args.num_encoder
    num_kernels_decoder = args.num_decoder
    learning_rate = args.learning_rate
    val_every_epoch = args.val_every_epoch
    num_runs = args.num_runs
    
    config = vars(args)
    config['seed'] = seed
    
    input_channel = 1 if dataset == "PinMNIST" else 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_extracted:
        folder = f"{dataset}_ddpm"
    else:
        folder = f"{dataset}"
    
    if dataset == "PinMNIST":
        if mesh:
            data_folder = f"./data/{folder}/{n}images_mesh_{d}step_{28}by{28}pixels_{r}radius_{seed}seed"
        else:
            data_folder = f"./data/{folder}/{n}images_random_{n_pins}pins_{28}by{28}pixels_{r}radius_{seed}seed"
    
    if dataset == "Synthetic":
        if mesh:
            data_folder = f"./data/{folder}/{n}images_{d}_distanced_grid_pins_{seed}seed/"
        else:
            data_folder = f"./data/{folder}/{n}images_upto{n_pins}pins_{seed}seed/"

    transform = transforms.Compose([
        ToTensor(),         # Convert to tensor (as you were doing)
        Resize()  # Resize to 100x100
    ])
    
    transformed_dataset = PinDataset(csv_file=f"{data_folder}/pins.csv",
                                          root_dir=f"{data_folder}/images/",
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Find best learning rate
    model = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)

    if (config['experiment_id'] == 0): # Training
        
        model = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)
        optimizer = torch.optim.Adam(np_img.parameters(), learning_rate)
        lr_finder_NP =NPLRFinder(model, optimizer, device=device)
        lr_finder_NP.find_lr(train_loader,input_channel=input_channel, start_lr=1e-5, end_lr=1, num_iter=20)
        best_lr_NP = lr_finder_NP.find_best_lr()
        print(f"Best Learning Rate for NP: {best_lr_NP}")
        
        config['best_lr'] = best_lr_NP
        # Run and save the pipeline data
        loss_vs_sigma_data, experiment_id = run_and_save_pipeline(train_loader, val_loader, test_loader,\
                                                   input_channel, epochs, val_every_epoch, config, num_runs, device)
        
        
        # Plot and save the plot using the saved data
        plot_and_save(loss_vs_sigma_data, sigmas, dataset, learning_rate, results_dir=f'./history/{experiment_id}')
        
    else: # Testing
        experiment_id = config['experiment_id']
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