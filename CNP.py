import torch
import torch.optim as optim
import os
import json
import time
from tools.data_utils import *
from tools.optimization import EarlyStoppingCallback, evaluate_model
from tools.NPmodels import *
from tools.NPtrain import *
from tabulate import tabulate
import numpy as np
import argparse


def run_pipeline_ci_np(train_loader, val_loader, test_loader, epochs, val_every_epoch, config, np_config, device, num_runs=3, print_freq=2):
    test_losses = []
    experiment_id = int(time.time())
    best_val_loss_NP = float('inf')
    experiment_id = config['experiment_id']

    r_dim = np_config.r_dim
    h_dim = np_config.h_dim
    z_dim = np_config.z_dim
    lr = config['learning_rate']

    if not os.path.exists(f'./history/CNP/{experiment_id}'):
        os.makedirs(f'./history/CNP/{experiment_id}')
    with open(f"./history/CNP/{experiment_id}/config_np.json", "w") as outfile: 
        json.dump(config, outfile)
        
    global_val_loss = float('inf')
    
    for run in range(num_runs):
        count = 0
        GP_test_losses = []
        
        early_stopping = EarlyStoppingCallback(patience=5, min_delta=0.001)
        model = NeuralProcessImg(r_dim, z_dim, h_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        np_trainer = NeuralProcessTrainer(device, model, optimizer, early_stopping, experiment_id, print_freq=print_freq)

        np_trainer.train(train_loader, val_loader, epochs)
        if np_trainer.best_val_loss <= global_val_loss:
            global_val_loss = np_trainer.best_val_loss 
            torch.save(np_trainer.neural_process.state_dict(), f'./history/CNP/{experiment_id}' + f'/best_{config["dataset"]}_np.pt')
        count += 1

    return experiment_id


def run_and_save_pipeline_np(train_loader, val_loader, test_loader, epochs, val_every_epoch, config, np_config, num_runs, device):
    test_partial_percents = [0.25, 0.5, 0.75, 1]
    test_losses = []
    r2_list = []
    global_r2_list = []
    table = []
    table.append(['Dataset', 'Mode', 'd', 'n_pins', 'LR', 'PLP', 'MSE error', 'R2', 'global_R2'])
    experiment_id = run_pipeline_ci_np(train_loader, val_loader, 
                    test_loader, epochs, val_every_epoch, config, np_config, device, num_runs)
    
    model = NeuralProcessImg(np_config.r_dim, np_config.z_dim, np_config.h_dim).to(device)
    model.load_state_dict(torch.load(f'./history/CNP/{experiment_id}/best_{config["dataset"]}_np.pt'))
    for partial_percent in test_partial_percents:
        test_loss, r2, global_r2 = evaluate_np(model, test_loader, device, partial_percent=partial_percent)
        print(f"pp: {partial_percent} MSE loss: {test_loss} R2 score: {r2} Global r2 score: {global_r2}")
        table.append([config["dataset"], config["mode"], config["d"], config["n_pins"], config['learning_rate'], partial_percent, test_loss, r2, global_r2])
        test_losses.append(test_loss)
        r2_list.append(r2)
        global_r2_list.append(global_r2)
    
    table = tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=True)
    print(table)
    with open('./history/CNP/table.txt', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write(table + '\n')
        f.write('\n')
    return test_losses, r2_list, global_r2_list, experiment_id


class NP_config:
    def __init__(self):
        self.r_dim = 512
        self.h_dim = 512
        self.z_dim = 512
        self.lr = 4e-5
        self.epochs = 100


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Process Training')
    parser.add_argument('--dataset', type=str, default="Synthetic")
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--mode', type=str, default="random")
    parser.add_argument('--feature', type=str, default="AE")
    parser.add_argument('--modality', type=str, default="PS-RGBNIR")
    parser.add_argument('--d', type=int, default=36)
    parser.add_argument('--n_pins', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--partial_percent', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_every_epoch', type=int, default=10)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np_config = NP_config()
    # Set the random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = vars(args)
    timestamp = int(time.time())
    config["experiment_id"] = f"{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, eval_loader = data_prepare(config)

    start_time = time.time()
    test_losses, r2, global_r2, experiment_id = run_and_save_pipeline_np(
        train_loader, val_loader, eval_loader, args.epochs, args.val_every_epoch, config, np_config, args.num_runs, device)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time elapsed:", elapsed_time, "seconds")