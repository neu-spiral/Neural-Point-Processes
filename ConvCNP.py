import torch
import torch.optim as optim
import os
import json
import time
from tools.data_utils import *
from tools.optimization import EarlyStoppingCallback, evaluate_model
from tools.ConvCNPtrain import ConvCNPTrainer
from tabulate import tabulate
import numpy as np
import argparse
from functools import partial

from npf import CNPPFLoss, CNPFLoss
from npf import ConvCNP, GridConvCNP
from npf.architectures import CNN, MLP, ResConvBlock, SetConv, discard_ith_arg, ConvBlock
from npf.utils.helpers import CircularPad2d, make_abs_conv, make_padded_conv


def run_pipeline_ci_np(train_loader, val_loader, test_loader, epochs, val_every_epoch, config, device, num_runs=3, print_freq=2):
    test_losses = []
    experiment_id = int(time.time())
    best_val_loss_NP = float('inf')
    experiment_id = config['experiment_id']
    lr = config['learning_rate']

    if not os.path.exists(f'./history/ConvCNP/{experiment_id}'):
        os.makedirs(f'./history/ConvCNP/{experiment_id}')
    with open(f"./history/ConvCNP/{experiment_id}/config_np.json", "w") as outfile: 
        json.dump(config, outfile)
        
    global_val_loss = float('inf')
    
    for run in range(num_runs):
        count = 0
        # create model
        model_2d = partial(
            GridConvCNP,
            x_dim=1,  # for gridded conv it's the mask shape
            y_dim=1,
            CNN=partial(
                CNN,
                Conv=torch.nn.Conv2d,
                Normalization=torch.nn.BatchNorm2d,
                n_blocks=5,
                kernel_size=9,
                **CNN_KWARGS,
            ),
            **model_KWARGS,
        )
        
        trainer_kwargs = {
        'optimizer_params': {'lr': trainer_KWARGS['lr']},
        'scheduler_params': {'step_size': trainer_KWARGS['decay_lr'], 'gamma': 0.8},
        'num_epochs': config["epochs"]
        }

        trainer = ConvCNPTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_2d=model_2d,
            criterion=CNPPFLoss(),
            experiment_id=experiment_id,
            early_stop_patience=5,  # Patience of 5 epochs for early stopping
            **trainer_kwargs
        )

        trainer.train()

        # Save best model based on validation performance
        if trainer.best_val_loss <= global_val_loss:
            global_val_loss = trainer.best_val_loss 
            torch.save(trainer.model.state_dict(), f'./history/ConvCNP/{experiment_id}/best_model_global.pt')
        count += 1

    return trainer


def run_and_save_pipeline_np(train_loader, val_loader, test_loader, epochs, val_every_epoch, config, num_runs, device):
    test_partial_percents = [0.25, 0.5, 0.75, 1]
    test_losses = []
    r2_list = []
    global_r2_list = []
    table = []
    table.append(['Dataset', 'Mode', 'd', 'n_pins', 'LR', 'PLP', 'MSE error', 'R2', 'global_R2'])
    trainer = run_pipeline_ci_np(train_loader, val_loader, 
                    test_loader, epochs, val_every_epoch, config, device, num_runs)

    for partial_percent in test_partial_percents:
        test_loss, r2, global_r2 = trainer.evaluate(partial_percent=partial_percent)
        print(f"pp: {partial_percent} MSE loss: {test_loss} R2 score: {r2} Global r2 score: {global_r2}")
        table.append([config["dataset"], config["mode"], config["d"], config["n_pins"], config['learning_rate'], partial_percent, test_loss, r2, global_r2])
        test_losses.append(test_loss)
        r2_list.append(r2)
        global_r2_list.append(global_r2)
    
    table = tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=True)
    print(table)
    with open('./history/ConvCNP/table.txt', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write(table + '\n')
        f.write('\n')
    return test_losses, r2_list, global_r2_list


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
    parser.add_argument('--num_runs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    ## model params
    R_DIM = 128
    model_KWARGS = dict(
        r_dim=R_DIM,
        Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
            partial(MLP, n_hidden_layers=1, hidden_size=R_DIM), i=0
        ),
    )

    CNN_KWARGS = dict(
        ConvBlock=ConvBlock, #ResConvBlock
        is_chan_last=True,  # all computations are done with channel last in our code
        # n_conv_layers=1,  # layers per block
    )


    args = parse_args()
    # Set the random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = vars(args)
    
    # trainer params
    trainer_KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        criterion=CNPPFLoss,
            chckpnt_dirname="history/ConvCNP/",
        device=None,
        lr=config["learning_rate"],
        decay_lr=10,
        seed=123,
        batch_size=32,
    )
    
    timestamp = int(time.time())
    config["experiment_id"] = f"{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, train_loader, val_loader, eval_loader = data_prepare(config)

    start_time = time.time()
    test_losses, r2, global_r2 = run_and_save_pipeline_np(
        train_loader, val_loader, eval_loader, args.epochs, args.val_every_epoch, config, args.num_runs, device)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(config["experiment_id"], "Time elapsed:", elapsed_time, "seconds")
    with open('./history/ConvCNP/table.txt', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write(f"Time elapsed: {elapsed_time} seconds"+ '\n')
        f.write('\n')