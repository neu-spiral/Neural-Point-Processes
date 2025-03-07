#!/usr/bin/env python
import sys
import os
# Adjust the Python path to include the directory where /tools is located.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torchvision
import torch
import random
from tools.data_utils import gen_pins, gen_mesh_pins, count_pins, save_data
from tools.plot_utils import visualize_pins, plot_label_pin
from tools.models import DDPM, UNet
import matplotlib.pyplot as plt

def extract_with_ddpm(data, n_steps, store_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DDPM Model
    ddpm = DDPM(UNet(), n_steps=n_steps, device=device)
    ddpm.load_state_dict(torch.load(store_path, map_location=device))
    
    # Move input to the same device as the model
    x0 = data.unsqueeze(0).unsqueeze(0).to(device)
    n = len(x0)

    # Picking some noise for each image in the batch, a timestep, and the respective alpha_bars
    eta = torch.randn_like(x0).to(device)
    t = torch.randint(0, n_steps, (n,)).to(device)

    # Compute the noisy image based on x0 and the timestep (forward process)
    noisy_imgs = ddpm(x0, t, eta)

    # Get the model’s estimation of noise based on the images and timestep
    eta_theta, concatenated_feature_map = ddpm.backward(noisy_imgs, t.reshape(n, -1))

    return concatenated_feature_map

def PinMNIST(mnist, n, n_pin_max, r, fixed_pins=True, mesh=False, d=1, store_path=f"./history/ddpm_MNIST.pt", n_steps=200):
    """
    Generate PinMNIST dataset by selecting n unique samples from the original MNIST and adding pins.

    Args:
        mnist (dataset): Image dataset from MNIST.
        n (int): Number of unique samples to select.
        n_pin_max (int): Maximum number of pins to generate for each sample.
        r (int): Radius for counting adjacent ones around each pin.
        fixed_pins (bool): If True, use a fixed number of pins for all samples.
        mesh (bool): If True, generate pins on a mesh grid based on the 'd' parameter.
        d (int): Spacing between pins when using mesh.
        store_path (str): Path to the saved DDPM model.
        n_steps (int): Number of diffusion steps.

    Returns:
        ddpm_images (list): List of processed images using DDPM.
        pins (list): List of lists of (x, y) coordinate tuples.
        labels (list): List of count labels for each image.
    """
    ddpm_images = []
    pins = []
    labels = []
    
    for i in range(n):
        # Get the image from the dataset
        mnist_image = mnist[i]
        
        # Generate pin locations based on the mode selected
        if mesh:
            pin_locations = gen_mesh_pins(mnist_image, d)
        else:
            n_pins = n_pin_max if fixed_pins else random.randint(1, n_pin_max)
            pin_locations = gen_pins(mnist_image, n_pins)
        
        # Count the adjacent ones around each pin
        label = count_pins(mnist_image.numpy(), pin_locations, r)
        mnist_image = extract_with_ddpm(mnist_image, n_steps, store_path)
        mnist_image = mnist_image.squeeze()
        ddpm_images.append(mnist_image)
        pins.append(pin_locations)
        labels.append(label)

    return ddpm_images, pins, labels

if __name__ == "__main__":
    # Load the original MNIST dataset
    original_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    
    n = 1000         # Set number of images
    n_pins = 10
    fixed_pins = True
    mesh = False
    d = 3
    r = 3
    seed = 4
    random.seed(seed)

    # Process the images: normalize and adjust pixel values
    images = []
    for i in range(n):
        mnist_image = original_mnist.data[i] / 255.0  # Normalize to [0, 1]
        mnist_image = 2 * (mnist_image - 0.5)           # Normalize to [-1, 1]
        images.append(mnist_image)
    
    # Shuffle the MNIST dataset data for randomness
    random.shuffle(original_mnist.data)
    
    ddpm_images, pins, labels = PinMNIST(images, n, n_pins, r, mesh=mesh, d=d)

    # Save data to folders
    for folder in ["PinMNIST", "PinMNIST_ddpm"]:
        if mesh:
            data_folder = f"./data/{folder}/mesh_{d}step_{28}by{28}pixels_{r}radius_{seed}seed/"
        else:
            data_folder = f"./data/{folder}/random_fixed{fixed_pins}_{n_pins}pins_{28}by{28}pixels_{r}radius_{seed}seed/"
        os.makedirs(data_folder, exist_ok=True)
        if folder.split("_")[-1] == "ddpm":
            save_data(ddpm_images, pins, labels, f"./data/{folder}/", data_folder)
        else:
            save_data(images, pins, labels, f"./data/{folder}/", data_folder)
