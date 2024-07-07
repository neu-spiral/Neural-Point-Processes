import torchvision
import torch
import sys, shutil
sys.path.append('../..')
import random
from tools.data_utils import gen_pins, gen_mesh_pins, count_pins, save_data
from tools.plot_utils import visualize_pins, plot_label_pin
from tools.models import DDPM, UNet
import os
import matplotlib.pyplot as plt

def extract_with_ddpm(ddpm, data, n_steps, store_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move input to the same device as the model
    x0 = data.unsqueeze(0).unsqueeze(0).to(device)
    n = len(x0)

    # Picking some noise for each of the images in the batch, a timestep, and the respective alpha_bars
    eta = torch.randn_like(x0).to(device)
    t = torch.randint(0, n_steps, (n,)).to(device)

    # Computing the noisy image based on x0 and the time-step (forward process)
    noisy_imgs = ddpm(x0, t, eta)

    # Getting model estimation of noise based on the images and the time-step
    eta_theta, concatenated_feature_map = ddpm.backward(noisy_imgs, t.reshape(n, -1))

    return concatenated_feature_map

def PinMNIST(mnist, n, n_pin_max, r, fixed_pins=True, mesh=False, d=1):
    """
    Generate PinMNIST dataset by selecting n unique samples from the original MNIST and adding pins.

    Args:
    original_mnist(dataset): Fixed image dataset to generate all labels
    n (int): Number of unique samples to select from the original MNIST.
    n_pin_max (int): Maximum number of pins to generate for each sample.
    r (int): Radius for counting adjacent ones around each pin.
    fixed_pins (bool): If True, generates a fixed number of pins (n_pin_max) for all samples. If False, generates a random number of pins for each sample.
    mesh (bool): If True, generates pins on a mesh grid based on the 'd' parameter.
    d (int): Spacing between pins when 'mesh' is True.

    Returns:
    images (list): List of binary images.
    pins (list): List of lists of (x, y) coordinate tuples.
    count_images (list): List of count images where counts are the counted values at the pin locations.
    """
    # Load the original MNIST dataset
    # original_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)

    # Shuffle the original MNIST dataset
    # random.shuffle(original_mnist.data)

    # Initialize lists to store images, pins, and count images
    pins = []
    labels = []
    # !!!!!!!!!!!!!!!!!! MAKE IT BATCH!! FASTER!!
    for i in range(n):
        # Get the binary image from the shuffled MNIST dataset
        mnist_image = mnist[i]

        # Determine the number of pins to generate
        if mesh:
            pin_locations = gen_mesh_pins(mnist_image, d)
        else:
            if fixed_pins:
                n_pins = n_pin_max
            else:
                n_pins = random.randint(1, n_pin_max)
            # Generate random pins for the binary image
            pin_locations = gen_pins(mnist_image, n_pins)

        # Count the adjacent ones for the pins using the count_pins function
        label = count_pins(mnist_image.numpy(), pin_locations, r)
        pins.append(pin_locations)
        labels.append(label)

    return pins, labels

def main():

    seed = 4
    random.seed(seed)

    original_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)

    # Shuffle the original MNIST dataset
    random.shuffle(original_mnist.data)

    # Set number of images n 
    n=1000 
    # Initialize lists to store images, pins, and count images
    images = []
    ddpm_images = []
    n_steps=200
    # Load DDPM Model
    store_path = f"../../history/ddpm_MNIST.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(UNet(), n_steps=n_steps, device=device)
    ddpm.load_state_dict(torch.load(store_path, map_location=device))

    for i in range(n):
        # Get the binary image from the shuffled MNIST dataset
        mnist_image = original_mnist.data[i] / 255.0  # Normalize to [0, 1]
        mnist_image = 2 * (mnist_image - 0.5) # Normalize to [-1, 1]
        images.append(mnist_image)
        mnist_image = extract_with_ddpm(ddpm, mnist_image, n_steps, store_path)
        mnist_image = mnist_image.squeeze()
        ddpm_images.append(mnist_image)

    # Set hyperparameters to generate other versions of the PinMNIST dataset by changing the n_pins, d and r
    densities = ['sparse', 'dense']
    layouts = ['grid', 'random']
    n_pins = [10, 100]
    d = [10, 3]
    r = 3

    for i, density in enumerate(densities):
        for j, l in enumerate(layouts):
            mesh = layouts[j] == 'grid'
            pins, labels = PinMNIST(images, n, n_pins[i], r, mesh=mesh, d=d[i])
            print(f"Save data for {density} and {l}")
            for folder in ["PinMNIST", "PinMNIST_ddpm"]:
                if mesh:
<<<<<<< HEAD
                    data_folder = f"../../data/{folder}/mesh_{d[i]}step_{28}by{28}pixels_{r}radius_{seed}seed/"
                else:
                    data_folder = f"../../data/{folder}/random_fixedTrue_{n_pins[i]}pins_{28}by{28}pixels_{r}radius_{seed}seed/"
=======
                    data_folder = f"../../data/{folder}/mesh_{d[i]}step/"
                else:
                    data_folder = f"../../data/{folder}/random_{n_pins[i]}pins/"
>>>>>>> c2590415a42d835372d01ae92c8a3d8eed06d3ed
                os.makedirs(data_folder, exist_ok=True)
                if (folder.split("_")[-1] == "ddpm"):
                    save_data(ddpm_images, pins, labels, f"../../data/{folder}/", data_folder)
                else:
                    save_data(images, pins, labels, f"../../data/{folder}/", data_folder)
            
    # Remove MNIST data downloaded from torch
    shutil.rmtree('./data')

if __name__ == "__main__":
    main()