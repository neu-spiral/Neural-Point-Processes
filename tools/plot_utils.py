import numpy as np
from tools.data_utils import count_pins, gen_mesh_pins
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import imageio
import einops

def visualize_pins(image, pins, color_map='viridis'):
    """
    Visualize pin locations with color mapping for the values and displaying the image using plt.imshow.

    Args:
    image (numpy.ndarray): Binary image.
    pins (list of tuples): List of (x, y) coordinate tuples representing pin locations.
    color_map (str): The colormap to use for displaying the values.

    """
    # Create a copy of the image to avoid modifying the original
    image_with_pins = np.zeros_like(image.detach().cpu().numpy().copy())

    # Set the locations with pins to 1
    for pin in pins:
        x, y = pin
        image_with_pins[x, y] = 1
    
    # Display the image with pins and values using plt.imshow
    # plt.imshow(image_with_pins, cmap=color_map)
    # plt.colorbar()
    # plt.axis('off')  # Hide axes
    # plt.show()
    
    return image_with_pins


def plot_label_pin(image, pins, labels):
    """
    Assign values according to labels to pixels associated with the pins and plot the resulting image.

    Args:
    image (numpy.ndarray): Input image as a NumPy array.
    pins (list): List of (x, y) coordinate tuples representing pin locations.
    labels (list): List of labels to assign to the pins.

    Returns:
    labeled_image (numpy.ndarray): Image with pin values assigned based on labels.

    This function takes an input image and assigns values from the 'labels' list to the pixels associated with the 'pins' locations. It returns the resulting image with updated pixel values and also plots the image.
    """

    # Create an all-zero labeled image with the same shape as the input image
    labeled_image = np.zeros_like(image)

    # Loop through the pins and labels to assign values
    for pin, label in zip(pins, labels):
        x, y = pin
        labeled_image[x, y] = label

    # Plot the resulting image
    # plt.imshow(labeled_image, cmap='gray')
    # plt.title("Image with Labels Assigned to Pins")
    # plt.show()

    return labeled_image


def plot_all(image, r):
    pin_locations = gen_mesh_pins(image, 1)
    label = count_pins(image.numpy(), pin_locations, r)
    count_image = plot_label_pin(image, pin_locations, label)
    return count_image


def plot_loss(train_losses, val_losses, val_every_epoch, NPP, sigma, dataset, learning_rate, num_kernels_encoder, num_kernels_decoder, save_dir="./results/plots"):
    # Create a directory for saving plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the filename based on parameters
    filename = f"loss_plot_NPP_{NPP}_sigma_{sigma}_dataset_{dataset}_lr_{learning_rate}_encoder_{num_kernels_encoder}_decoder_{num_kernels_decoder}.png"
    save_path = os.path.join(save_dir, filename)

    # Plot the train and validation loss
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(val_every_epoch, len(train_losses), val_every_epoch), val_losses, '--', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MSE Loss')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    
def plot_and_save(loss_vs_sigma_data, sigmas, dataset, learning_rate, model_name="Auto encoder", results_dir = './results'):
    # Unpack the data
    GP_test_loss_npp_true, test_loss_npp_true, test_loss_npp_false = loss_vs_sigma_data
    test_loss_npp_false = [test_loss_npp_false for i in range(len(sigmas))]

    # Calculate mean and confidence intervals for NPP=True runs
    mean_test_loss_npp_true = np.mean(test_loss_npp_true, axis=0)
    ci_test_loss_npp_true = 1.96 * np.std(test_loss_npp_true, axis=0) / np.sqrt(len(test_loss_npp_true))
    
    # Calculate mean and confidence intervals for GP NPP=True runs
    mean_GP_test_loss_npp_true = np.mean(GP_test_loss_npp_true, axis=0)
    ci_GP_test_loss_npp_true = 1.96 * np.std(GP_test_loss_npp_true, axis=0) / np.sqrt(len(GP_test_loss_npp_true))

    # Duplicate NPP=False values for plotting
    mean_test_loss_npp_false = np.mean(test_loss_npp_false, axis=1)
    ci_test_loss_npp_false = 1.96 * np.std(test_loss_npp_false, axis=1) / np.sqrt(len(test_loss_npp_false))

    # Plot mean and confidence intervals for NPP=True
    plt.plot(sigmas, mean_test_loss_npp_true, marker='o', label='NPP', color='blue')
    
    # Plot mean and confidence intervals for GP NPP=True
    plt.plot(sigmas, mean_GP_test_loss_npp_true, marker='*', label='GP NPP', color='yellow')

    # Plot mean and confidence intervals for duplicated NPP=False
    plt.plot(sigmas, mean_test_loss_npp_false, color='red', linestyle='--', label='MSE')

    # Fill between for NPP=True with blue color
    plt.fill_between(sigmas, mean_test_loss_npp_true - ci_test_loss_npp_true, mean_test_loss_npp_true + ci_test_loss_npp_true, color='blue', alpha=0.2)
    
    # Fill between for GP NPP=True with blue color
    plt.fill_between(sigmas, mean_GP_test_loss_npp_true - ci_GP_test_loss_npp_true, mean_GP_test_loss_npp_true + ci_GP_test_loss_npp_true, color='yellow', alpha=0.2)

    # Fill between for NPP=False with red color
    plt.fill_between(sigmas, mean_test_loss_npp_false - ci_test_loss_npp_false, mean_test_loss_npp_false + ci_test_loss_npp_false, color='red', alpha=0.2)

    plt.xlabel('Sigma')
    plt.ylabel('Test Loss')
    plt.title(f'Test Loss vs. Sigma:{dataset} dataset with {model_name}')
    plt.legend()

    # Create a directory to save the results if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Generate a filename based on parameters in the title
    filename = f"test_loss_vs_sigma_{dataset}_{model_name}_lr_{learning_rate}.png"
    filepath = os.path.join(results_dir, filename)

    # Save the plot
    plt.savefig(filepath)

    # Show the plot
    plt.show()
    

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

    
def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break
        

def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta, con_feature = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)
                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            if frame.shape[-1] == 1:
                frame = frame.repeat(3, axis=-1)
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frame[-1].repeat(3, axis=-1))
    return x