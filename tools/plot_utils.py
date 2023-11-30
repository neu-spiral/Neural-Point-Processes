import numpy as np
from tools.data_utils import count_pins, gen_mesh_pins
import matplotlib.pyplot as plt
import os
import numpy as np


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
    
    
def plot_and_save(loss_vs_sigma_data, sigmas, dataset, learning_rate, model_name="Auto encoder"):
    # Unpack the data
    test_loss_npp_true, test_loss_npp_false = loss_vs_sigma_data
    test_loss_npp_false = [test_loss_npp_false for i in range(len(sigmas))]

    # Calculate mean and confidence intervals for NPP=True runs
    mean_test_loss_npp_true = np.mean(test_loss_npp_true, axis=0)
    ci_test_loss_npp_true = 1.96 * np.std(test_loss_npp_true, axis=0) / np.sqrt(len(test_loss_npp_true))

    # Duplicate NPP=False values for plotting
    mean_test_loss_npp_false = np.mean(test_loss_npp_false, axis=1)
    ci_test_loss_npp_false = 1.96 * np.std(test_loss_npp_false, axis=1) / np.sqrt(len(test_loss_npp_false))

    # Plot mean and confidence intervals for NPP=True
    plt.plot(sigmas, mean_test_loss_npp_true, marker='o', label='NPP=True', color='blue')

    # Plot mean and confidence intervals for duplicated NPP=False
    plt.plot(sigmas, mean_test_loss_npp_false, color='red', linestyle='--', label='NPP=False')

    # Fill between for NPP=True with blue color
    plt.fill_between(sigmas, mean_test_loss_npp_true - ci_test_loss_npp_true, mean_test_loss_npp_true + ci_test_loss_npp_true, color='blue', alpha=0.2)

    # Fill between for NPP=False with red color
    plt.fill_between(sigmas, mean_test_loss_npp_false - ci_test_loss_npp_false, mean_test_loss_npp_false + ci_test_loss_npp_false, color='red', alpha=0.2)

    plt.xlabel('Sigma')
    plt.ylabel('Test Loss')
    plt.title(f'Test Loss vs. Sigma:{dataset} dataset with {model_name}')
    plt.legend()

    # Create a directory to save the results if it doesn't exist
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate a filename based on parameters in the title
    filename = f"test_loss_vs_sigma_{dataset}_{model_name}_lr_{learning_rate}.png"
    filepath = os.path.join(results_dir, filename)

    # Save the plot
    plt.savefig(filepath)

    # Show the plot
    plt.show()