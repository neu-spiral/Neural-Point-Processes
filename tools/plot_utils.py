import numpy as np
from tools.data_utils import count_pins, gen_mesh_pins

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