import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random
import os
import csv
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from skimage import io, transform


#Pin generation
def gen_pins(image, n):
    """
    Generate a list of n unique random (x, y) coordinate pins within the dimensions of the image.

    Args:
    image (numpy.ndarray): Input image.
    n (int): Number of unique random pins to generate.

    Returns:
    list of tuples: List of (x, y) coordinate tuples representing unique random pins.
    """
    h, w = image.shape  # Get the dimensions of the image
    total_pixels = h * w

    # Ensure that n is not greater than the total number of pixels
    n = min(n, total_pixels)

    # Generate a list of all possible (x, y) coordinates
    all_possible_pins = [(x, y) for x in range(h) for y in range(w)]

    # Shuffle the list to randomize the order
    random.shuffle(all_possible_pins)

    # Select the first n coordinates to ensure uniqueness
    unique_pins = all_possible_pins[:n]

    return unique_pins


def gen_mesh_pins(image, d):
    """
    Generate a list of mesh pins' coordinates within an image.

    Args:
    - image (numpy.ndarray): The input image as a NumPy array.
    - d (int): The desired distance between mesh pins.

    Returns:
    - mesh_pins (list of tuples): A list of (x, y) coordinates representing mesh pins.

    This function takes an image and a specified distance 'd' as input and generates a grid of mesh pins within the image. The mesh pins are placed at regular intervals based on the given distance 'd'. The resulting list contains coordinate tuples, each representing the location of a mesh pin within the image.

    Example usage:
    image_path = 'path_to_your_image.jpg'
    distance = 10  # Set the desired distance between mesh pins
    mesh_pins = gen_mesh_pins(image, distance)
    """

    # Get the height and width of the image
    height, width = image.shape

    # Initialize an empty list to store mesh pins
    mesh_pins = []

    # Iterate through the image and add mesh pins
    for y in range(0, height, d):
        for x in range(0, width, d):
            mesh_pins.append((x, y))

    return mesh_pins


#Value counting
def count_within_radius(image, center_x, center_y, radius):
    """
    Count the number of pixels with value larger than 0 within a specified radius around a given pixel.

    Args:
    image (numpy.ndarray): Binary image where 1 represents the object and 0 the background.
    center_x (int): X-coordinate of the center pixel.
    center_y (int): Y-coordinate of the center pixel.
    radius (int): The radius within which to count adjacent ones.

    Returns:
    int: The count of adjacent ones within the specified radius around the center pixel.
    """
    h, w = image.shape
    count = 0

    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            if 0 <= x < h and 0 <= y < w:
                if image[x, y] > 0:
                    count += image[x, y]

    return count


import numpy as np

def count_all(binary_images, radius=1):
    """
    Count the adjacent ones within a specified radius around each pixel in a batch of binary images or a single binary image.

    Args:
    binary_images (numpy.ndarray): A 2D, 3D, or 4D array of binary images. If 2D, it's a single binary image with shape (height, width). If 3D, it's a single binary image with shape (1, height, width). If 4D, it's a batch of binary images with shape (batch_size, 1, height, width).
    radius (int): The radius within which to count adjacent ones.

    Returns:
    numpy.ndarray: A 2D, 3D, or 4D array with the same shape as the input binary_images, containing counts for each pixel.
    """
    if binary_images.ndim == 2:  # Single 2D image case
        height, width = binary_images.shape
        count_image = np.zeros_like(binary_images, dtype=np.uint8)

        for x in range(height):
            for y in range(width):
                count = count_within_radius(binary_images, x, y, radius)
                count_image[x, y] = count

        return count_image

    elif binary_images.ndim == 3:  # Single 3D image or single image with a squeezed channel dimension
        if binary_images.shape[0] == 1:
            binary_image = binary_images[0]
            height, width = binary_image.shape
            count_image = np.zeros_like(binary_image, dtype=np.uint8)

            for x in range(height):
                for y in range(width):
                    count = count_within_radius(binary_image, x, y, radius)
                    count_image[x, y] = count

            return count_image
        else:
            raise ValueError("Input binary_images with 3 dimensions must have a batch size of 1.")

    elif binary_images.ndim == 4:  # Batched images case
        batch_size, _, height, width = binary_images.shape
        count_images = np.zeros_like(binary_images, dtype=np.uint8)

        for b in range(batch_size):
            binary_image = binary_images[b, 0]  # Extract the single-channel image

            count_image = np.zeros_like(binary_image, dtype=np.uint8)

            for x in range(height):
                for y in range(width):
                    count = count_within_radius(binary_image, x, y, radius)
                    count_image[x, y] = count

            count_images[b, 0] = count_image  # Store the count image in the batch

        return count_images

    else:
        raise ValueError("Input binary_images must be either 2D, 3D, or 4D.")


def count_pins(binary_image, pins, radius):
    """
    Count the number of adjacent ones within a specified radius around a selection of points.

    Args:
    binary_image (numpy.ndarray): Binary image where 1 represents the object and 0 the background.
    pins (list of tuples): List of (x, y) coordinate tuples specifying the points of interest.
    radius (int): The radius within which to count adjacent ones.

    Returns:
    numpy.ndarray: An image with counts of adjacent ones within the specified radius around the points.
    """
    # Initialize an empty count image with the
    
    # same shape as the binary image
    count_image = np.zeros_like(binary_image, dtype=np.uint8)
    counted_pins = []
    for pin in pins:
        x, y = pin
        count = count_within_radius(binary_image, x, y, radius)
        counted_pins.append(count)  

    return counted_pins


#save data
def save_data(images, pins, labels, images_directory, output_directory):
    """
    Save multi-channel images, pin coordinates, and count labels to CSV, and save images to files in the specified directory.

    Args:
    images (list): List of multi-channel images.
    pins (list): List of lists of (x, y) coordinate tuples representing pin locations.
    labels (list): List of labels counted at the pin locations.
    images_directory (str): The directory where images will be saved.
    output_directory (str): The directory where data, and count labels will be saved.

    This function saves multi-channel images, pin coordinates, and count labels to CSV files, and also saves the images as files in the 'images' subdirectory within the 'output_directory'. Each image is saved as a numbered PNG or NPY file (e.g., "0.png", "1.npy") within the 'images' subdirectory. The CSV file 'pins.csv' contains columns for image filenames, pin coordinates, and count labels.

    Example usage:
    save_data(images, pins, labels, 'output_data')
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Define subdirectories for images and count labels
    images_directory = os.path.join(images_directory, 'images')

    # Create subdirectories for images and count labels
    os.makedirs(images_directory, exist_ok=True)

    # Save images as "0.png" or "0.npy", "1.png" or "1.npy", etc., and dump data to CSV
    with open(os.path.join(output_directory, 'pins.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(['image', 'pins', 'outputs'])

        for i, (image, image_pins, label) in enumerate(zip(images, pins, labels)):
            # Save the image as "i.png" or "i.npy" in the images subdirectory
            image_filename = os.path.join(images_directory, f"{i}")
            if image.ndim == 2:
                # For single-channel images, save as PNG
                image_filename += ".png"
                image = image.detach().cpu().numpy()
                im = Image.fromarray((image * 255).astype('uint8'))
                if not os.path.exists(image_filename):
                    im.save(image_filename)
            elif image.ndim == 3:
                # For multi-channel images, save as NPY
                image_filename += ".npy"
                if not os.path.exists(image_filename):
                    np.save(image_filename, image.detach().cpu().numpy())

            # Write data to CSV
            csv_writer.writerow([os.path.basename(image_filename), image_pins, label])

    print("Data and images have been saved to the CSV and image files.")
    


class PinDataset(Dataset):
    """Synthetic Heatmaps dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pins_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pins_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pins_frame.iloc[idx, 0])
                # Check the file extension
        if img_name.endswith('.npy'):
            image = np.load(img_name)
        else:
            image = io.imread(img_name)
        pins = np.asarray(eval(self.pins_frame.iloc[idx, 1]))
        outputs = np.asarray(eval(self.pins_frame.iloc[idx, 2]))

        sample = {'image': image, 'pins': pins, 'outputs': outputs}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pins, outputs = sample['image'], sample['pins'], sample['outputs']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = image.transpose((2, 0, 1))
        image = image/image.max()
        return {'image': torch.from_numpy(image),
                'pins': torch.from_numpy(pins),
                'outputs': torch.from_numpy(outputs).to(torch.float32)}
    
    
# Define a custom transform to resize the image
class Resize(object):
    def __call__(self, sample, size=(28,28)):
        image, pins, outputs = sample['image'], sample['pins'], sample['outputs']
        
        # Resize the image to desired sized pixels
        image = transforms.functional.resize(image, size)
        
        return {'image': image, 'pins': pins, 'outputs': outputs}

    
def custom_collate_fn(batch):
    images = [sample['image'] for sample in batch]
    pins = [sample['pins'] for sample in batch]
    outputs = [sample['outputs'] for sample in batch]


    return {
        'image': torch.stack(images, dim=0),
        'pins': pins,
        'outputs': outputs}


def save_loss(data, filename):
    np.save(filename, data)

# Function to load data
def load_loss(filename):
    return np.load(filename)