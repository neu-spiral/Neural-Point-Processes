# Neural Point Processes for Pixel-wise Regression

This repository contains the implementation of the paper "Neural Point Processes for Pixel-wise Regression," which introduces a novel approach combining 2D Gaussian Processes with neural networks to address pixel-wise regression problems in sparsely annotated images. Our method, termed Neural Point Processes (NPP), leverages spatial correlations between sparse labels on images to improve regression accuracy significantly.

## Project Description

Pixel-wise regression tasks in sparsely annotated images present unique challenges, as traditional regression methods often struggle to learn from unlabeled areas, resulting in distorted predictions. To overcome this limitation, we propose the usage of Neural Point Processes (NPPs), a novel approach that integrates 2D Gaussian Processes with deep neural networks. By treating labels at each image point as a Gaussian process and regressing their means through a neural network, we exploit spatial correlations effectively. This method demonstrates superior performance in terms of mean-squared error and $R^2$ scores across various datasets. NPPs offer a promising new direction for sparse pixel-wise image regression tasks by effectively capturing spatial correlations through the combined use of Gaussian processes and neural networks. Our experimental results validate the method's efficacy, showing improved accuracy over standard techniques. Future work may explore regressing covariance of the Gaussian process to further enhance the model's predictive power.

## Installation

This project can be installed and run in various environments, depending on your preferences and system setup. Below are instructions for setting up the project using Docker and Conda.

### Option 1: Using Docker

Docker is a platform for developing, shipping, and running applications in lightweight containers. This method ensures a consistent environment that's reproducible across different systems. If you do not have Docker already, follow the next optional step:

**Install Docker:** If you haven't already, [install Docker](https://docs.docker.com/get-docker/) on your system.

After installing Docker. Follow these steps to reproduce our results and experiments:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/neu-spiral/Satellite_Fusion.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Satellite_Fusion
   ```
   After cloning the repository, please refer to our [Data Description](#data-description) section to download the data and place it in the [data](data) folder.

3. **Build the Docker image**:
   Build a Docker image named `npp` (you can choose any name you like) with the Dockerfile provided in the repository.
   ```bash
   docker build -t npp .
   ```

4. **Run the Docker container**:
   Start a container from the image. This command maps port 8888 from the container to the host, allowing you to access Jupyter Notebooks.
   ```bash
   docker run -it --rm -p 8888:8888 npp
   ```

   Inside the container, run any training script as described below or you can also start JupyterLab by running:
   ```bash
   jupyter lab --ip 0.0.0.0 --no-browser --allow-root
   ```
   Copy the URL shown in the terminal to your browser to access the Jupyter Notebook interface.

### Option 2: Using Conda

Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs, and updates packages and their dependencies. If you are new to conda we recommend the following optional step:

**Install Conda**: Users can choose between Anaconda and Miniconda. Anaconda includes Conda, Python, and a large collection of pre-installed packages geared towards scientific computing. Miniconda includes only Conda and Python, allowing users to install only the packages they need.

   - To install **Anaconda**, users can download the installer from the [Anaconda website](https://www.anaconda.com/products/individual) and follow the installation instructions.
   - To install **Miniconda**, users can download the installer from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions.

Users with a conda installation can proceed with the next steps:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/neu-spiral/Satellite_Fusion.git
   ```

2. **Create a Conda environment**:
   Navigate to the project directory and create a Conda environment.
   ```bash
   cd Satellite_Fusion
   conda create --name npp python=3.8.13
   ```

3. **Activate the Conda environment**:
   ```bash
   conda activate npp
   ```

4. **Install the required dependencies**:
   Install all the dependencies listed in the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

5. **Running the project**:
   After installation, you can run your scripts or start JupyterLab:
   ```bash
   jupyter lab
   ```
After cloning the repository and initializing the environment, please refer to our [Data Description](#data-description) section to download the data and place it in the [data](data) folder.

## Usage

To train our models (NNP and NPP-GP) and the Plain model you need to use the `npp.py` script. After finishing the training of the models, the models are evaluated automatically for the following values of percentage of labels to be shown partially during testing: `0.25, 0.5, 0.75, 1.00`. The code can be executed with the following command:

```bash
python npp.py --dataset <dataset> --feature <feature> --mode <mode> --n <n> --d <d> --n_pins <n_pins> --partial_percent <partial_percent> --r <r> --epochs <epochs> --batch_size <batch_size> --learning_rate <learning_rate> --val_every_epoch <val_every_epoch> --num_runs <num_runs> --sigmas <sigmas> --num_encoder <num_encoder> --num_decoder <num_decoder> --deeper --experiment_name <experiment_name>
```

You can customize the execution by modifying the command-line arguments as per your requirements. Below is a description of each available option:

- `--dataset`: Specifies the name of the dataset to be used. Options are "Synthetic", "PinMNIST" and "Bulding". Default is "PinMNIST".

- `--modality`: Building dataset modality. Default is "PS-RGBNIR". Notice this option won't be used if dataset is "Synthetic" or "PinMNIST". 

- `--feature`: Indicates the type of feature extraction to use, options are "DDPM" or "AE". Default is "AE".

- `--mode`: Determines the mode for selecting points within images, can be either "mesh" for a mesh grid or "random" for random selection. Default is "mesh".

- `--n`: Number of samples in the dataset. Default is 100.

- `--d`: Used to define the spacing between points in a mesh grid. You do not need to define it if using "random". Default is 10.

- `--n_pins`: Specifies the number of points (pins) selected from each image in the random setting. You do not need to define it if using "mesh". Default is 500.

- `--partial_percent`: Sets the percentage of labels to be shown partially during the process of training evaluation, ranges from 0 to 1. Default is 0.00. We recommend leaving the default value since other partial percentages are automatically tested afterwards.

- `--r`: Defines a radius or another parameter relevant to the dataset or feature extraction method. Default is 3.

- `--epochs`: The number of epochs for training the model. Default is 200.

- `--batch_size`: Size of the batch used during training. Default is 64.

- `--learning_rate`: Initial learning rate for the optimization algorithm. Default is 0.1.

- `--val_every_epoch`: Frequency of validation, specified as the number of epochs between validations. Default is 5.

- `--num_runs`: The number of times to train the model with different initializations or datasets. Default is 3.

- `--manual_lr`: A flag that, when used, disables the custom learning rate finder, falling back to the specified `--learning_rate`.

- `--sigmas`: A list of sigma values to test, used for Gaussian kernel or other components that utilize a bandwidth parameter. Default is `[0.1, 0.2, 0.5, 1, 2, 5, 10, 20]`.

- `--num_encoder`: List of kernel sizes for the encoder part of a model. Default is `[64, 32]`.

- `--num_decoder`: List of kernel sizes for the decoder part of a model. Default is `[64]`.

- `--deeper`: A flag to add extra convolutional layers to the model, enhancing its complexity.

- `--experiment_name`: Specifies a custom folder name under which to save the generated experiments, allowing for organized storage and retrieval of experimental results.

These options allow for extensive customization and tuning of the model's training and evaluation process, facilitating experiments across various datasets and configurations.

In order to obtain the NP results you need ... TO DO:
---

## Examples

Before executing the example commands, ensure you have followed the installation instructions and have the necessary datasets.

### Training and Evaluating the Model

```bash
python npp.py --dataset PinMNIST --epochs 200 --batch_size 64 --learning_rate 0.01 --val_every_epoch 5 --num_runs 3 --sigmas 0.1 0.5 1.0 --num_encoder 32 16 --num_decoder 32 --experiment_name my_experiment
```

Once the script is run, you can visualize the results using the notebook: `results_summary.ipynb`.

## Data Description

This project utilizes three datasets: Synthetic Elevation Heatmaps, PMNIST, and Rotterdam. Below are brief descriptions and access links to processed datasets:

- **Synthetic Elevation Heatmaps:** Generated representations of random elevations as heatmaps. The generated datased has 1000 samples. Each heatmap is a 28x28 matrix representing "elevation" values derived from a combination of sine and cosine functions. [Access Dataset](#)

- **PMNIST:** A subset of 1000 images from the MNIST dataset, where labels are produced by summing pixel values in a specified radius around selected points. Again the images are 28x28. [Access Dataset](#)

- **Rotterdam:** 1000 selected satellite images from the SpaceNet6 dataset over Rotterdam, Netherlands. Labels are generated by counting unique buildings within a specified radius around selected points. Images are 100x100. [Access Dataset](#)

Please ensure you have access rights to the datasets or contact the authors for more information. After downloading them, unzip the files inside the [data](data) folder so that the scripts can access it automatically. We also provide the scripts and notebooks used to generate the labels and final datasets too. Tou can find them in [processing](processing), with two folders to generate the labels for the rotterdam and synthetic datasets.

<!-- >## Citation

If you find this work useful, please cite our paper:
```
@article{NeuralPointProcesses2023,
  title={Neural Point Processes for Pixel-wise Regression},
  author={Authors},
  journal={Journal Name},
  year={2023}
}
```
-->
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Acknowledgments

- We thank the reviewers for their valuable feedback.
- This work was supported by [Funding Agency/Institution].
