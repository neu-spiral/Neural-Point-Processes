# Neural Point Processes for Pixel-wise Regression

This repository contains the implementation of the paper "Neural Point Processes for Pixel-wise Regression," which introduces a novel approach combining 2D Gaussian Processes with neural networks to address pixel-wise regression problems in sparsely annotated images. Our method, termed Neural Point Processes (NPP), leverages spatial correlations between sparse labels on images to improve regression accuracy significantly.

## Project Description

Pixel-wise regression tasks in sparsely annotated images present unique challenges, as traditional regression methods often struggle to learn from unlabeled areas, resulting in distorted predictions. To overcome this limitation, we propose the usage of Neural Point Processes (NPPs), a novel approach that integrates 2D Gaussian Processes with deep neural networks. By treating labels at each image point as a Gaussian process and regressing their means through a neural network, we exploit spatial correlations effectively. This method demonstrates superior performance in terms of mean-squared error and $R^2$ scores across various datasets. NPPs offer a promising new direction for sparse pixel-wise image regression tasks by effectively capturing spatial correlations through the combined use of Gaussian processes and neural networks. Our experimental results validate the method's efficacy, showing improved accuracy over standard techniques. Future work may explore regressing covariance of the Gaussian process to further enhance the model's predictive power.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/neu-spiral/Satellite_Fusion.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main code can be executed with the following command:
```bash
python main.py --dataset <dataset> --feature <feature> --mode <mode> --n <n> --d <d> --n_pins <n_pins> --partial_percent <partial_percent> --r <r> --epochs <epochs> --batch_size <batch_size> --learning_rate <learning_rate> --val_every_epoch <val_every_epoch> --num_runs <num_runs> --sigmas <sigmas> --num_encoder <num_encoder> --num_decoder <num_decoder> --deeper --experiment_name <experiment_name>
```

You can customize the execution by modifying the command-line arguments as per your requirements. Refer to the provided `main.py` script for a detailed description of all available options.

## Examples

Before executing the example commands, ensure you have followed the installation instructions and have the necessary datasets.

### Training the Model

```bash
python main.py --dataset PinMNIST --epochs 200 --batch_size 64 --learning_rate 0.01 --val_every_epoch 5 --num_runs 3 --sigmas 0.1 0.5 1.0 --num_encoder 32 16 --num_decoder 32 --experiment_name my_experiment
```

### Evaluating the Model

To evaluate the model, use the generated `experiment_id` from the training phase:
```bash
python main.py --experiment_id <your_experiment_id> --mode test
```

## Data Description

This project utilizes three main datasets: Synthetic Elevation Heatmaps, PMNIST, and Rotterdam. Below are brief descriptions and access links to processed datasets:

- **Synthetic Elevation Heatmaps:** Generated representations of random elevations as heatmaps. The generated datased has 1000 samples. Each heatmap is a 28x28 matrix representing "elevation" values derived from a combination of sine and cosine functions. [Access Dataset](#)

- **PMNIST:** A subset of 1000 images from the MNIST dataset, where labels are produced by summing pixel values in a specified radius around selected points. Again the images are 28x28. [Access Dataset](#)

- **Rotterdam:** 1000 selected satellite images from the SpaceNet6 dataset over Rotterdam, Netherlands. Labels are generated by counting unique buildings within a specified radius around selected points. Images are 100x100. [Access Dataset](#)

Please ensure you have access rights to the datasets or contact the authors for more information.

%## Citation

If you find this work useful, please cite our paper:
```
@article{NeuralPointProcesses2023,
  title={Neural Point Processes for Pixel-wise Regression},
  author={Authors},
  journal={Journal Name},
  year={2023}
}
```%

%## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.%

## Acknowledgments

- We thank the reviewers for their valuable feedback.
- This work was supported by [Funding Agency/Institution].
