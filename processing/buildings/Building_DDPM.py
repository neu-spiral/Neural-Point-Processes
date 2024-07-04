import os
import sys
sys.path.append('../..')
from torchvision.transforms import transforms, Compose, ToTensor, Lambda
from torch.utils.data import DataLoader, random_split
from tools.data_utils import *
import numpy as np
import tools.sr3 as sr3
from tqdm import tqdm  # Import tqdm for progress bar


def concat_feature_maps(feature_maps, layers=[5, 6, 7, 8]):
    # Define empty list to store upsampled feature maps for specific layers
    upsampled_maps = []

    # Upsample and store feature maps for specified layers
    for t in range(len(feature_maps)):
        for layer_idx in layers:
            fmap = feature_maps[t][layer_idx]
            upsampled_fmap = torch.nn.functional.interpolate(fmap, size=(128, 128), mode='bilinear',
                                                             align_corners=False)
            upsampled_maps.append(upsampled_fmap.half().detach().cpu())

    # Concatenate the upsampled feature maps along the channel dimension
    concatenated_maps = torch.cat(upsampled_maps, dim=1)

    return concatenated_maps




def save_fm_by_batch(opt, data_loader, images_directory, output_directory, output_featuremap = True):

    opt = sr3.dict_to_nonedict(opt)
    # Loading diffusion model
    diffusion = sr3.DDPM(opt)
    diffusion.netG.eval()
    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    images_directory = os.path.join(images_directory, 'images')

    # Create subdirectories for images and count labels
    os.makedirs(images_directory, exist_ok=True)

    # Save images as "0.png" or "0.npy", "1.png" or "1.npy", etc., and dump data to CSV
    with open(os.path.join(output_directory, 'pins.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(['image', 'pins', 'outputs'])
        # Use tqdm to wrap the data_loader for progress visualization
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
                torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                images = batch['image'].to(device)  # get RGB instead of RGBA
                pins = batch['pins']
                outputs = batch['outputs']
                if output_featuremap:
                    diffusion.feed_data(images)

                    # Disable gradient calculation
                    diffusion.feed_data(images)
                    f_A = []
                    for t in opt['model_cd']['t']:
                        fe_A_t, fd_A_t = diffusion.get_feats(t=t)  # np.random.randint(low=2, high=8)
                        f_A.append(fd_A_t)
                    concat_fm = concat_feature_maps(f_A)

                for i in range(len(images)):
                    # Save the image as "overall_index.png" or "overall_index.npy" in the images subdirectory
                    image_filename = os.path.join(images_directory, f"{count}")
                    # For multi-channel images, save as NPY
                    image_filename += ".npy"
                    if output_featuremap and not os.path.exists(image_filename):
                        np.save(image_filename, concat_fm[i].detach().cpu().numpy())
                    # Write data to CSV
                    tuple_pin = [tuple(point.detach().cpu().numpy()) for point in pins[i]]
                    list_outputs = [int(output.item()) for output in outputs[i]]
                    csv_writer.writerow([os.path.basename(image_filename), tuple_pin, list_outputs])
                    count += 1
                if output_featuremap:
                    del f_A
    print("Data and images have been saved to the CSV and image files.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


opt = {
    "name": "ddpm-RS-CDHead-LEVIR",
    "phase": "test",
    "gpu_ids": [
        0
    ],
     "path": {
        "resume_state": "../../history/pretrained_sr3/sr3_50_100"
    },

    "datasets": {
        "train": {
            "name": "LEVIR-CD-256",
            "dataroot": "dataset/LEVIR-CD256/",
            "resolution": 256,
            "batch_size": 8,
            "num_workers": 8,
            "use_shuffle": True,
            "data_len": -1
        },
        "val": {
            "name": "LEVIR-CD-256",
            "dataroot": "dataset/LEVIR-CD256/",
            "resolution": 256,
            "batch_size": 4,
            "num_workers": 8,
            "use_shuffle": True,
            "data_len": -1
        },
        "test": {
            "name": "LEVIR-CD-256",
            "dataroot": "dataset/LEVIR-CD256/",
            "resolution": 256,
            "batch_size": 4,
            "num_workers": 8,
            "use_shuffle": False,
            "data_len": -1
        }
    },
    "model_cd": {
        "feat_scales": [2, 5, 8, 11, 14],
        "out_channels": 2,
        "loss_type": "ce",
        "output_cm_size": 256,
        "feat_type": "dec",
        "t": [50, 100]
    },

    "model": {
        "which_model_G": "sr3",
        "finetune_norm": False,
        "unet": {
            "in_channel": 3,
            "out_channel": 3,
            "inner_channel": 128,
            "channel_multiplier": [
                1,
                2,
                4,
                8,
                8
            ],
            "attn_res": [
                16
            ],
            "res_blocks": 2,
            "dropout": 0.2
        },
        "beta_schedule": {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            },
            "val": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            },
            "test": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            }
        },
        "diffusion": {
            "image_size": 256,
            "channels": 3,
            "loss": "l2",
            "conditional": False
        }
    }
}

# Hyperparameters
dataset = "Building"
batch_size = 16
input_channel = 3
# put in your n
n = 32
resize = Resize256
output_directory = f"../../data/Building_ddpm"
# choose random/mesh
data_folder = f"mesh_{n}_step"
transformed_dataset = PinDataset(csv_file=f"../../data/{dataset}/{data_folder}/pins.csv",
                                 root_dir=f"../../data/{dataset}/images/",
                                 transform=Compose([ToTensor(), resize(), Lambda()]))

data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


save_fm_by_batch(opt, data_loader, images_directory="../../data/Building_ddpm/", output_directory=f"../../data/Building_ddpm/{data_folder}")