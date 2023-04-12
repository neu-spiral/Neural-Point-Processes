import os
import torch
import numpy as np
import rasterio as rio
import skimage.io as io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class SpaceNet6(Dataset):
    def __init__(self, files, crop_size, exec_mode):

        self.files = files
        self.crop_size = crop_size
        self.exec_mode = exec_mode

    def OpenImage(self, idx, invert=False):
        image = rio.open(self.files[idx][0]).read()
        if invert:
            image = image.transpose((2, 0, 1))
        return image / np.iinfo(image.dtype).max

    def OpenMask(self, idx):
        mask = io.imread(self.files[idx][1])
        # temp = np.row_stack([mask, np.zeros(mask.shape, dtype=np.uint8)])
        # temp = np.row_stack([temp, np.zeros(mask.shape, dtype=np.uint8)])
        return mask

    def __getitem__(self, idx):
        # read the images and masks as numpy arrays

        x = self.OpenImage(idx)
        y = self.OpenMask(idx)

        x, y = self.padding((x, y[None]))

        # if it is the training phase, create random (C, 430, 430) crops
        # if it is the evaluation phase, we will leave the orginal size (C, 1024, 1024)
        if self.exec_mode in ['train', 'predict']:
            x, y = self.crop(x[None], y[None], self.crop_size)
            x, y = x[0], y[0]

        # numpy array --> torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.uint8)

        # normalize the images (image- image.mean()/image.std())
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return normalize(x), y

    def __len__(self):
        return len(self.files)

    def padding(self, sample):
        image, mask = sample
        C, H, W = image.shape
        if H == 900 and W == 900:
            return image, mask

        if H != 900:
            image = np.pad(image, (((0, 0), (1, 0), (0, 0))), 'constant', constant_values=(0))
            mask = np.pad(mask, (((0, 0), (1, 0), (0, 0))), 'constant', constant_values=(0))

        if W != 900:
            image = np.pad(image, (((0, 0), (0, 0), (1, 0))), 'constant', constant_values=(0))
            mask = np.pad(mask, (((0, 0), (0, 0), (1, 0))), 'constant', constant_values=(0))

        return image, mask

    def crop(self, data, seg, crop_size=256):
        data_shape = tuple([len(data)] + list(data[0].shape))
        data_dtype = data[0].dtype
        dim = len(data_shape) - 2

        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype
        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

        crop_size = [crop_size] * dim
        data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)


        for b in range(data_shape[0]):
            data_shape_here = [data_shape[0]] + list(data[b].shape)
            seg_shape_here = [[seg_shape[0]]] + list(seg[0].shape)

            lbs = []
            for i in range(len(data_shape_here) - 2):
                lbs.append(np.random.randint(0, data_shape_here[i+2] - crop_size[i]))

            ubs = [lbs[d] + crop_size[d] for d in range(dim)]

            slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            data_cropped = data[b][tuple(slicer_data)]

            slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

            data_return[b] = data_cropped
            seg_return[b] = seg_cropped

        return data_return, seg_return


class SpaceNet6DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.spaceNet6_val = None
        self.spaceNet6_train = None
        self.args = args
        self.data = args.base_dir

    def setup(self, stage=None):

        allpths = [os.path.join(self.data, p, os.listdir(os.path.join(self.data, p))[0])
                   for p in os.listdir(self.data)]

        imgs = sorted([p for p in allpths if "PS-RGB" in os.path.basename(p)],
                      key=lambda x: os.path.dirname(x).split("tile_")[1])

        msks = sorted([p for p in allpths if "mask" in os.path.basename(p)],
                      key=lambda x: os.path.dirname(x).split("tile_")[1])

        data = list(zip(imgs, msks))

        train_files, test_files = train_test_split(data, test_size=0.1, random_state=self.args.seed)
        train_files, val_files = train_test_split(data, test_size=0.4, random_state=self.args.seed)

        self.spaceNet6_train = SpaceNet6(train_files, self.args.crop_size, self.args.exec_mode)
        self.spaceNet6_test = SpaceNet6(test_files, self.args.crop_size, self.args.exec_mode)
        self.spaceNet6_val = SpaceNet6(val_files, self.args.crop_size, self.args.exec_mode)

    def train_dataloader(self):
        train_sampler = self.ImageSampler(len(self.spaceNet6_train), self.args.samples_per_epoch)
        train_bSampler = BatchSampler(train_sampler, batch_size=self.args.batch_size, drop_last=True)
        return DataLoader(self.spaceNet6_train, batch_sampler=train_bSampler, num_workers=self.args.num_workers)

    def test_dataloader(self):
        return DataLoader(self.spaceNet6_test,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.spaceNet6_val,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.spaceNet6_val,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          drop_last=False)

    class ImageSampler(Sampler):
        def __init__(self, num_images=300, num_samples=500):
            self.num_images = num_images
            self.num_samples = num_samples

        def generate_iteration_list(self):
            return np.random.randint(0, self.num_images, self.num_samples)

        def __iter__(self):
            return iter(self.generate_iteration_list())

        def __len__(self):
            return self.num_samples
