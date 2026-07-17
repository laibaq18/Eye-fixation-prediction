from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class EyeFixationDataset(Dataset):
    def __init__(self, data_root, image_list_file, fixation_list_file=None, image_size=(224, 224)):

        self.data_root = Path(data_root)
        self.image_size = image_size

        image_lines = self.read_lines(self.data_root / image_list_file)
        self.image_paths = self.get_paths(image_lines)

        self.has_fixations = fixation_list_file is not None

        if self.has_fixations:
            fixation_lines = self.read_lines(
                self.data_root / fixation_list_file)
            self.fixation_paths = self.get_paths(fixation_lines)

        """
        Interpolation BILINEAR -> smoothly interpolates pixel values while resizing
        ToTensor() -> converts the PIL image into a PyTorch tensor
        values from [0,255] to now [0,1] and shape from (H x W x C)  to  (C x H x W)

        The mean and std values are ImageNet statistics
        Why ImageNet? Because ResNet-50 was pretrained on ImageNet, so it expects input images normalized in the same way.

        After normalization, values are no longer between 0 and 1. 
        They may become negative or greater than 1. which is why we also keep the raw tensor
        """

        self.image_transform = transforms.Compose([
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.raw_transform = transforms.Compose([
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.fixation_transform = transforms.Compose([
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # Converts uint8 [0,255] to float [0,1]
        ])

    def read_lines(self, path):
        img_files = []
        with open(path, "r") as f:
            for line in f.readlines():
                img_files.append(line.strip())
        return img_files

    def get_paths(self, img_files):
        paths = []
        for file in img_files:
            paths.append(self.data_root / file)
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        raw_image_tensor = self.raw_transform(image)

        sample = {
            "size": image_tensor.shape,
            "image": image_tensor,          # normalized RGB image for model
            "raw_image": raw_image_tensor,  # unnormalized RGB image for plotting
            "name": image_path.name,
        }

        if self.has_fixations:
            fixation_path = self.fixation_paths[idx]
            fixation = Image.open(fixation_path).convert("L")
            fixation_tensor = self.fixation_transform(fixation)

            # grayscale fixation map for loss
            sample["fixation"] = fixation_tensor

        return sample


# fd = EyeFixationDataset(data_root="/Users/laibaqureshi/Desktop/CV2-proj/cv2_project_data",
#                         image_list_file="train_images.txt",
#                         fixation_list_file="train_fixations.txt",
#                         image_size=(224, 224))

# print(fd.__getitem__(0))
