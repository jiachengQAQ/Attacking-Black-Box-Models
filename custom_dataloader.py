import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torchvision
import os
import pdb

class CustomDataset(Dataset):
    def __init__(self, root_dir, args):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.75, 1)),
            transforms.ToTensor(),
        ])
        if args.data_type == "cifar100":
            self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x))
        else:
            self.classes = sorted(os.listdir(root_dir)) ## 0-9 string
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)} ##:
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class Imagenet_Dataset(Dataset):
    def __init__(self, root_dir, mode):
        self.mode = mode
        self.root_dir = root_dir
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.classes = sorted(os.listdir(root_dir)) ## 0-9 string
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)} ##
        self.images = self._load_images()
    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.mode == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, label
