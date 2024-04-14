import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import constants

class DataPreprocessor:
    def __init__(self, dataset_path, transform_type, random_seed):
        #print("Initialized Data Manager")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform_type = transform_type
        self.dataset_path = dataset_path
        self.random_seed = random_seed

    def setup_original_transform(self):
        
        self.original_transform = transforms.Compose([
            transforms.Resize(constants.DEFAULT_ORIGNAL_TRANSFORM_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = constants.DEFAULT_ORIGNAL_NORMALIZE_MEAN, std = constants.DEFAULT_ORIGNAL_NORMALIZE_STD),
        ])
        
        return self.original_transform

    def setup_augmented_transform(self):
        self.augmented_transform = transforms.Compose([
            transforms.Resize(constants.DEFAULT_AUGMENTED_TRANSFORM_RESIZE),
            transforms.RandomHorizontalFlip(p = constants.DEFAULT_AUGMENTED_RANDOM_HORIZONTAL_FLIP),
            transforms.RandomRotation(degrees = constants.DEFAULT_AUGMENTED_RANDOM_ROTATION),
            transforms.ToTensor(),
            transforms.Normalize(mean = constants.DEFAULT_ORIGNAL_NORMALIZE_MEAN, std = constants.DEFAULT_ORIGNAL_NORMALIZE_STD),
        ])

        return self.augmented_transform
    
    def setup_augmented_transform_googlenet(self):
        self.augmented_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(p = constants.DEFAULT_AUGMENTED_RANDOM_HORIZONTAL_FLIP),
            transforms.RandomRotation(degrees = constants.DEFAULT_AUGMENTED_RANDOM_ROTATION),
            transforms.ToTensor(),
            transforms.Normalize(mean = constants.DEFAULT_ORIGNAL_NORMALIZE_MEAN, std = constants.DEFAULT_ORIGNAL_NORMALIZE_STD),
        ])

        return self.augmented_transform

    def process_dataset(self):
        
        # Set up transformation
        if self.transform_type == 'original':
            transform = self.setup_original_transform()
        elif self.transform_type == 'augmented':
            transform = self.setup_augmented_transform()
        else:
            transform = None
        
        self.dataset = datasets.ImageFolder(root=self.dataset_path, transform=transform)
        train_dataset, validation_dataset, test_dataset = self.split_dataset()

        return train_dataset, validation_dataset, test_dataset
    
    def process_dataset_googlenet(self):
        
        # Set up transformation
        if self.transform_type == 'original':
            transform = self.setup_original_transform()
        elif self.transform_type == 'augmented':
            transform = self.setup_augmented_transform_googlenet()
        else:
            transform = None
        
        self.dataset = datasets.ImageFolder(root=self.dataset_path, transform=transform)
        train_dataset, validation_dataset, test_dataset = self.split_dataset()

        return train_dataset, validation_dataset, test_dataset
        

    def split_dataset(self):
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.15 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        targets = [s[1] for s in self.dataset.samples]
        train_val_idx, test_idx = train_test_split(
            range(len(targets)),
            test_size=test_size / len(self.dataset),
            stratify=targets,
            random_state=self.random_seed
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (train_size + val_size),
            stratify=[targets[i] for i in train_val_idx],
            random_state=self.random_seed
        )

        train_dataset = Subset(self.dataset, train_idx)
        validation_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)

        return train_dataset, validation_dataset, test_dataset