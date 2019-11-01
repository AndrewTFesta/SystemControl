"""
@title
@description
"""

from __future__ import print_function, division

import os
# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from SystemControl import DATA_DIR
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource, int_to_subject_str
from SystemControl.DataTransformer import Rescale, CustomToTensor, Interpolation

warnings.filterwarnings("ignore")


class ImageDataset(Dataset):

    def __init__(self, base_dir: str, transform=None):
        self._base_dir = base_dir
        self._transform = transform

        self._image_paths = []
        self.__load_subject_images()
        return

    def __iter__(self):
        for image_entry in self._image_paths:
            sample = self.__prepare_entry(image_entry)
            yield sample
        return

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_entry = self._image_paths[item]
        sample = self.__prepare_entry(image_entry)
        return sample

    def __load_subject_images(self):
        if not os.path.isdir(self._base_dir):
            print(f'Image directory not found: {self._base_dir}')
            return
        event_dirs = [
            os.path.join(self._base_dir, each_dir)
            for each_dir in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, each_dir))
        ]
        image_paths = {
            os.path.basename(each_dir).split('_')[-1]: [
                os.path.join(each_dir, each_file)
                for each_file in os.listdir(each_dir)
                if os.path.isfile(os.path.join(each_dir, each_file))
            ]
            for each_dir in event_dirs
        }
        for event_type, path_list in image_paths.items():
            for each_path in path_list:
                image_entry = {
                    'event': event_type,
                    'path': each_path
                }
                self._image_paths.append(image_entry)
        return

    def __prepare_entry(self, image_entry):
        img_path = image_entry['path']
        img_type = image_entry['event']

        image = io.imread(img_path)
        sample = {'image': image, 'event': img_type}
        if self._transform:
            sample = self._transform(sample)
        return sample


def count_custom_events(dataset):
    event_counts = {}
    for each_point in dataset:
        event = each_point['event']
        if event not in event_counts:
            event_counts[event] = 0
        event_counts[event] += 1
    return event_counts


def analyze_custom_dataset(dataset, batch_size: int, num_workers: int, shuffle: bool):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f'Base dataset')
    # print(f'Displaying batches: {batch_size}')
    # for idx_batch, sample_batched in enumerate(data_loader):
    #     print(idx_batch, sample_batched['image'].size(), sample_batched['event'])

    event_counts = count_custom_events(dataset)
    total_count = sum(event_counts.values())
    print(f'Total: {total_count}')
    for each_event, each_count in event_counts.items():
        print(f'{each_event}: {each_count}: {each_count / total_count: 0.4f}')

    print('Random split')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print('Train dataset')
    event_counts = count_custom_events(train_dataset)
    total_count = sum(event_counts.values())
    print(f'Total: {total_count}')
    for each_event, each_count in event_counts.items():
        print(f'{each_event}: {each_count}: {each_count / total_count: 0.4f}')

    print('Test dataset')
    event_counts = count_custom_events(test_dataset)
    total_count = sum(event_counts.values())
    print(f'Total: {total_count}')
    for each_event, each_count in event_counts.items():
        print(f'{each_event}: {each_count}: {each_count / total_count: 0.4f}')
    return


def target_to_class(class_to_idx, target):
    class_str = ''
    for each_class, each_target in class_to_idx.items():
        if each_target is target:
            class_str = each_class
            break
    return class_str


def analyze_builtin_dataset(dataset, batch_size: int, num_workers: int, shuffle: bool):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f'Base dataset')
    # print(f'Displaying batches: {batch_size}')
    # for idx_batch, sample_batched in enumerate(data_loader):
    #     print(idx_batch, sample_batched['image'].size(), sample_batched['event'])

    targets = set(dataset.targets)
    event_counts = {}
    for each_target in targets:
        event_counts[target_to_class(dataset.class_to_idx, each_target)] = dataset.targets.count(each_target)
    total_count = sum(event_counts.values())
    print(f'Total: {total_count}')
    for each_event, each_count in event_counts.items():
        print(f'{each_event}: {each_count}: {each_count / total_count: 0.4f}')

    print('Random split')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(len(train_dataset), len(test_dataset))
    return


def main():
    ########################################################################
    print('Using custom dataset')
    subject = 1
    subject_str = int_to_subject_str(subject)
    interpolation = Interpolation.LINEAR.name
    base_dir = os.path.join(DATA_DIR, 'heatmaps', PhysioDataSource.NAME, interpolation, subject_str)

    transform_list = [Rescale(256), CustomToTensor()]
    composed_transforms = transforms.Compose(transform_list)
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 4

    train_size = 0.7
    test_size = 0.15
    val_size = 0.15

    # img_dataset = ImageDataset(base_dir, transform=composed_transforms)
    # analyze_custom_dataset(img_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    print('Using ImageFolder')
    img_folder = ImageFolder(base_dir, transform=ToTensor())
    analyze_builtin_dataset(img_folder, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return
    ###################################################################################
    batch_size = 16
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print(train_indices, val_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, sampler=valid_sampler)

    # Usage Example:
    num_epochs = 10
    for epoch in range(num_epochs):
        # Train:
        for batch_index, (images, labels) in enumerate(train_loader):
            print(batch_index, len(images), len(labels))

    for batch_index, (images, labels) in enumerate(validation_loader):
        print(batch_index, len(images), len(labels))
    return


if __name__ == '__main__':
    main()
