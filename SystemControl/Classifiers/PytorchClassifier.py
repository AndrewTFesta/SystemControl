"""
@title
@description
"""
import argparse
import os
import time
import types
from collections import namedtuple
from concurrent.futures.thread import ThreadPoolExecutor
from copy import copy
from math import ceil

import numpy as np
import skimage
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
from skimage import transform
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.PhysioDataSource import int_to_subject_str, PhysioDataSource
from SystemControl.DataTransformer import Interpolation

# todo add test_time
# todo add train_time
# todo add confusion matrix
metric_fields = ('epoch', 'train_loss', 'accuracy', 'test_loss', 'tp', 'fp', 'tn', 'fn')

MetricEntry = namedtuple(
    'MetricEntry',
    metric_fields
)


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, event = sample['image'], sample['event']

        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)
        img = skimage.transform.resize(image, (new_height, new_width))
        return {'image': img, 'event': event}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size.
        If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        return

    def __call__(self, sample):
        image, event = sample['image'], sample['event']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        return {'image': image, 'event': event}


class CustomToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image, event = sample['image'], sample['event']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'event': event}


def update_future_pbar(future):
    pbar = future.arg
    pbar.update(1)
    return


def load_model(model_func):
    try:
        model = model_func(pretrained=True, progress=True)
    except (NotImplementedError, ValueError):
        model = model_func(pretrained=False, progress=True)
    return model


def load_all_models():
    model_choices = {
        name: model_func
        for name, model_func in models.__dict__.items()
        if name.islower() and not name.startswith("__") and isinstance(model_func, types.FunctionType)
    }
    # Instancing a pre-trained model will download its weights to a cache directory.
    # This directory can be set using the TORCH_HOME environment variable.
    model_dir = os.path.join(DATA_DIR, 'models', 'pretrained')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    os.environ['TORCH_HOME'] = model_dir

    pbar = tqdm(total=len(model_choices.items()), desc=f'Loading all models')
    with ThreadPoolExecutor(max_workers=10) as executor:
        for model_idx, (model_name, model_func) in enumerate(model_choices.items()):
            load_future = executor.submit(load_model, model_func)
            load_future.arg = pbar
            load_future.add_done_callback(update_future_pbar)
    pbar.close()
    return


def train(model, device, epoch, data_loader, criterion, optimizer, display=True):
    print(f'Train epoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if display:
            print(f'\t\tpred: {predicted}')
            print(f'\t\ttarg: {targets}')
    return train_loss


def test(model, device, epoch, data_loader, criterion, display: bool = True):
    print(f'Test epoch: {epoch}')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if display:
                print(f'\t\tpred: {predicted}')
                print(f'\t\ttarg: {targets}')
        acc = 100. * correct / total
        print(f'**Accuracy: {acc:0.2f}%**')
        print(f'**Saving checkpoint**')
        fp = None
        tp = None
        fn = None
        tn = None
    return test_loss, acc, fp, tp, fn, tn


def save_checkpoint(model, metric_list, checkpoint_fname):
    state = {
        'model': model.state_dict(),
        'metric_list': metric_list
    }
    checkpoint_dir = os.path.dirname(checkpoint_fname)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_fname)
    return


def get_newest_checkpoint(checkpoint_path):
    latest_checkpoint_fname = ''
    if not os.path.isdir(checkpoint_path):
        print(f'Unable to locate checkpoints for model: {checkpoint_path}')
    else:
        checkpoint_files = sorted([
            each_file
            for each_file in os.listdir(checkpoint_path)
            if each_file.endswith('.pth')
        ])

        if len(checkpoint_files) > 0:
            latest_checkpoint_fname = checkpoint_files[-1]
        else:
            print(f'No checkpoint files located in model directory: {checkpoint_path}')
    return os.path.join(checkpoint_path, latest_checkpoint_fname)


def get_best_epoch(metric_list) -> int:
    last_metric = metric_list[-1]
    last_epoch = last_metric['epoch']
    return last_epoch


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parsed_args = parser.parse_args()
    return parsed_args


def main(main_args):
    ####################################################################################
    print(f'Setting parameters')
    load_all = False
    model_name = 'resnet18'
    model_dir = os.path.join(DATA_DIR, 'models', 'pretrained')
    os.environ['TORCH_HOME'] = model_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'
    cudnn.benchmark = device == 'cuda'

    num_workers = 4
    num_epochs = 12
    resume_training = False
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_train_batches = 8
    num_test_batches = 2
    shuffle: bool = True
    img_size = 224
    lr_step_size = 30
    lr_gamma = 0.1
    display_metrics = True

    start_epoch = 0
    metric_list = []

    subject = 1
    interpolation = Interpolation.LINEAR.name

    subject_str = int_to_subject_str(subject)
    base_dir = os.path.join(DATA_DIR, 'heatmaps', PhysioDataSource.NAME, interpolation, subject_str)
    out_dir = os.path.join(DATA_DIR, 'output', PhysioDataSource.NAME, interpolation, subject_str, f'{model_name}')

    checkpoint_path = os.path.join(out_dir, 'checkpoints')
    time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
    checkpoint_fname = os.path.join(checkpoint_path, f'checkpoint_{time_stamp}.pth')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    img_dataset = ImageFolder(base_dir)
    ####################################################################################
    print(f'Building missing directories')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ####################################################################################
    print(f'Building model: {model_name}')
    s_time = time.time()
    if load_all:
        load_all_models()
    model_choices = {
        name: model_func
        for name, model_func in models.__dict__.items()
        if name.islower() and not name.startswith("__") and isinstance(model_func, types.FunctionType)
    }

    model = load_model(model_choices[model_name])
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    latest_checkpoint_fname = get_newest_checkpoint(checkpoint_path)
    if latest_checkpoint_fname:
        if display_metrics:
            print(f'==> Displaying metrics for model: {latest_checkpoint_fname}')
            checkpoint = torch.load(latest_checkpoint_fname)
            model.load_state_dict(checkpoint['model'])
            metric_list = checkpoint['metric_list']
            for each_entry in metric_list:
                print(f'{each_entry}')
            return
        elif resume_training:
            # Load checkpoint
            print(f'==> Resuming from checkpoint: {latest_checkpoint_fname}')
            checkpoint = torch.load(latest_checkpoint_fname)
            model.load_state_dict(checkpoint['model'])
            metric_list = checkpoint['metric_list']
            start_epoch = get_best_epoch(metric_list)

    e_time = time.time()
    print(f'Time to load model: {e_time - s_time:0.4f} seconds')
    ####################################################################################
    print(f'Loading data:')
    s_time = time.time()
    train_size = int(0.8 * len(img_dataset))
    test_size = len(img_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(img_dataset, [train_size, test_size])
    train_dataset.dataset = copy(img_dataset)

    train_dataset.dataset.transform = transform_train
    test_dataset.dataset.transform = transform_test

    train_batch_size = ceil(len(train_dataset) / num_train_batches)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_batch_size = ceil(len(test_dataset) / num_test_batches)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    e_time = time.time()
    print(f'Time to load data: {e_time - s_time:0.4f} seconds')
    ####################################################################################
    print(f'Training model: {model_name}')
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss = train(
            model=model, device=device, epoch=epoch, data_loader=data_loader_train,
            criterion=loss_func, optimizer=optimizer, display=False
        )
        test_loss, acc, fp, tp, fn, tn = test(
            model=model, device=device, epoch=epoch, data_loader=data_loader_test,
            criterion=loss_func, display=False
        )
        metric_entry = MetricEntry(
            epoch=epoch,
            train_loss=train_loss, test_loss=test_loss,
            accuracy=acc, fp=fp, tp=tp, fn=fn, tn=tn
        )
        metric_list.append(metric_entry)
        save_checkpoint(model=model, metric_list=metric_list, checkpoint_fname=checkpoint_fname)
    return


if __name__ == '__main__':
    cline_args = parse_args()
    main(cline_args)
