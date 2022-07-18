#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

from cProfile import label
import torch
import random
import copy

import numpy as np

from torch.utils.data import DataLoader, random_split

from utils.utils_ import pad_tensor, weighted_sampler
from augmentation import AffineScaling, Zoom, ChannelsShuffle


class Dataset():
    """Returns samples from an mne.io.Raw object along with a target.
    Dataset which serves samples from an mne.io.Raw object along with a target.
    The target is unique for the dataset, and is obtained through the
    `description` attribute.
    Parameters
    ----------
    data : Continuous data.
    labels : labels.
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        if self.transforms is not None:
            for transform in self.transforms:
                X = transform(X)
        return X, y

    def __len__(self):
        return len(self.data)

class PadCollate():

    """ Custom collate_fn that pads according to the longest sequence in
        a batch of sequences.
    """

    def __init__(self, dim=1):

        """
        Args:
            dim (int): Dimension to pad on.
        """

        self.dim = dim

    def pad_collate(self,
                    batch):

        """
        Args:
            batch (list): List of (tensor, label).

        Return:
            xs (tensor): Padded data.
            ys (tensor): Labels.
        """
        # Find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))

        # Pad according to max_len
        data = map(lambda x: pad_tensor(x[0], n_pads=max_len, dim=self.dim),
                   batch)
        labels = map(lambda x: torch.tensor(x[1]), batch)

        # Stack all
        xs = torch.stack(list(data), dim=0)
        ys = torch.stack(list(labels), dim=0)

        return xs, ys

    def __call__(self,
                 batch):

        return self.pad_collate(batch)


class Loader():

    """ Create dataloader of data. """

    def __init__(self,
                 data,
                 labels,
                 annotated_channels,
                 data_augment,
                 single_channel,
                 batch_size,
                 num_workers,
                 transforms=None,
                 subject_LOPO=None,
                 seed=42):

        """
        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            annotated_channels (list): channels where the spike occurs.
            single_channel (bool): If True, select just one channel
                                   in training.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            split_dataset (bool): If True, split dataset into training,
                                  validation, test.
            seed (int): Seed for reproductibility.
        """

        self.data = data
        self.labels = labels
        self.annotated_channels = annotated_channels
        self.data_augment = data_augment
        self.single_channel = single_channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.subject_LOPO = subject_LOPO
        self.seed = seed

    def pad_loader(self,
                   data,
                   labels,
                   chans,
                   single_channel,
                   shuffle,
                   batch_size,
                   transforms,
                   num_workers):

        """ Create dataloader of data.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.

        Returns:
            dataloader (array): Array of dataloader.
        """

        # Get dataloader

        dataset = Dataset(data, labels, transforms=transforms)
        loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=PadCollate(dim=1))

        return loader

    def LOPO_dataloader(self,
                        data_base,
                        labels_base,
                        annotated_channels,
                        single_channel,
                        batch_size,
                        num_workers,
                        subject_LOPO,
                        seed):
        """ Split dataset into training, validation, test dataloaders.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            seed (int): Seed for reproductibility.

        Returns:
            tuple: tuple of all dataloaders and the training labels.
        """
        data = copy.deepcopy(data_base)
        labels = copy.deepcopy(labels_base)
        # Get dataset of every tuple (data, label)
        subject_ids = np.asarray(list(data.keys()))

        train_subject_ids = np.delete(subject_ids,
                                      np.where(subject_ids == subject_LOPO))
        size = int(0.20 * train_subject_ids.shape[0])

        if size > 1:
            random.seed(seed)
            val_subject_ids = np.asarray(random.sample(list(train_subject_ids),
                                                       size))
        else:
            np.random.seed(seed)
            val_subject_ids = np.asarray([np.random.choice(train_subject_ids)])
        for id in val_subject_ids:
            train_subject_ids = np.delete(train_subject_ids,
                                          np.where(train_subject_ids == id))
        print('Test on: {}, '
              'Validation on: {}'.format(subject_LOPO,
                                         val_subject_ids))
        if self.data_augment == "offline":
            affine_scaling = AffineScaling(
                probability=1,  # defines the probability of modifying the input
                a_min=0.5,
                a_max=1.5,
            )

            zoom = Zoom(
                probability=1,  # defines the probability of modifying the input
                coeff=0.3,
            )

            channels_shuffle = ChannelsShuffle(
                probability=1,  # defines the probability of modifying the input
                p_shuffle=0.2
            )

            augments = [affine_scaling, zoom, channels_shuffle]
            for subject_id in train_subject_ids:
                for i in range(len(data[subject_id])):
                    n_spikes = np.sum(labels[subject_id][i])
                    n = len(labels[subject_id][i])
                    if n_spikes/n < 0.3:
                        n_no_spike_to_remove = ((n-n_spikes) - n_spikes)
                        pos_no_spike =np.where(labels[subject_id][i] == 0)[0]
                        no_spike_to_remove = np.random.choice(pos_no_spike, size=n_no_spike_to_remove, replace=False)
                        labels[subject_id][i] = np.delete(labels[subject_id][i], no_spike_to_remove)
                        data[subject_id][i] = np.delete(data[subject_id][i], no_spike_to_remove, axis=0)
                    n = len(labels[subject_id][i])
                    # if n != 0:
                    #     pos_spike = np.where(labels[subject_id][i] == 1)[0]
                    #     ratio = n_spikes/n
                    #     if ratio < 0.3:
                    #         n_data_augment = 2
                    #     elif ratio < 0.4:
                    #         n_data_augment = 1
                    #     else:
                    #         n_data_augment = 0
                    #     base_new_data = data[subject_id][i][pos_spike]
                    #     new_data = []
                    #     for _ in range(n_data_augment):
                    #         augment_data = base_new_data
                    #         for augment in augments:
                    #             augment_data = augment(augment_data)
                    #         new_data.append(augment_data)
                    #     if new_data != []:
                    #         new_data = np.concatenate(new_data)
                    #         data[subject_id][i] = np.concatenate([data[subject_id][i], new_data])
                    #         labels[subject_id][i] = np.concatenate([labels[subject_id][i], [1 for _ in range(len(new_data))]])

        # Training data
        train_data = []
        train_labels = []
        train_chan = []
        for id in train_subject_ids:
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    x = data[id][n_sess][n_trial]
                    y = labels[id][n_sess][n_trial]
                    chan = annotated_channels[id][n_sess]
                    if single_channel:
                        # Select only the channels where a spike occurs
                        if len(chan) != 0:
                            x = x[chan]
                        for i in range(x.shape[0]):
                            train_data.append(x[i])
                            train_labels.append(y)
                    else:
                        train_data.append(x)
                        train_labels.append(y)

        # Z-score normalization
        target_mean = np.mean([np.mean(data) for data in train_data])
        target_std = np.mean([np.std(data) for data in train_data])

        train_data = [np.expand_dims((data-target_mean) / target_std, axis=0)
                       for data in train_data] 
        print(len(train_labels), np.sum(train_labels)/len(train_labels))
        # Validation data
        val_data = []
        val_labels = []
        val_chan = []
        for id in val_subject_ids:
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    x = data[id][n_sess][n_trial]
                    y = labels[id][n_sess][n_trial]
                    chan = annotated_channels[id][n_sess]
                    if single_channel:
                        # Select only the channels where a spike occurs
                        if len(chan) != 0:
                            x = x[chan]
                        for i in range(x.shape[0]):
                            val_data.append(x[i])
                            val_labels.append(y)
                    else:
                        val_data.append(x)
                        val_labels.append(y)

        # Z-score normalization
        val_data = [np.expand_dims((data-target_mean) / target_std,
                                    axis=0)
                    for data in val_data] 

        # Test data
        test_data = []
        test_labels = []
        test_chan = []

        for n_sess in range(len(data[subject_LOPO])):
            for n_trial in range(len(data[subject_LOPO][n_sess])):
                x = data[subject_LOPO][n_sess][n_trial]
                y = labels[subject_LOPO][n_sess][n_trial]
                chan = annotated_channels[subject_LOPO][n_sess]
                if single_channel:
                    # Select only the channels where a spike occurs
                    if len(chan) != 0:
                        x = x[chan]
                    for i in range(x.shape[0]):
                        test_data.append(x[i])
                        test_labels.append(y)
                else:
                    test_data.append(x)
                    test_labels.append(y)

        # Z-score normalization
        test_data = [np.expand_dims((data-target_mean) / target_std,
                                     axis=0) for data in test_data]
        train_loader = self.pad_loader(train_data,
                                       train_labels,
                                       train_chan,
                                       single_channel,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       transforms=self.transforms,
                                       num_workers=num_workers)
        val_loader = self.pad_loader(val_data,
                                     val_labels,
                                     val_chan,
                                     single_channel,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     transforms=None,
                                     num_workers=num_workers)
        test_loader = self.pad_loader(test_data,
                                      test_labels,
                                      test_chan,
                                      single_channel=False,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      transforms=None,
                                      num_workers=num_workers)

        return train_loader, val_loader, test_loader, train_labels

    def train_val_test_dataloader(self,
                                  data,
                                  labels,
                                  annotated_channels,
                                  single_channel,
                                  batch_size,
                                  num_workers,
                                  seed):

        """ Split dataset into training, validation, test dataloaders.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            seed (int): Seed for reproductibility.

        Returns:
            tuple: tuple of all dataloaders and the training labels.
        """
        dataset = []
        subject_ids = np.asarray(list(data.keys()))

        for id in subject_ids:
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    x = np.expand_dims(data[id][n_sess][n_trial], axis=0)
                    y = labels[id][n_sess][n_trial]
                    chan = annotated_channels[id][n_sess]
                    dataset.append((x, y, chan))

        # Define training, validation, test splits
        N = len(dataset)
        ratio = 0.80
        train_size = int(ratio * N)
        test_size = N - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(dataset,
                                                   [train_size, test_size],
                                                   generator=generator)
        N = len(train_dataset)
        train_size = int(ratio * N)
        val_size = N - train_size
        train_dataset, val_dataset = random_split(train_dataset,
                                                  [train_size, val_size],
                                                  generator=generator)

        # Z-score normalization
        target_mean = np.mean([np.mean(data[0]) for data in train_dataset])
        target_std = np.mean([np.std(data[0]) for data in train_dataset])
        train_data, val_data, test_data = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for (x, y, chan) in train_dataset:

            x = (x-target_mean) / target_std
            if single_channel:
                # Select only the channels where a spike occurs
                if chan != []:
                    x = x[:, chan]
                for i in range(x.shape[1]):
                    train_data.append(x[:, i])
                    train_labels.append(y)

            else:
                train_data.append(x)
                train_labels.append(y)
        for (x, y, chan) in val_dataset:
            x = (x-target_mean) / target_std
            if single_channel:
                # Select only the channels where a spike occurs
                if chan != []:
                    x = x[:, chan]
                for i in range(x.shape[1]):
                    val_data.append(x[:, i])
                    val_labels.append(y)
            else:
                val_data.append(x)
                val_labels.append(y)

        for (x, y, _) in test_dataset:
            test_data.append((x-target_mean) / target_std)
            test_labels.append(y)

        train_dataset = Dataset(train_data, train_labels, transforms=self.transforms)
        val_dataset = Dataset(val_data, val_labels, transforms=None)
        test_dataset = Dataset(test_data, test_labels, transforms=None)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=PadCollate(dim=1))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=PadCollate(dim=1))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=PadCollate(dim=1))

        return train_loader, val_loader, test_loader, train_labels

    def load(self):
        if self.subject_LOPO:
            return self.LOPO_dataloader(self.data,
                                        self.labels,
                                        self.annotated_channels,
                                        self.single_channel,
                                        self.batch_size,
                                        self.num_workers,
                                        self.subject_LOPO,
                                        self.seed)
        else:
            return self.train_val_test_dataloader(self.data,
                                                  self.labels,
                                                  self.annotated_channels,
                                                  self.single_channel,
                                                  self.batch_size,
                                                  self.num_workers,
                                                  self.seed)
