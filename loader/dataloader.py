#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch

import numpy as np

from torch.utils.data import DataLoader

from utils.utils_ import pad_tensor, weighted_sampler


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
                 balanced,
                 shuffle,
                 batch_size,
                 num_workers):

        """
        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            balanced (bool): If True, number of trials with
                             spike events is limited.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
        """

        self.data = data
        self.labels = labels
        self.balanced = balanced
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def pad_loader(self,
                   data,
                   labels,
                   shuffle,
                   batch_size,
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
        dataset = []
        for id in range(len(data)):
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    dataset.append((data[id][n_sess][n_trial],
                                    labels[id][n_sess][n_trial]))
        loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=PadCollate(dim=1))
        dataloader = np.array([loader])

        return dataloader

    def balance_pad_loader(data,
                           labels,
                           batch_size,
                           num_workers):

        """ Create dataloader of data.
            If balanced is True, number of trials with spike events is limited
            for each subject.

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.

        Returns:
            dataloader (array): Array of subject specific dataloaders .
        """

        # Get dataloader
        datasets = []
        dataloader = []
        n_ied_segments = []
        label_distributions = []
        for id in range(len(data)):
            dataset = []
            label_distribution = np.concatenate(labels[id])
            label_distributions.append(label_distribution)
            ied_segment = np.sum(label_distribution == 1)
            n_ied_segments.append(ied_segment)
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    dataset.append((data[id][n_sess][n_trial],
                                    labels[id][n_sess][n_trial]))
            datasets.append(dataset)

        # Monitor number of IEDs segments for each subject
        n_ied_segments = np.array(n_ied_segments)
        mean_n_ied_segments = np.mean(n_ied_segments)

        # Limit number of IEDs segments to the mean
        above_mean = (n_ied_segments >= mean_n_ied_segments)
        n_ied_segments[above_mean] = mean_n_ied_segments

        # Create loader for each subject
        for id in range(len(data)):
            sampler = weighted_sampler(torch.tensor(label_distributions[id]),
                                       2*n_ied_segments[id])
            loader = DataLoader(dataset=datasets[id],
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=num_workers,
                                collate_fn=PadCollate(dim=1))
            dataloader.append(loader)

        # Sort loaders in descending order
        argsort = np.argsort(n_ied_segments)
        dataloader = np.array(dataloader)[np.flipud(argsort)]

        return dataloader

    def load(self):
        if self.balanced:
            return self.balance_pad_loader(self.data,
                                           self.labels,
                                           self.batch_size,
                                           self.num_workers)
        else:
            return self.pad_loader(self.data,
                                   self.labels,
                                   self.shuffle,
                                   self.batch_size,
                                   self.num_workers)
