#!/usr/bin/env python

"""
This script is used to train models.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import argparse
import os
import numpy as np
import pandas as pd

from loguru import logger
from torch.nn import BCELoss
from torch.optim import Adam

from models.architectures import RNN_self_attention, STT
from models.training import make_model
from loader.dataloader import Loader
from loader.data import Data
from utils.cost_sensitive_loss import get_criterion
from utils.utils_ import define_device, reset_weights


def get_parser():

    """ Set parameters for the experiment."""

    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection using attention layer"
    )
    parser.add_argument("--path_root", type=str, default="../BIDSdataset/")
    parser.add_argument("--method", type=str, default="RNN_self_attention")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--average", type=str, default="weighted")
    parser.add_argument("--n_windows", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--cost_sensitive", action="store_true")
    parser.add_argument("--lambd", type=float, default=0.0)

    return parser


# Recover paths and method
parser = get_parser()
args = parser.parse_args()
path_root = args.path_root
method = args.method
save = args.save
balanced = args.balanced
n_windows = args.n_windows
batch_size = args.batch_size
num_workers = args.num_workers
n_epochs = args.n_epochs
cost_sensitive = args.cost_sensitive
lambd = args.lambd

# Recover params
lr = 1e-3  # Learning rate
patience = 10
weight_decay = 0
gpu_id = 0

# Define device
available, device = define_device(gpu_id)

# Define loss
criterion = BCELoss().to(device)
train_criterion = get_criterion(criterion,
                                cost_sensitive,
                                lambd)

# Recover results
results = []

# Recover dataset
assert method in ("RNN_self_attention", "transformer_classification",
                  "transformer_detection")
logger.info(f"Method used: {method}")
if method == 'RNN_self_attention':
    single_channel = True
else:
    single_channel = False

dataset = Data(path_root, 'spikeandwave', n_windows, single_channel)
data, labels, spikes, sfreq = dataset.all_datasets()
subject_ids = np.asarray(list(data.keys()))

# Apply Leave-One-Patient-Out strategy

""" Each subject is chosen once as test set while the model is trained
    and validate on the remaining ones.
"""

for test_subject_id in subject_ids:
    data_train = []
    labels_train = []
    data_train = []
    train_subject_ids = np.delete(subject_ids,
                                  np.where(subject_ids == test_subject_id))
    val_subject_id = np.random.choice(train_subject_ids)
    train_subject_ids = np.delete(train_subject_ids,
                                  np.where(train_subject_ids
                                           == val_subject_id))
    print('Test on: {}, '
          'Validation on: {}'.format(test_subject_id,
                                     val_subject_id))
    print(train_subject_ids)
    # Training dataloader
    train_data = []
    train_labels = []
    train_spikes = []
    for id in train_subject_ids:
        train_data.append(data[id])
        train_labels.append(labels[id])
        train_spikes.append(spikes[id])

    # Z-score normalization
    target_mean = np.mean([np.mean([np.mean(data) for data in data_id])
                           for data_id in train_data])
    target_std = np.mean([np.mean([np.std(data) for data in data_id])
                          for data_id in train_data])
    train_data = [[np.expand_dims((data-target_mean) / target_std, axis=1)
                   for data in data_id] for data_id in train_data]

    # Dataloader
    if method == "transformer_detection":

        # Labels are the spike events times
        train_loader = Loader(train_data,
                              train_spikes,
                              balanced=balanced,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers)
    else:

        # Label is 1 with a spike occurs in the trial, 0 otherwise
        train_loader = Loader(train_data,
                              train_labels,
                              balanced=balanced,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers)
    train_dataloader = train_loader.load()

    # Validation dataloader
    val_data = []
    val_labels = []
    val_spikes = []
    val_data.append(data[val_subject_id])
    val_labels.append(labels[val_subject_id])
    val_spikes.append(spikes[val_subject_id])

    # Z-score normalization
    val_data = [[np.expand_dims((data-target_mean) / target_std,
                                axis=1)
                for data in data_id] for data_id in val_data]

    # Dataloader
    if method == "transformer_detection":

        # Labels are the spike events times
        val_loader = Loader(val_data,
                            val_spikes,
                            balanced=False,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)
    else:

        # Label is 1 with a spike occurs in the trial, 0 otherwise
        val_loader = Loader(val_data,
                            val_labels,
                            balanced=False,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)
    val_dataloader = val_loader.load()

    # Test dataloader
    test_data = []
    test_labels = []
    test_spikes = []
    test_data.append(data[test_subject_id])
    test_labels.append(labels[test_subject_id])
    test_spikes.append(spikes[test_subject_id])

    # Z-score normalization
    test_data = [[np.expand_dims((data-target_mean) / target_std,
                                 axis=1)
                 for data in data_id] for data_id in test_data]

    # Dataloader
    if method == "transformer_detection":

        # Labels are the spike events times
        test_loader = Loader(test_data,
                             test_spikes,
                             balanced=False,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers)
    else:

        # Label is 1 with a spike occurs in the trial, 0 otherwise
        test_loader = Loader(test_data,
                             test_labels,
                             balanced=False,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers)
    test_dataloader = test_loader.load()

    # Define architecture
    if method == "RNN_self_attention":
        architecture = RNN_self_attention()
    elif method == "transformer_classification":
        architecture = STT(n_windows=n_windows)
    elif method == "transformer_detection":
        detection = True
        architecture = STT(n_windows=n_windows, detection=detection)
    architecture.apply(reset_weights)

    # Define optimizer
    optimizer = Adam(architecture.parameters(), lr=lr,
                     weight_decay=weight_decay)

    # Define training pipeline
    architecture = architecture.to(device)
    model = make_model(architecture,
                       train_dataloader,
                       val_dataloader,
                       test_dataloader,
                       optimizer,
                       train_criterion,
                       criterion,
                       n_epochs=n_epochs,
                       patience=patience)

    # Train Model
    history = model.train()

    # Compute test performance and save it
    acc, f1, precision, recall = model.score()

    results.append(
        {
            "method": method,
            "balance": balanced,
            "test_subj_id": test_subject_id,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    )

    if save:

        # Save results file as csv
        if not os.path.exists("../results"):
            os.mkdir("../results")

        results_path = (
            "../results/csv"
        )
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        df_results = pd.DataFrame(results)
        df_results.to_csv(
            os.path.join(results_path,
                         "accuracy_results_spike_detection_method-{}"
                         "_balance-{}_{}"
                         "-subjects.csv".format(method, balanced,
                                                len(subject_ids))))
