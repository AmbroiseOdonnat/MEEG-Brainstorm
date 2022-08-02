#!/usr/bin/env python

"""
This script is used to train models.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import argparse
import os
import torch

import numpy as np
import pandas as pd

from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.architectures import EEGNet_1D, RNN_self_attention
from models.architectures import EEGNet, GTN, STT
from models.training import make_model
from loader.dataloader import Loader
from loader.data import Data
from utils.losses import get_criterion
from utils.utils_ import define_device, get_pos_weight
from utils.utils_ import reset_weights, get_alpha
from utils.select_subject import select_subject
from augmentation import AffineScaling, Zoom, ChannelsShuffle


def get_parser():

    """ Set parameters for the experiment."""

    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection"
    )
    parser.add_argument("--path_root",
                        type=str,
                        default="/home/GRAMES.POLYMTL.CA/p117205/"
                                "duke/projects/ivadomed/MEEG-Brainstorm/"
                                "EEG_Epilepsy_27_subjects/")
    parser.add_argument("--method", type=str, default="RNN_self_attention")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_good_detection", type=int, default=2)
    parser.add_argument("--n_subjects", type=int, default=5)
    parser.add_argument("--selected_subjects", type=str, nargs="+", default=[])
    parser.add_argument("--weight_loss", action="store_true")
    parser.add_argument("--cost_sensitive", action="store_true")
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--len_trials", type=float, default=2)
    parser.add_argument("--data_augment", type=str, default=None)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser


# Recover paths and method
parser = get_parser()
args = parser.parse_args()
path_root = args.path_root
method = args.method
save = args.save
scheduler = args.scheduler
batch_size = args.batch_size
lr = args.lr
n_epochs = args.n_epochs
n_good_detection = args.n_good_detection
n_subjects = args.n_subjects
selected_subjects = args.selected_subjects
weight_loss = args.weight_loss
cost_sensitive = args.cost_sensitive
lambd = args.lambd
focal = args.focal
gamma = args.gamma
alpha = args.alpha
len_trials = args.len_trials
data_augment = args.data_augment
balanced = args.balanced
patience = args.patience
gpu_id = args.gpu_id
# Recover params
weight_decay = 0

# Define device
available, device = define_device(gpu_id)

# Define loss
criterion = nn.BCEWithLogitsLoss().to(device)

# Recover results
results = []
mean_acc, mean_f1, mean_precision, mean_recall = 0, 0, 0, 0
steps = 0

# Recover dataset
assert method in ("EEGNet", "EEGNet_1D", "GTN",
                  "RNN_self_attention", "STT")
logger.info(f"Method used: {method}")
if method in ["EEGNet_1D", "RNN_self_attention"]:
    single_channel = True
else:
    single_channel = False

path_subject_info = (
    "results/info_subject_{}".format(len_trials)
)
if selected_subjects == []:
    selected_subjects = select_subject(n_subjects,
                                       path_subject_info,
                                       path_root,
                                       len_trials)

dataset = Data(path_root,
               "spikeandwave",
               selected_subjects,
               len_trials=len_trials)

data, labels, annotated_channels = dataset.all_datasets()
subject_ids = np.asarray(list(data.keys()))

# Define transform for data augmentation
if data_augment == "online":

    affine_scaling = AffineScaling(
        probability=0.5,  # defines the probability of modifying the input
        a_min=0.5,
        a_max=1.7,
    )

    zoom = Zoom(
        probability=0.5,  # defines the probability of modifying the input
        coeff=0.1,
    )

    channels_shuffle = ChannelsShuffle(
        probability=0.5,  # defines the probability of modifying the input
        p_shuffle=0.2
    )

    if single_channel:
        transforms = [affine_scaling, zoom]
    else:
        transforms = [affine_scaling, zoom, channels_shuffle]

elif data_augment == "offline":
    transforms = None

else:
    transforms = None
    data_augment = False


# Apply Leave-One-Patient-Out strategy

""" Each subject is chosen once as test set while the model is trained
    and validate on the remaining ones.
"""
for gen_seed in range(1):
    np.random.seed(gen_seed)
    seed_list = [np.random.randint(0, 100)
                 for _ in range(len(selected_subjects))]
    for i, test_subject_id in enumerate(subject_ids):
        seed = seed_list[i]
        # Labels are the spike events times
        loader = Loader(data,
                        labels,
                        annotated_channels,
                        single_channel=single_channel,
                        batch_size=batch_size,
                        balanced=balanced,
                        data_augment=data_augment,
                        transforms=transforms,
                        subject_LOPO=test_subject_id,
                        seed=seed,
                        )
        train_loader, val_loader, test_loader, train_labels = loader.load()

        # Define architecture
        if method == "EEGNet":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = EEGNet()
        if method == "EEGNet_1D":
            architecture = EEGNet_1D()
        elif method == "GTN":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = GTN(n_time_points=n_time_points)
        elif method == "RNN_self_attention":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = RNN_self_attention(n_time_points=n_time_points)
        elif method == "STT":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = STT(n_time_points=n_time_points)
        architecture.apply(reset_weights)

        # Define training loss
        if weight_loss:
            pos_weight = get_pos_weight(train_labels).to(device)
            train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            train_criterion = train_criterion.to(device)
        else:
            train_criterion = criterion
        if focal:
            alpha = get_alpha(train_labels)
        train_criterion = get_criterion(train_criterion,
                                        cost_sensitive,
                                        lambd,
                                        focal,
                                        alpha,
                                        gamma)
        # Define optimizer
        optimizer = Adam(architecture.parameters(),
                         lr=lr,
                         weight_decay=weight_decay)
        if scheduler:
            scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
        else:
            scheduler = None
        # Define training pipeline
        architecture = architecture.to(device)
        model = make_model(architecture,
                           train_loader,
                           val_loader,
                           optimizer,
                           scheduler,
                           train_criterion,
                           criterion,
                           single_channel=single_channel,
                           n_epochs=n_epochs,
                           patience=patience,
                           n_good_detection=n_good_detection)

        # Train Model
        best_model, history = model.train()

        torch.save(
            best_model.state_dict(),
            "results/model/model_{}_{}_{}s_{}"
            "_subjects".format(method,
                               test_subject_id,
                               len_trials,
                               len(selected_subjects)),
        )

        # Compute test performance and save it
        acc, f1, precision, recall = model.score(test_loader)
        results.append(
            {
                "method": method,
                "lr": lr,
                "batch_size": batch_size,
                "weight_loss": weight_loss,
                "cost_sensitive": cost_sensitive,
                "focal": focal,
                "len_trials": len_trials,
                "n_good_detection": n_good_detection,
                "test_subject_id": test_subject_id,
                "balanced": balanced,
                "data_augment": data_augment,
                "fold": seed,
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
        )
        mean_acc += acc
        mean_f1 += f1
        mean_precision += precision
        mean_recall += recall
        steps += 1

        if save:

            # Save results file as csv
            if not os.path.exists("results"):
                os.mkdir("results")

            results_path = (
                "results/csv_LOPO_hyparameters"
            )
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            results_path = os.path.join(results_path,
                                        "results_LOPO_spike_detection_{}"
                                        "-subjects.csv".format(
                                                        len(subject_ids))
                                        )
            df_results = pd.DataFrame(results)
            results = []

            if os.path.exists(results_path):
                df_old_results = pd.read_csv(results_path)
                df_results = pd.concat([df_old_results, df_results])
            df_results.to_csv(results_path, index=False)

    print("Mean accuracy \t Mean F1-score \t Mean precision \t Mean recall")
    print("-" * 80)
    print(
        f"{mean_acc/steps:0.4f} \t {mean_f1/steps:0.4f} \t"
        f"{mean_precision/steps:0.4f} \t {mean_recall/steps:0.4f}\n"
    )
