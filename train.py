#!/usr/bin/env python

"""
This script is used to train models.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import argparse
import json
import os
import torch
import wandb

import numpy as np
import pandas as pd
import pathlib

from loguru import logger
from torch import nn
from torch.optim import Adam

from models.architectures import EEGNet_1D, RNN_self_attention
from models.architectures import EEGNet, GTN, STT, VAE
from models.training import make_model
from loader.dataloader import Loader
from loader.data import Data
from utils.losses import get_criterion
from utils.utils_ import get_pos_weight, reset_weights, get_alpha
from augmentation import AffineScaling, ChannelsShuffle, Zoom
from utils.select_subject import select_subject


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
    parser.add_argument("--architecture",
                        type=str,
                        default="RNN_self_attention")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_good_detection", type=int, default=1)
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--weight_loss", action="store_true")
    parser.add_argument("--cost_sensitive", action="store_true")
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--len_trials", type=float, default=2)
    parser.add_argument("--data_augment", type=str, default=None)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--output", type=str, default="../results")

    return parser


# Recover paths and method
parser = get_parser()
args = parser.parse_args()
path_root = args.path_root
architecture = args.architecture
batch_size = args.batch_size
lr = args.lr
n_epochs = args.n_epochs
patience = args.patience
n_good_detection = args.n_good_detection
n_subjects = args.n_subjects
weight_loss = args.weight_loss
cost_sensitive = args.cost_sensitive
lambd = args.lambd
focal = args.focal
gamma = args.gamma
alpha = args.alpha
len_trials = args.len_trials
data_augment = args.data_augment
balanced = args.balanced
if args.model_config:
    if os.path.exists(args.model_config):
        model_config = json.load(args.model_config)
    else:
        raise Exception(
            f"The model config file has to exist. {args.model_config} doesn't exist."
        )
output = pathlib.Path(args.output)

# Hardcoded training parameters
weight_decay = 0

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss
if architecture == "beta-VAE":
    criterion = None
else:
    criterion = nn.BCEWithLogitsLoss().to(device)

# Recover results
results = []
mean_acc, mean_f1, mean_precision, mean_recall = 0, 0, 0, 0
steps = 0

# Recover dataset
assert architecture in ("EEGNet",
                        "EEGNet_1D",
                        "GTN",
                        "RNN_self_attention",
                        "STT", 
                        "beta-VAE")

logger.info(f"architecture used: {architecture}")
if architecture in ["EEGNet_1D", "RNN_self_attention", "beta-VAE"]:
    single_channel = True
else:
    single_channel = False

path_subject_info = (
    "results/info_subject_{}".format(len_trials)
)

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

# Dataloader
config = vars(args)
wandb.init(project="spike_detection", config=config)

loader = Loader(data,
                labels,
                annotated_channels,
                single_channel=single_channel,
                batch_size=batch_size,
                balanced=balanced,
                data_augment=data_augment,
                transforms=transforms,
                )

train_loader, val_loader, test_loader, train_labels = loader.load()

# Define architecture
if architecture == "EEGNet":
    architecture = EEGNet()
    warmup = False
if architecture == "EEGNet_1D":
    architecture = EEGNet_1D()
    warmup = False
elif architecture == "GTN":
    n_time_points = len(data[subject_ids[0]][0][0][0])
    architecture = GTN(n_time_points=n_time_points)
elif architecture == "RNN_self_attention":
    n_time_points = len(data[subject_ids[0]][0][0][0])
    architecture = RNN_self_attention(n_time_points=n_time_points)
    warmup = False
elif architecture == "STT":
    n_time_points = len(data[subject_ids[0]][0][0][0])
    architecture = STT(n_time_points=n_time_points)
elif architecture == "beta-VAE":
    architecture = VAE(model_config)
architecture.apply(reset_weights)

if weight_loss:
    pos_weight = get_pos_weight([[train_labels]]).to(device)
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

# Define training pipeline
architecture = architecture.to(device)

if architecture == "beta-VAE":
    model = make_model(architecture,
                       train_loader,
                       val_loader,
                       optimizer,
                       train_criterion,
                       criterion,
                       single_channel=single_channel,
                       n_epochs=n_epochs,
                       patience=patience,
                       n_good_detection=n_good_detection,
                       beta=model_config["beta"],
                       unsupervised=True)
else:
    model = make_model(architecture,
                       train_loader,
                       val_loader,
                       optimizer,
                       train_criterion,
                       criterion,
                       single_channel=single_channel,
                       n_epochs=n_epochs,
                       patience=patience,
                       n_good_detection=n_good_detection)

# Train Model
best_model = model.train()

if not os.path.exists("../results"):
    os.mkdir("../results")

acc, f1, precision, recall = model.score(test_loader)
wandb.summary['test_acc'] = acc
wandb.summary['test_f1'] = f1
wandb.summary['test_precision'] = precision
wandb.summary['test_recall'] = recall
wandb.finish()

if architecture != "beta-VAE":
    print("Mean accuracy \t Mean F1-score \t Mean precision \t Mean recall")
    print("-" * 80)
    print(
        f"{mean_acc/steps:0.4f} \t {mean_f1/steps:0.4f} \t"
        f"{mean_precision/steps:0.4f} \t {mean_recall/steps:0.4f}\n"
    )
