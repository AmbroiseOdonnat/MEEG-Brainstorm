import argparse
import os

from itertools import chain, combinations
from sre_constants import OP_IGNORE


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Epileptic spike detection"
    )
    parser.add_argument("--path_root", type=str, default="../IvadomedNifti/")

    parser.add_argument("--lrs", type=float, nargs="+",
                        default=["RNN_self_attention"])
    parser.add_argument("--options", type=str, nargs="+",
                        default=[])
    parser.add_argument(
        "--training", type=str, nargs="+", default=['train']
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[2]
    )
    return parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# Experiment name
parser = get_parser()
args = parser.parse_args()
lrs = args.lrs
options = args.options
trainings = args.training
batch_sizes = args.batch_sizes
# load data filtered
path_root = args.path_root
for training in trainings:
    for batch_size in batch_sizes:
        for lr in lrs:
           


            os.system(' python {}.py'
                    ' --save --method RNN_self_attention --data_augment offline --n_subjects 20 --len_trials 1 --scheduler --batch_size {} --lr {}'.format(training,
                                                    batch_size,
                                                    lr))
