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

    parser.add_argument("--method", type=str, nargs="+",
                        default=["RNN_self_attention"])
    parser.add_argument("--options", type=str, nargs="+",
                        default=[])
    parser.add_argument(
        "--training", type=str, nargs="+", default=['train']
    )
    parser.add_argument(
        "--len_trials", type=float, nargs="+", default=[2]
    )
    return parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# Experiment name
parser = get_parser()
args = parser.parse_args()
methods = args.method
options = args.options
trainings = args.training
len_trials = args.len_trials
# load data filtered
path_root = args.path_root
options = ['', '--data_augment offline', '--weight_loss', '--focal', '--focal --data_augment offline']
for training in trainings:
    for len_trial in len_trials:
        for method in methods:
            # for i, combo in enumerate(powerset(options), 1):
            #     options_combo = ''
            for option in options:


                os.system(' python {}.py'
                        ' --save --method {} --n_subjects 10 --len_trials {} {}'.format(training,
                                                        method,
                                                        len_trial,
                                                        option))
