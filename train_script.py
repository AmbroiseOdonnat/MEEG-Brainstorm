import argparse
import os

from itertools import chain, combinations


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Epileptic spike detection"
    )
    parser.add_argument("--path_root", type=str, default="../IvadomedNifti/")

    parser.add_argument("--len_trials", type=float, nargs="+",
                        default=[1])
    parser.add_argument("--options", type=str, nargs="+",
                        default=[])
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["STT"]
    )
    parser.add_argument(
        "--n_good_detections", type=int, nargs="+", default=[2]
    )
    parser.add_argument("--gpu_id", type=int, default=1)
    return parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# Experiment name
parser = get_parser()
args = parser.parse_args()
methods = args.methods
options = args.options
n_good_detections = args.n_good_detections
len_trials = args.len_trials
gpu_id = args.gpu_id
# load data filtered
path_root = args.path_root

for method in methods:
    for len_trial in len_trials:
        for n_good_detection in n_good_detections:
            os.system("python train_LOPO.py"
                      " --save --method {} --gpu_id {}"
                      " --balanced --n_subjects 20 "
                      "--len_trials {} --n_good_detection {}"
                      .format(method,
                              gpu_id,
                              len_trial,
                              n_good_detection))
