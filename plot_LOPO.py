import argparse

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "spike detection", description="spike detection using attention layer"
    )
    parser.add_argument("--path_data", type=str, default="results/csv_LOPO_hyparameters/")
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--n_subjects", type=int, default=1)
    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()
path_data = args.path_data
n_subjects = args.n_subjects
metric = args.metric

# Choose where to load the data
# fnames = list(
#     Path(path_data).glob("results_LOPO_spike_detection"
#                          "_20_subjects.csv")
# )

# # concatene all the dataframe
# df = pd.concat([pd.read_csv(fname) for fname in fnames], axis=0)
df = pd.read_csv("/home/GRAMES.POLYMTL.CA/p117205/data_nvme_p117205/MEEG-Brainstorm/results/csv_LOPO_hyperparameters/results_LOPO_spike_detection_20-subjects.csv")
fig = plt.figure()

data = df
sns.boxplot(data=data.loc[(data["len_trials"] == 2) & (data["n_good_detection"] == 3)],
            x="method",
            y=metric,
            hue="data_augment",
            palette="Set2")

sns.swarmplot(data=data.loc[(data["len_trials"] == 2) & (data["n_good_detection"] == 3)],
              x="method",
              y=metric,
              hue="data_augment",
              dodge=True,
              palette="tab10")

plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig.savefig(
            "results/images/results_LOPO"
            "_{}_{}_subjects.png".format(metric,
                                         20),
            bbox_inches="tight",
           )
fig = plt.figure()

sns.lineplot(data=data.loc[(data["method"] == "RNN_self_attention") & (data["data_augment"] == "False")],
             x="n_good_detection",
             y=metric,
             hue="len_trials",
             palette="Set2")

plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig.savefig(
            "results/images/results_LOPO"
            "_{}_{}_subjects_lineplot_rnn.png".format(metric,
                                                  20),
            bbox_inches="tight",
           )
# print results
print(df.groupby(["method",
                  "data_augment",
                  "len_trials",
                  "n_good_detection"]).mean().reset_index())

print(df.groupby(["method",
                  "data_augment",
                  "len_trials",
                  "n_good_detection"]).std().reset_index())

mean_data = data.groupby(["method",
                  "data_augment",
                  "len_trials",
                  "n_good_detection"]).mean().reset_index()

row_max = mean_data["f1"].idxmax()
print(mean_data.iloc[row_max])
