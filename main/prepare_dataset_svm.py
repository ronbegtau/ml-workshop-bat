import os

import pandas as pd
import numpy as np
import librosa
import librosa.display

DATASET_PATH = "dataset.csv"
VOCS_PATH = "../data/vocs/unzipped"
global count, completed
count = 0
completed = 0


def get_audio(file, start_idx=None, end_idx=None):
    signal, rate = librosa.load(file, sr=250000)
    if start_idx is None or end_idx is None:
        return signal, rate
    return signal[start_idx:end_idx], rate


def extract_feature(path, start_idx=None, end_idx=None):
    global count, total, completed
    count += 1
    if int(100 * (count / total)) > completed:
        completed = int(100 * (count / total))
        print(f"{completed}%")

    sig, rate = get_audio(path, start_idx, end_idx)
    mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=64, n_mels=64)

    return pd.Series(np.mean(mfcc, axis=1))


annots_df = pd.read_csv(DATASET_PATH)

print(len(annots_df))

total = len(annots_df)
dataset = annots_df.apply(
    lambda x: extract_feature(os.path.join(VOCS_PATH, x["File name"]), x["Start sample"], x["End sample"]), axis=1)

dataset["Recording channel"] = annots_df["Recording channel"]
dataset["label"] = annots_df["Addressee"]
dataset["Emitter"] = annots_df["Emitter"]

# ------ NORMALIZE (PER RECORDING CH) ------
means = dataset.groupby("Recording channel").mean()
dataset = dataset.merge(means, on="Recording channel")

for i in range(64):
    dataset[str(i)] = dataset[str(i) + "_x"] - dataset[str(i) + "_y"]
# ------ NORMALIZE (PER RECORDING CH) ------

dataset = dataset[[str(i) for i in range(64)] + ["Recording channel", "label_x", "Emitter_x"]]
dataset.rename({'label_x': 'label'}, axis=1, inplace=True)
dataset.rename({'Emitter_x': 'Emitter'}, axis=1, inplace=True)

print(dataset)
print(dataset.groupby(["Recording channel"]).mean())

dataset[[str(i) for i in range(64)] + ["label"] + ["Emitter"]].to_csv("dataset_svm.csv", index=False)
