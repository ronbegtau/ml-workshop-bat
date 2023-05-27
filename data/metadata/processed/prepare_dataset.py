import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display

count = 0
total = 1
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


annots_df = pd.read_csv("res.csv")
annots_df = annots_df[annots_df["File folder"] == "files223"]
# annots_df = annots_df.iloc[:20]
annots_df = annots_df[["File name", "Addressee", "Start sample", "End sample", "Recording channel"]]

total = len(annots_df)
dataset = annots_df.apply(
    lambda x: extract_feature("../../files223/" + x["File name"], x["Start sample"], x["End sample"]), axis=1)

dataset["Recording channel"] = annots_df["Recording channel"]
dataset["label"] = annots_df["Addressee"]

means = dataset.groupby("Recording channel").mean()
dataset = dataset.merge(means, on="Recording channel")
print(dataset)

for i in range(64):
    dataset[str(i)] = dataset[str(i) + "_x"] - dataset[str(i) + "_y"]

dataset = dataset[[str(i) for i in range(64)] + ["Recording channel", "label_x"]]
dataset.rename({'label_x': 'label'}, axis=1, inplace=True)

print(dataset)
print(dataset.groupby(["Recording channel"]).mean())

dataset[[str(i) for i in range(64)] + ["label"]].to_csv("dataset.csv", index=False)
