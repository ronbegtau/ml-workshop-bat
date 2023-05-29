import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display

count = 0
total = 1
completed = 0
SELECTED_TREATMENTS = [9, 10, 16, 17, 18, 19, 20]
ADDRESSEE_SAMPLE_THRESHOLD = 100


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

    # try:
    sig, rate = get_audio(path, start_idx, end_idx)
    mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=64, n_mels=64)
    # except:
    # return pd.Series([0]*64)
    return pd.Series(np.mean(mfcc, axis=1))


annots_df = pd.read_csv("res.csv")
annots_df = annots_df[annots_df["Treatment ID"].isin(SELECTED_TREATMENTS)]
annots_df = annots_df[annots_df["Addressee"] > 0]
# annots_df = annots_df.iloc[:20]
# bats_ids = [218, 228, 233, 201, 203, 204, 205, 213, 225, 226]  # bats with over 100 samples

# bats with over 100 samples
counts = annots_df.groupby(["Addressee"]).count().iloc[:, 0]
bats_ids = list(counts[counts > ADDRESSEE_SAMPLE_THRESHOLD].index)
annots_df = annots_df[annots_df["Addressee"].isin(bats_ids)]

annots_df = annots_df[["File name", "Emitter", "Addressee", "Start sample", "End sample", "Recording channel"]]

total = len(annots_df)
dataset = annots_df.apply(
    lambda x: extract_feature("../data/vocs/unzipped/" + x["File name"], x["Start sample"], x["End sample"]), axis=1)

dataset["Recording channel"] = annots_df["Recording channel"]
dataset["label"] = annots_df["Addressee"]
dataset["Emitter"] = annots_df["Emitter"]
means = dataset.groupby("Recording channel").mean()
dataset = dataset.merge(means, on="Recording channel")

for i in range(64):
    dataset[str(i)] = dataset[str(i) + "_x"] - dataset[str(i) + "_y"]

dataset = dataset[[str(i) for i in range(64)] + ["Recording channel", "label_x", "Emitter_x"]]
dataset.rename({'label_x': 'label'}, axis=1, inplace=True)
dataset.rename({'Emitter_x': 'Emitter'}, axis=1, inplace=True)

print(dataset)
print(dataset.groupby(["Recording channel"]).mean())

dataset[[str(i) for i in range(64)] + ["label"] + ["Emitter"]].to_csv("dataset.csv", index=False)
dataset[[str(i) for i in range(64)] + ["label"] + ["Emitter"]].to_csv(
    "-".join([str(i) for i in bats_ids]) + "-dataset.csv", index=False)
