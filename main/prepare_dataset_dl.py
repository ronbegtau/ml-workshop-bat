import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display
from PIL import Image


FILTER = True
SELECTED_TREATMENTS = [9, 10, 16, 17, 18, 19, 20]
ADDRESSEE_SAMPLE_THRESHOLD = 1000
SAMPLE_RATE = 250000
SIGNAL_TOTAL_LEN = 1 * SAMPLE_RATE


def get_audio(file, start_idx=None, end_idx=None):
    signal, rate = librosa.load(file, sr=250000)
    if start_idx is None or end_idx is None:
        return signal, rate
    return signal[start_idx:end_idx], rate


def pad_or_trunc(signal, total_len=SIGNAL_TOTAL_LEN):
    signal = signal[:total_len]
    signal = np.concatenate((signal, [0] * (total_len - len(signal))))
    return signal


def wav2melspec(fp, start_idx=None, end_idx=None):
    y, sr = get_audio(fp, start_idx, end_idx)
    y = pad_or_trunc(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    # get current figure without white border
    img = plt.gcf()
    img.gca().xaxis.set_major_locator(plt.NullLocator())
    img.gca().yaxis.set_major_locator(plt.NullLocator())
    img.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    img.gca().xaxis.set_major_locator(plt.NullLocator())
    img.gca().yaxis.set_major_locator(plt.NullLocator())
    # to pil image
    img.canvas.draw()
    img = Image.frombytes('RGB', img.canvas.get_width_height(), img.canvas.tostring_rgb())
    plt.close()
    return img


annots_df = pd.read_csv("res.csv")

# ------ FILTERS ------
if FILTER:
    annots_df = annots_df[annots_df["Treatment ID"].isin(SELECTED_TREATMENTS)]  # only colonies
    annots_df = annots_df[annots_df["Addressee"] > 0]  # filter out unknown and negatives
    # annots_df = annots_df.iloc[:20]
    # bats_ids = [218, 228, 233, 201, 203, 204, 205, 213, 225, 226]  # bats with over 100 samples

    # bats with over 100 samples
    # counts = annots_df.groupby(["Addressee"]).count().iloc[:, 0]
    # bats_ids = list(counts[counts > ADDRESSEE_SAMPLE_THRESHOLD].index)
    # annots_df = annots_df[annots_df["Addressee"].isin(bats_ids)]

# ------ FILTERS ------

annots_df = annots_df[["File name", "Emitter", "Addressee", "Start sample", "End sample", "Recording channel"]]
annots_df.to_csv("dataset_dl.csv")

c = 1
total = len(annots_df["File name"])
import time
st = time.time()
for index, row in annots_df.iterrows():
    fp = row["File name"]
    a = row["Start sample"]
    b = row["End sample"]
    if index % 10 == 0:
        et = time.time()
        print(f"{index}/{total}")
        print(et-st)
        st = et
    spec = wav2melspec("../data/vocs/unzipped/" + fp, a, b)
    spec.save("../data/spectograms/" + f"{fp[:-4]}-{a}-{b}" + ".png")
    spec.close()
    c += 1
