import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

FILTER = True
SELECTED_TREATMENTS = [9, 10, 16, 17, 18, 19, 20]
ADDRESSEE_SAMPLE_THRESHOLD = 1000
SAMPLE_RATE = 250000
WINDOW_SIZE = 1
SIGNAL_TOTAL_LEN = WINDOW_SIZE * SAMPLE_RATE


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
    annots_df = annots_df[annots_df["Addressee"] > 0]  # filter out unknown and negatives addressees
    annots_df = annots_df[annots_df["Emitter"] > 0]  # filter out unknown and negatives emitters

    # addressee sample count threshold
    counts = annots_df.groupby(["Addressee"]).count().iloc[:, 0]
    bats_ids = list(counts[counts > ADDRESSEE_SAMPLE_THRESHOLD].index)
    annots_df = annots_df[annots_df["Addressee"].isin(bats_ids)]

# ------ FILTERS ------

annots_df = annots_df[["File name", "Emitter", "Addressee", "Start sample", "End sample", "Recording channel"]]
annots_df.to_csv("dataset_dl.csv")
