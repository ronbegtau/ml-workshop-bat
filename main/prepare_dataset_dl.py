import random
import time
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

SAMPLE_RATE = 250000
WINDOW_SIZE = 1
SIGNAL_TOTAL_LEN = WINDOW_SIZE * SAMPLE_RATE
DATASET_PATH = "dataset.csv"
# SPEC_OUTPUT_DIR = "../data/spectograms-1/"
SPEC_OUTPUT_DIR = "../data/TEST/"
VOCS_DIR = "../data/vocs/unzipped/"


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


annots_df = pd.read_csv(DATASET_PATH)

if len(sys.argv) < 3:
    start_row = 0
    end_row = len(annots_df)
else:
    start_row, end_row = int(sys.argv[1]), int(sys.argv[2])

print(f"Running from {start_row} to {end_row}")

c = 0
total = len(annots_df.iloc[start_row:end_row, :])
st = time.time()

random.seed(0)

for index, row in annots_df.iloc[start_row:end_row, :].iterrows():
    fp = row["File name"]
    a = row["Start sample"]
    b = row["End sample"]
    emitter = row["Emitter"]
    addr = row["Addressee"]
    r = random.random()

    # split train-test
    if r < 0.8:
        target_folder = "train"
    else:
        target_folder = "test"
    new_file_path = os.path.join(SPEC_OUTPUT_DIR, target_folder, f"{fp[:-4]}-{a}-{b}-{emitter}-{addr}" + ".png")
    spec = wav2melspec(VOCS_DIR + fp, a, b)
    spec.save(new_file_path)
    spec.close()
    c += 1
    if c % 10 == 0 or c >= total:
        et = time.time()
        print(f"{c}/{total} ({start_row} - {end_row})")
        print(et - st)
        st = et

print("done")
