import os
import random
import pandas as pd
import librosa
import numpy as np

# df = pd.read_csv("dataset.csv")
#
# counts = df.groupby(["Emitter"]).count().iloc[:, 0]
#
# print(counts[counts > 200])

SIG_LEN = 500000


def get_audio(file, start_idx=None, end_idx=None):
    signal, rate = librosa.load(file, sr=250000)
    if start_idx is None or end_idx is None:
        return signal, rate
    return signal[start_idx:end_idx], rate


def pad_or_trunc(signal, total_len=SIG_LEN):
    signal = signal[:total_len]
    signal = np.concatenate((signal, [0] * (total_len - len(signal))))
    return signal


#
# fp = "121110052006193927.WAV"
# a1 = 1
# b1 = 700000
# b2 = 100000
#
# data1 = get_audio("../data/vocs/unzipped/" + fp, a1, b1)
# data2 = get_audio("../data/vocs/unzipped/" + fp, a1, b2)
#
# pad1 = pad_or_trunc(data1[0])
# print(len(pad1))
# print(pad1)
#
# print("------------")
#
# pad2 = pad_or_trunc(data2[0])
# print(len(pad2))
# print(pad2)

def get_all_classes(root):
    all_files = os.listdir(root)
    classes = set()
    for fp in all_files:
        addr = fp.split("-")[-1][:-4]
        classes.add(addr)
    classes = sorted(classes)
    return classes


def split_data_set():
    root = "../data/spectograms-2"
    all_files = [fp for fp in os.listdir(root) if fp.endswith(".png")]
    random.shuffle(all_files)
    test, train = all_files[:len(all_files) // 5], all_files[len(all_files) // 5:]

    print("start test")
    for fp in test:
        os.rename(os.path.join(root, fp), os.path.join(root, "test", fp))

    print("start train")
    for fp in train:
        os.rename(os.path.join(root, fp), os.path.join(root, "train", fp))


def main():
    root = "../data/spectograms-1/test"
    all_files = os.listdir(root)
    c = 0
    emitters = set()
    for fp in all_files:
        file_id, start_frame, end_frame, emitter, addressee = fp.split("-")
        emitters.add(emitter)
    print(sorted(list(emitters)))

if __name__ == "__main__":
    main()
