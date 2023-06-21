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


fp = "121110052006193927.WAV"
a1 = 1
b1 = 700000
b2 = 100000

data1 = get_audio("../data/vocs/unzipped/" + fp, a1, b1)
data2 = get_audio("../data/vocs/unzipped/" + fp, a1, b2)

pad1 = pad_or_trunc(data1[0])
print(len(pad1))
print(pad1)

print("------------")

pad2 = pad_or_trunc(data2[0])
print(len(pad2))
print(pad2)
