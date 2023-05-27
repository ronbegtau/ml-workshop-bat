import wave
import struct
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd

from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import ssc
import matplotlib.pyplot as plt

count = 0


def get_audio(file, start_idx=None, end_idx=None):
    audio_file = wave.open(file)
    length = audio_file.getnframes()
    signal = np.zeros(length)
    for i in range(0, length):
        data = audio_file.readframes(1)
        data = struct.unpack("<h", data)
        signal[i] = int(data[0])
    rate = audio_file.getframerate()
    signal = np.divide(signal, float(2 ** 15))
    if start_idx is None or end_idx is None:
        return signal, rate
    return signal[start_idx:end_idx], rate


def featuresplot(sig, rate, typo, plot=False):
    global count
    count += 1
    print(count)
    m = mfcc(sig, rate, winlen=0.002048)
    fbank_feat = logfbank(sig, rate, winlen=0.002048)
    s = ssc(sig, rate, winlen=0.002048)
    mlst = []
    slst = []
    for i in range(0, len(m)):
        l = m[0:4]
        mlst.append(m[i][2])
        slst.append(s[i][4])
    m = []
    s = []
    m.append(np.mean(mlst))
    s.append(np.mean(slst))
    clst = []
    for i in range(0, len(fbank_feat)):
        l = m[0:4]
        clst.append(np.mean(fbank_feat[i]))
    c = [np.mean(clst)]
    if plot:
        plt.plot(m, c, typo)
    return s[0], m[0], c[0]


def extract_feature(path, start_idx=None, end_idx=None, typo=".y"):
    sig, rate = get_audio(path, start_idx, end_idx)
    return featuresplot(sig, rate, typo)


files = ["130604012931122519.WAV", "130604012433250328.WAV", "130604030501639406.WAV"]
for f in files:
    sig, rate = get_audio(f"../../vocs/files223/{f}")
    m = mfcc(sig, rate, winlen=0.002048, numcep=1024)
    print(m.shape)
    feat = np.mean(m, axis=0)
    print(feat)

exit(0)
annots_df = pd.read_csv("res.csv")
annots_df = annots_df[annots_df["File folder"] == "files223"]

# annots_df = annots_df.iloc[:5]

annots_df = annots_df[["File name", "Addressee", "Start sample", "End sample"]]

dataset = annots_df.apply(

    lambda x: extract_feature("../../files223/" + x["File name"], x["Start sample"], x["End sample"]), axis=1)
annots_df["features"] = dataset
# annots_df.to_csv("features.csv")
