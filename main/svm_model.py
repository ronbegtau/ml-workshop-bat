import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display
import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("207-208-215-218-220-221-231-233-TH1000-dataset.csv")
# for i in range(39, 40):
#     accs = []
#     accs.append(acc)
#     print(accs.index(max(accs)) + 1, max(accs))

emitters = set(df["Emitter"])
emitters = [230]  # emitter with good dist of samples
for emitter in emitters:
    print("Emitter:", emitter)
    try:
        emitter_df = df[df["Emitter"] == emitter]
        X, Y = emitter_df.iloc[:, :64], emitter_df["label"]
        le = LabelEncoder()
        Y = le.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        clf = svm.SVC(kernel='rbf', C=25)
        clf.fit(X_train, Y_train)
        print("N:", len(X_test))
        # y_test = Y_test[X_test["Emitter"] == emitter]
        # x_test = X_test[X_test["Emitter"] == emitter]
        # x_test = x_test.iloc[:, :64]

        acc = clf.score(X_test, Y_test)
        y_predict = clf.predict(X_test)
        cm = confusion_matrix(Y_test, y_predict)
        # cm = cm / cm.sum(axis=1)[:, None]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()

        print(str(acc * 100) + "% accuracy")
    except ValueError as ex:
        print(ex)

    print("------------------------------------------------------------------------")


# X, Y = df, df["label"]
# le = LabelEncoder()
# Y = le.fit_transform(Y)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# clf = svm.SVC(kernel='rbf', C=25)
# X_train = X_train.iloc[:, :64]
# clf.fit(X_train, Y_train)
#
# emitters = set(df["Emitter"])
# # emitters = [215]  # emitter with good dist of samples
# for emitter in emitters:
#     print("Emitter:", emitter)
#     _X_test = X_test[X_test["Emitter"] == emitter]
#     _X_test, Y_test = _X_test.iloc[:, :64], _X_test["label"]
#     le = LabelEncoder()
#     Y_test = le.fit_transform(Y_test)
#     print("N:", len(_X_test))
#     # y_test = Y_test[X_test["Emitter"] == emitter]
#     # x_test = X_test[X_test["Emitter"] == emitter]
#     # x_test = x_test.iloc[:, :64]
#
#     acc = clf.score(_X_test, Y_test)
#     y_predict = clf.predict(_X_test)
#     cm = confusion_matrix(Y_test, y_predict)
#     print(cm)
#     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
#     # disp.plot()
#     # plt.show()
#
#     print(str(acc * 100) + "% accuracy")
#
#     print("------------------------------------------------------------------------")
