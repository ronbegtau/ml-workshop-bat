import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df = pd.read_csv("dataset.csv")
df = df[df["label"] > 0]
print(len(df))
X, Y = df.iloc[:, :64], df["label"]
print(X)
print(Y)

le = LabelEncoder()
Y = le.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

accs = []
for i in range(1, 101):
    clf = svm.SVC(kernel='poly', C=i)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accs.append(acc)
    y_predict = clf.predict(X_test)
    print(confusion_matrix(y_test, y_predict))
    print(str(acc * 100) + "% accuracy")

print(accs.index(max(accs)), max(accs))
