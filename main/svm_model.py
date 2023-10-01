import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("dataset_svm.csv")

c = 150  # sigma parameter of RBF kernel
try:
    emitter_df = df
    X, Y = emitter_df.iloc[:, :64], emitter_df["label"]
    X["Emitter"] = emitter_df["Emitter"]
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = X_train.iloc[:, :64]
    X_test_emitters = X_test["Emitter"]
    X_test = X_test.iloc[:, :64]

    clf = svm.SVC(kernel='rbf', C=c)
    clf.fit(X_train, Y_train)
    target_names = le.inverse_transform(clf.classes_)
    target_names = [str(e) for e in target_names]
    print("Classes:", le.inverse_transform(clf.classes_))
    print("Number of test samples:", len(X_test))

    acc = clf.score(X_test, Y_test)
    Y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    # cm = cm / cm.sum(axis=1)[:, None]  # uncomment this to see CM with percentage (instead of quantity)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(clf.classes_))
    disp.plot()
    plt.show()
    print(classification_report(Y_test, Y_pred, target_names=target_names))

    print(str(acc * 100) + "% accuracy")
except ValueError as ex:
    print(ex)

output_df = pd.DataFrame({
    "addressee": le.inverse_transform(Y_test),
    "emitter": X_test_emitters,
    "prediction": le.inverse_transform(Y_pred)
})
output_df.to_csv("svm_result.csv", index=False)
