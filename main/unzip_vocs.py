import zipfile
import pandas as pd


SELECTED_TREATMENTS = [9, 10, 16, 17, 18, 19, 20]

annots_df = pd.read_csv("res.csv")

annots_df = annots_df[annots_df["Treatment ID"].isin(SELECTED_TREATMENTS)]
annots_df = annots_df[annots_df["Addressee"] > 0]

files = list("../data/vocs/" + annots_df["File folder"] + ".zip@" + annots_df["File name"])
c = 0
total = len(files)
for f in files:
    c += 1
    print(f"{c}/{total}")
    zip_file, wav_file = f.split("@")
    zipfile.ZipFile(zip_file).extract(wav_file, "../data/vocs/unzipped/")
