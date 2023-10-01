import pandas as pd

ANNOT_PATH = "Annotations.csv"
FILE_INFO_PATH = "FileInfo.csv"

count = 0
total = 1
completed = 0
SELECTED_TREATMENTS = [9, 10, 16, 17, 18, 19, 20]
ADDRESSEE_SAMPLE_THRESHOLD = 1000

# ------ JOIN ------
annots_df = pd.read_csv(ANNOT_PATH)
file_info_df = pd.read_csv(FILE_INFO_PATH, names=[i for i in range(332)], low_memory=False)

file_info_df = file_info_df.loc[1:]
new_df = file_info_df.iloc[:, :6]
new_df[0] = new_df[0].astype('int64')
joined = new_df.merge(annots_df, left_on=0, right_on="FileID")
# ------ JOIN ------

annots_df = joined
annots_df.rename({1: "Treatment ID", 2: "File name", 4: "Recording channel"}, axis=1, inplace=True)
annots_df["Treatment ID"] = annots_df["Treatment ID"].astype("int32")
annots_df["Addressee"] = annots_df["Addressee"].astype("int32")
annots_df["Emitter"] = annots_df["Emitter"].astype("int32")

# ------ FILTERS ------
annots_df = annots_df[annots_df["Treatment ID"].isin(SELECTED_TREATMENTS)]  # only colonies
annots_df = annots_df[annots_df["Addressee"] > 0]  # filter out unknown addressees
annots_df = annots_df[annots_df["Emitter"] > 0]  # filter out unknown emitters

# addressee sample count threshold
counts = annots_df.groupby(["Addressee"]).count().iloc[:, 0]
bats_ids = list(counts[counts > ADDRESSEE_SAMPLE_THRESHOLD].index)
annots_df = annots_df[annots_df["Addressee"].isin(bats_ids)]
# ------ FILTERS ------

annots_df = annots_df[["File name", "Emitter", "Addressee", "Start sample", "End sample", "Recording channel"]]
annots_df.to_csv("dataset.csv", index=False)
