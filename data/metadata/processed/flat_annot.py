import math
import pandas as pd


ANNOT_PATH = "Annotations.csv"
FILE_INFO_PATH = "FileInfo.csv"


annots_df = pd.read_csv(ANNOT_PATH)
file_info_df = pd.read_csv(FILE_INFO_PATH, names=[i for i in range(332)], low_memory=False)


def agg_cols(columns):
    result = []
    for column in columns:
        val = str(column)
        if val != "nan":
            result.append(int(column))

    return result


def intersect(columns):
    return list(filter(lambda x: columns["Start sample"] <= x <= columns["End sample"], columns["segments"]))


df = file_info_df.iloc[37:39, 6:]
print(df)
df2 = df.apply(agg_cols, axis=1)
print(df2)


file_info_df = file_info_df.loc[1:]
segments = file_info_df.iloc[:, 6:].apply(agg_cols, axis=1)
new_df = file_info_df.iloc[:, :6]
new_df["segments"] = segments
new_df[0] = new_df[0].astype('int64')
joined = new_df.merge(annots_df, left_on=0, right_on="FileID")
filtered_segments = joined.apply(intersect, axis=1)
joined["filtered_segments"] = filtered_segments
joined.to_csv("res.csv")

