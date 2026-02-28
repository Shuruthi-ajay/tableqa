import pandas as pd

def infer_column_types(df):
    numeric = df.select_dtypes(include=["number"]).columns
    text = df.select_dtypes(exclude=["number"]).columns
    return numeric, text

def prune_table(df, sketch):
    numeric, text = infer_column_types(df)

    if sketch["type"] == "SPAN":
        return df[text]              # blind numbers
    if sketch["type"] == "NUMBER":
        return df[numeric]           # blind long text

    return df