import pandas as pd


def for_score(df):
    new_df = pd.DataFrame()
    new_df['input_text'] = df.apply(lambda r: "sentence1: " + r["x1"] + " sentence2: " + r["x2"], axis=1)
    new_df['target_text'] = df['y_score']
    new_df['prefix'] = 'similarity'

    print(new_df.head())
    return new_df


def for_type(df):
    new_df = pd.DataFrame()
    new_df['input_text'] = df.apply(lambda r: "sentence1: " + r["x1"] + " sentence2: " + r["x2"], axis=1)
    new_df['target_text'] = df['y_type']
    new_df['prefix'] = 'multilabel classification'

    return new_df


