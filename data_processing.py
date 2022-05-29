import pandas as pd


def for_score(df):
    new_df = pd.DataFrame()
    new_df['input_text'] = df.apply(lambda r: "sentence1: " + r["x1"] + " sentence2: " + r["x2"], axis=1)
    new_df['target_text'] = df['y_score']
    new_df['prefix'] = 'stsb'  # TODO: stsb?

    return new_df


def for_type(df):
    new_df = pd.DataFrame()
    new_df['input_text'] = df.apply(lambda r: "sentence1: " + r["x1"] + " sentence2: " + r["x2"], axis=1)
    new_df['target_text'] = df['y_type']
    new_df['prefix'] = 'multilabel classification'

    return new_df


def _for_eval(df):
    result = []
    for _, row in df.iterrows():
        result.append(row['prefix'] + ": " + row['input_text'])
    return result


def for_type_eval(df):
    preprocessed = for_type(df)
    return _for_eval(preprocessed)


def for_score_eval(df):
    preprocessed = for_score(df)
    return _for_eval(preprocessed)
