def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def handle_missing_values(df):
    return df.fillna(df.mean())

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y