from sklearn.model_selection import TimeSeriesSplit

def create_time_series_splits(df,n_splits = 5):
    splitter = TimeSeriesSplit(n_splits=n_splits)

    splits= []

    for train_index, test_index in splitter.split(df):
        splits.append((train_index, test_index))

    return splits



