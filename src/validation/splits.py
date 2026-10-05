from sklearn.model_selection import TimeSeriesSplit

def create_time_series_splits(df,n_splits = 5, gap=0):
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    splits= []

    for train_index, test_index in splitter.split(df):
        splits.append((train_index, test_index))

    return splits


