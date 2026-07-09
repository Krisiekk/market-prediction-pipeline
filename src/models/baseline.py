

def always_up_baseline(n_samples):
    return [1]*n_samples


def previous_day_direction_baseline(df):
    return (df["close"].diff()>0).astype(int)