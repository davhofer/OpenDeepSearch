import pandas as pd
import numpy as np

def stratify(df, max_samples_per_cluster: int = 20):
    idx = []
    for t in df.reasoning_types.unique():
        msk = np.arange(len(df), dtype=int)[(df.reasoning_types == t).values]
        if len(msk) > max_samples_per_cluster:
            idx.append(np.random.choice(msk, size=max_samples_per_cluster, replace=False))
        else:
            idx.append(msk)
        
    idx = np.concatenate(idx, axis=0)

    return df.iloc[idx, :]


if __name__ == "__main__":
    seed = 7
    file = "./evals/datasets/frames_test_set.csv"
    np.random.seed(seed)
    df = pd.read_csv(file)
    df = stratify(df, 20)
    df.to_csv(file.split(".csv")[0]+f"_subset_{len(df)}.csv")