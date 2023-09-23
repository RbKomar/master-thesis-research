import pandas as pd

def save_results(results, filename):
    df = pd.DataFrame(results, index=[0])
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
