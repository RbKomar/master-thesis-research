import pandas as pd


class ResultsSaver:

    @staticmethod
    def save_to_csv(results, filename):
        df = pd.DataFrame(results, index=[0])
        df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
