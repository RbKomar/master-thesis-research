import pandas as pd
import os


class CSVLoader:

    def __init__(self, data_path, year):
        self.data_path = data_path
        self.year = year

    def load(self):
        """Load CSV files containing dataset metadata based on the specified year."""
        df_list = []
        for subdir, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".csv"):
                    if "metadata" in file.lower():
                        continue
                    if self.year == 2016:
                        df = pd.read_csv(os.path.join(subdir, file), header=None)
                        df.columns = ['id', 'label']
                        df['label'] = df['label'].replace({'benign': 0, 'malignant': 1})
                    else:
                        df = pd.read_csv(os.path.join(subdir, file))
                        df.rename(columns={df.columns[0]: 'id'}, inplace=True)
                        if self.year in [2017, 2018, 2019]:
                            mel_keyword = 'melanoma' if self.year == 2017 else 'MEL'
                            df['label'] = df[mel_keyword]
                        else:  # year == 2020
                            df.rename(columns={'target': 'label'}, inplace=True)
                    df = df[['id', 'label']]
                    df.set_index('id', inplace=True)
                    df_list.append(df)
        return pd.concat(df_list)
