import pandas as pd
import os
from sklearn import preprocessing


folders = ["James' Data", "Roy's Data", "Sophie's Data"]


for folder in folders:
    for file in os.listdir(folder):
        df = pd.read_csv(folder + '/' + file)
        time = df.iloc[:, 0]
        label = df.iloc[:, -1]

        window_size = 5

        df.iloc[:, 1:4] = df.iloc[:, 1:4].rolling(window_size, min_periods=1).mean()
        sc = preprocessing.StandardScaler()
        df.iloc[:, 1:4] = sc.fit_transform(df.iloc[:, 1:4])

        df.iloc[:, 0] = time
        df.iloc[:, -1] = label
        df.to_csv(folder + ' Processed' + '/' + file, index=False)

