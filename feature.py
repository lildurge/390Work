import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

folders = ["James' Data Processed", "Roy's Data Processed", "Sophie's Data Processed"]
folders1 = ["James' Data Processed Features", "Roy's Data Processed Features", "Sophie's Data Processed Features"]

for folder in folders:
    for file in os.listdir(folder):
        df = pd.read_csv(folder + '/' + file)
        window_size = 5
        max_time = df.iloc[-1, 0]
        num_segments = int(max_time / window_size)
        feature_names = ['segment', 'rms_x', 'std_x', 'range_x', 'median_x', 'var_x', 'skew_x', 'mean_x',
                         'rms_y', 'std_y', 'range_y', 'median_y', 'var_y', 'skew_y', 'mean_y',
                         'rms_z', 'std_z', 'range_z', 'median_z', 'var_z', 'skew_z', 'mean_z', 'label']
        features = pd.DataFrame(columns=feature_names)
        for i in range(num_segments):
            start_time = i * window_size
            end_time = start_time + window_size
            segment_data = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] < end_time)]

            segment_rms_x = np.sqrt(np.mean(segment_data.iloc[:, 1] ** 2))
            segment_std_x = segment_data.iloc[:, 1].std()
            segment_range_x = segment_data.iloc[:, 1].max() - segment_data.iloc[:, 1].min()
            segment_median_x = segment_data.iloc[:, 1].median()
            segment_var_x = segment_data.iloc[:, 1].var()
            segment_skew_x = segment_data.iloc[:, 1].skew()
            segment_mean_x = segment_data.iloc[:, 1].mean()

            segment_rms_y = np.sqrt(np.mean(segment_data.iloc[:, 2] ** 2))
            segment_std_y = segment_data.iloc[:, 2].std()
            segment_range_y = segment_data.iloc[:, 2].max() - segment_data.iloc[:, 2].min()
            segment_median_y = segment_data.iloc[:, 2].median()
            segment_var_y = segment_data.iloc[:, 2].var()
            segment_skew_y = segment_data.iloc[:, 2].skew()
            segment_mean_y = segment_data.iloc[:, 2].mean()

            segment_rms_z = np.sqrt(np.mean(segment_data.iloc[:, 3] ** 2))
            segment_std_z = segment_data.iloc[:, 3].std()
            segment_range_z = segment_data.iloc[:, 3].max() - segment_data.iloc[:, 3].min()
            segment_median_z = segment_data.iloc[:, 3].median()
            segment_var_z = segment_data.iloc[:, 3].var()
            segment_skew_z = segment_data.iloc[:, 3].skew()
            segment_mean_z = segment_data.iloc[:, 3].mean()

            segment_label = segment_data.iloc[0, -1]
            segment_features = pd.DataFrame([[i + 1,
                                              segment_rms_x, segment_std_x, segment_range_x, segment_median_x,
                                              segment_var_x, segment_skew_x, segment_mean_x,
                                              segment_rms_y, segment_std_y, segment_range_y, segment_median_y,
                                              segment_var_y, segment_skew_y, segment_mean_y,
                                              segment_rms_z, segment_std_z, segment_range_z, segment_median_z,
                                              segment_var_z, segment_skew_z, segment_mean_z,
                                              segment_label]],
                                            columns=feature_names)
            features = pd.concat([features, segment_features], ignore_index=True)

        features.to_csv(folder + ' Features' + '/' + file, index=False)

merged_df = pd.DataFrame()

for folder in folders1:
    for file in os.listdir(folder):
        df = pd.read_csv(folder + '/' + file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

merged_df.to_csv('merged_features.csv', index=False)

merged_df = pd.read_csv('merged_features.csv')
merged_df.drop(merged_df.columns[0], axis=1, inplace=True)
cols = list(merged_df.columns)
cols = [cols[-1]] + cols[:-1]
merged_df = merged_df[cols]

rows_to_shuffle = merged_df.index[1:].tolist()
random.shuffle(rows_to_shuffle)
shuffled_df = merged_df.copy()
shuffled_df.iloc[1:, :] = shuffled_df.iloc[rows_to_shuffle, :].values

shuffled_df.to_csv('merged_features.csv', index=False)

shuffled_df = shuffled_df.replace({'Jumping': 1, 'Walking': 0})

train_data, test_data = train_test_split(shuffled_df, test_size=0.1, random_state=42)
train_data = pd.concat([train_data]).reset_index(drop=True)
test_data = pd.concat([test_data]).reset_index(drop=True)

# Write the two dataframes to CSV files
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

