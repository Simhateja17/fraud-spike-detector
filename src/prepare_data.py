import os
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, '..', 'data', 'creditcard.csv')

df = pd.read_csv(csv_path)
df['Amount'] = np.log1p(df['Amount'])
train = df[df['Time'] < 86400]
val = df[(df['Time'] >= 86400) & (df['Time'] < 129600)]
test = df[df['Time'] >= 129600]
train_normal = train[train['Class'] == 0]

features = []
for i in range(1, 29):
    features.append('V' + str(i))
features.append('Amount')

mean = train_normal[features].mean()
std = train_normal[features].std()

train_arr = ((train_normal[features] - mean) / std).to_numpy()
val_arr = ((val[features] - mean) / std).to_numpy()
test_arr = ((test[features] - mean) / std).to_numpy()
val_y = val['Class'].to_numpy()
test_y = test['Class'].to_numpy()
