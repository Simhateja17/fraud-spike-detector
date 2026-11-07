import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
df=pd.read_csv('/Users/teja/fraud-spike-detector/data/creditcard.csv')
df['Amount']=np.log1p(df['Amount'])
train=df[df['Time']<86400]
val=df[(df['Time']>=86400)&(df['Time']<129600)]
test=df[df['Time']>=129600]
train_normal=train[train['Class']==0]
print(train_normal.shape)
print(df['Amount'].max())
features=[]
for i in range(1,29):
    features.append('V'+str(i))
features.append('Amount')
print(features[1])

mean=train_normal[features].mean()
std=train_normal[features].std()

train_scaled=(train_normal[features]-mean)/std

val_scaled=(val[features]-mean)/std
test_scaled=(test[features]-mean)/std
train_arr=train_scaled.to_numpy()
val_arr=val_scaled.to_numpy()
test_arr=test_scaled.to_numpy()
val_y=val['Class'].to_numpy()
test_y=test['Class'].to_numpy()
print(val_y.sum())
pca=PCA(n_components=28)
pca.fit(train_arr)
code=pca.transform(train_arr)
train_rebuilt=pca.inverse_transform(code)
val_code=pca.transform(val_arr)
val_rebuilt=pca.inverse_transform(val_code)
val_squared_differences=[]
for n in range(len(val_rebuilt)):
    x=np.square(val_rebuilt[n]-val_arr[n])
    val_squared_differences.append(x)
val_score=np.array(val_squared_differences).sum(axis=1)
print(average_precision_score(val_y,val_score))



