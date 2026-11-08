from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from prepare_data import train_arr, val_arr, val_y

for k in [2, 4, 8, 16, 20, 24, 27, 28, 29]:
    pca = PCA(n_components=k)
    pca.fit(train_arr)
    rebuilt = pca.inverse_transform(pca.transform(val_arr))
    score = ((val_arr - rebuilt) ** 2).sum(axis=1)
    train_rebuilt = pca.inverse_transform(pca.transform(train_arr))
    train_mse = ((train_arr - train_rebuilt) ** 2).mean()
    print('PCA k', k, 'train_mse', round(float(train_mse), 4), 'val_prauc', round(average_precision_score(val_y, score), 4), flush=True)
