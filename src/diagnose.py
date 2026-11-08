import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from prepare_data import train_arr, val_arr, val_y, features

torch.set_num_threads(2)

width = int(sys.argv[1])
code = int(sys.argv[2])
seed = int(sys.argv[3])

train_x = torch.tensor(train_arr).float()
val_x = torch.tensor(val_arr).float()
loss_fn = torch.nn.MSELoss()

torch.manual_seed(seed)
model = torch.nn.Sequential(
    torch.nn.Linear(29, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, code),
    torch.nn.ReLU(),
    torch.nn.Linear(code, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, 29),
)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
for step in range(2000):
    loss = loss_fn(model(train_x), train_x)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    dead = []
    out = train_x
    for layer in model:
        out = layer(out)
        if isinstance(layer, torch.nn.ReLU):
            dead.append(int(((out > 0).sum(dim=0) == 0).sum().item()))
    rebuilt = model(val_x).numpy()

err = (val_arr - rebuilt) ** 2
score = err.sum(axis=1)
fraud = val_y == 1
ap = average_precision_score(val_y, score)
v17 = features.index('V17')
print('DIAG width', width, 'code', code, 'seed', seed,
      'val_prauc', round(ap, 4),
      'median_normal', round(float(np.median(score[~fraud])), 3),
      'median_fraud', round(float(np.median(score[fraud])), 3),
      'max_normal', round(float(score[~fraud].max()), 1),
      'fraud_v17_error', round(float(err[fraud, v17].mean()), 2),
      'dead', '/'.join(str(d) for d in dead), flush=True)
