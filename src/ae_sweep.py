import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from prepare_data import train_arr, val_arr, val_y

torch.set_num_threads(2)

width = int(sys.argv[1])
code = int(sys.argv[2])
mode = sys.argv[3]
amount = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

train_x = torch.tensor(train_arr).float()
val_x = torch.tensor(val_arr).float()
loss_fn = torch.nn.MSELoss()


def build(width, code):
    if width == 0:
        return torch.nn.Sequential(
            torch.nn.Linear(29, code),
            torch.nn.Linear(code, 29),
        )
    return torch.nn.Sequential(
        torch.nn.Linear(29, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, code),
        torch.nn.ReLU(),
        torch.nn.Linear(code, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 29),
    )


results = []
for seed in range(1, 6):
    torch.manual_seed(seed)
    model = build(width, code)
    decay = amount if mode == 'wd' else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=decay)
    for step in range(2000):
        if mode == 'denoise':
            inputs = train_x + amount * torch.randn_like(train_x)
        else:
            inputs = train_x
        loss = loss_fn(model(inputs), train_x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    rebuilt = model(val_x).detach().numpy()
    score = ((val_arr - rebuilt) ** 2).sum(axis=1)
    ap = average_precision_score(val_y, score)
    results.append(ap)
    print('RESULT mode', mode, amount, 'width', width, 'code', code, 'seed', seed, 'train_loss', round(loss.item(), 4), 'val_prauc', round(ap, 4), flush=True)

v = np.array(results)
print('SUMMARY mode', mode, amount, 'width', width, 'code', code, 'mean', round(v.mean(), 4), 'std', round(v.std(ddof=1), 4), flush=True)
