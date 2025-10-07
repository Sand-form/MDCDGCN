import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = r"your_dataset_path.npz"
results_dir = "results_example"
os.makedirs(results_dir, exist_ok=True)

time_point = 1000
channel = 128
class_num = 2
hidden_dim = 64
batch_size = 128
lr = 1e-4
weight_decay = 1e-4
epochs = 50
k_splits = 10

data = np.load(data_path)
X = data['features']
y = data['labels']
subjects = X.shape[0]

def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channel * time_point, class_num)
    )
    return model.to(device)

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, fold):
    best_metrics = {}
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total
        print(f"[Fold {fold+1}] Epoch {epoch+1}: Train Acc={train_acc:.3f}")
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                outputs = model(Xv)
                preds.extend(outputs.argmax(1).cpu().numpy())
                trues.extend(yv.cpu().numpy())
        cm = confusion_matrix(trues, preds)
        acc = (cm[0,0]+cm[1,1]) / cm.sum()
        bal_acc = balanced_accuracy_score(trues, preds)
        print(f"  Val Acc={acc:.3f}, Bal Acc={bal_acc:.3f}")
        best_metrics = {"accuracy": acc, "balanced_accuracy": bal_acc}
    return best_metrics

kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
all_metrics = []

for fold, (train_subjects, val_subjects) in enumerate(kf.split(range(subjects))):
    print(f"\n===== Fold {fold+1}/{k_splits} =====")
    X_train = X[train_subjects].reshape(-1, channel, time_point)
    X_val = X[val_subjects].reshape(-1, channel, time_point)
    y_train = y[train_subjects].reshape(-1)
    y_val = y[val_subjects].reshape(-1)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    metrics = train_one_fold(model, train_loader, val_loader, criterion, optimizer, fold)
    all_metrics.append(metrics)
    print(f"Fold {fold+1} Metrics:", metrics)

df = pd.DataFrame(all_metrics)
df.loc['Mean'] = df.mean()
csv_path = os.path.join(results_dir, "final_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
print(df)
