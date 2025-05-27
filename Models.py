import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DS(Dataset):
    def __init__(self, x, y):
        self.X = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):        
        return len(self.X)

    def __getitem__(self, i):  
        return self.X[i], self.y[i]


class ContinousModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContinousModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First layer (input -> 64 units)
        self.fc2 = nn.Linear(64, 32)         # Second layer (64 -> 32 units)
        self.fc3 = nn.Linear(32, output_dim)          # Output layer (32 -> 3 units, one for K, N, P)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)              # Output layer
        return x


class Model:
    def __init__(self, df: pd.DataFrame, targets, batch_size=64, model = ContinousModel):
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Store preprocessing info
        self.dummy_columns = df.drop(targets, axis=1).columns
        self.scaler = StandardScaler()
        # Split inputs and targets
        self.y = df[targets].values.astype(np.float32)
        self.x = df.drop(targets, axis=1).values.astype(np.float32)
        self.x = self.scaler.fit_transform(self.x)

        # Dataset and DataLoader
        self.dataset = DS(self.x, self.y)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Model
        self.model = model(self.x.shape[1], self.y.shape[1])
        self.model.to(device)
        print(f"Using {device}")

    def train(self, epochs=100, lr=1e-3):
        loss_fn = nn.MSELoss() if isinstance(self.model, ContinousModel) else nn.BCELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        for i in range(epochs):
            self.model.train()
            running = 0.0
            for xb, yb in self.data_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = self.model(xb)
                
                loss = loss_fn(pred, yb)

                optim.zero_grad()
                loss.backward()
                optim.step()
                running += loss.item() * len(xb)

            if i % (epochs // 10) == 0 or i == 0:
                avg_loss = running / len(self.data_loader.dataset)
                print(f"Epoch {i+1:02}/{epochs} · train {loss_fn.__class__.__name__} = {avg_loss:.4f}")

    def preprocess(self, new_df: pd.DataFrame) -> torch.Tensor:
        new_df = pd.get_dummies(new_df, drop_first=True)

    # Efficient column insertion:
        missing_cols = [col for col in self.dummy_columns if col not in new_df.columns]
        if missing_cols:
            zeros_df = pd.DataFrame(0, index=new_df.index, columns=missing_cols)
            new_df = pd.concat([new_df, zeros_df], axis=1)

        new_df = new_df[self.dummy_columns]

        x = self.scaler.transform(new_df.values.astype(np.float32))
        return torch.from_numpy(x).to(device)


    def eval(self, x_df: pd.DataFrame, y: torch.Tensor, *loss_fns):
        self.model.eval()
        x = self.preprocess(x_df)
        y = y.to(device)
        with torch.no_grad():
            y_pred = self.model(x)
            for fn in loss_fns:
                loss_val = fn(y_pred, y).item()
                print(f"{fn.__class__.__name__}: {loss_val:.4f}")
        
        if isinstance(self.model, ClassificationModel):
            pred_labels = (y_pred > 0.5).float()

            acc = (pred_labels == y).float().mean().item()
            print(f"Accuracy: {acc:.4f}")

    def predict(self, x_df: pd.DataFrame, return_tensor: bool = False):
        self.model.eval()
        x = self.preprocess(x_df)
        with torch.no_grad():
            preds = self.model(x)
        return preds if return_tensor else preds.cpu().numpy()


class ClassificationModel(nn.Module): 
    def __init__(self, input_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # num_classes = 3 for your case
        self.sigmoid = nn.Sigmoid()           # Add sigmoid layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)