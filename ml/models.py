import torch
import torch.nn as nn

class TabularDeepModel(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in cat_dims
        ])
        total_in_dim = (len(cat_dims) * emb_dim) + num_dim

        self.net = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_num], dim=1)
        return self.net(x).squeeze(1)
