# models/cnn_model.py
import torch.nn as nn

class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(26*26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
