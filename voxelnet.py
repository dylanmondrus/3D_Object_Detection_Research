import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVFE(nn.Module):
    def __init__(self, input_dim=4, output_dim=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        B, V, T, C = x.shape  # (B, V, T, 4)
        x = x.view(B * V * T, C)
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(B, V, T, -1)
        x = x.max(dim=2)[0]  # (B, V, output_dim)
        return x

class VoxelNetBackbone(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv3d(x)  # (B, C, D, H, W)

class VoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = SimpleVFE()
        self.backbone = VoxelNetBackbone(input_channels=32)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # x, y, z, dx, dy, dz, heading
        )

    def forward(self, x):
        x = self.vfe(x)
        x = x.permute(0, 2, 1).contiguous()  # (B, 100, 32) → (B, 32, 100)
        x = x.view(x.shape[0], 32, 10, 10, 1)  # make it 5D for Conv3D: (B, C, D, H, W)
        x = self.backbone(x)
        x = self.classifier(x)
        return x
