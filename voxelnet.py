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
        return self.conv3d(x)  # (B, 64, D, H, W)


class RPNHead(nn.Module):
    def __init__(self, in_channels, out_channels=8):  # 7 box params + 1 score
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)  # shape: (B, 8, H, W)

        box_params = torch.zeros_like(out)
        box_params[:, 0:3, :, :] = out[:, 0:3, :, :]            # x, y, z (raw)
        box_params[:, 3:6, :, :] = torch.exp(out[:, 3:6, :, :])  # l, w, h (positive)
        box_params[:, 6, :, :] = torch.tanh(out[:, 6, :, :])     # yaw (bounded)
        box_params[:, 7, :, :] = torch.sigmoid(out[:, 7, :, :])  # score (0 to 1)

        return box_params  # shape: (B, 8, H, W)


class VoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = SimpleVFE(input_dim=4, output_dim=32)
        self.backbone = VoxelNetBackbone(input_channels=32)
        self.rpn = RPNHead(in_channels=64, out_channels=8)

    def forward(self, voxels, coords):
        x = self.vfe(voxels)  # (B, V, 32)

        B, V, C = x.shape
        D = coords[:, 0].max().item() + 1
        H = coords[:, 1].max().item() + 1
        W = coords[:, 2].max().item() + 1

        volume = torch.zeros((B, C, D, H, W), dtype=x.dtype, device=x.device)

        for v in range(V):
            z, y, x_ = coords[v].tolist()

            volume[0, :, z, y, x_] = x[0, v]


        x = self.backbone(volume)  # (B, 64, D, H, W)
        x = x.mean(dim=2)  # (B, 64, H, W)
        out = self.rpn(x)  # (B, 8, H, W)
        return out

