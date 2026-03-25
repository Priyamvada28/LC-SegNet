import torch
import torch.nn as nn
import torch.nn.functional as F

class SIEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Channel MLP
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        # Offset prediction (for H and W)
        self.offset_conv = nn.Conv2d(channels, 2, kernel_size=1)

        # Projection after sampling
        self.proj = nn.Conv2d(channels, channels, 1)

        self.norm = nn.BatchNorm2d(channels)

        # Boundary prediction head (training only)
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 🔹 Channel MLP
        f_channel = self.channel_mlp(x)

        # 🔹 Predict offsets
        offsets = self.offset_conv(x)  # (B, 2, H, W)
        offset_h, offset_w = offsets[:, 0:1], offsets[:, 1:2]

        # 🔹 Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )

        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        # 🔹 Normalize offsets
        offset_w = offset_w.permute(0, 2, 3, 1)
        offset_h = offset_h.permute(0, 2, 3, 1)

        new_grid = grid + torch.cat((offset_w, offset_h), dim=-1)

        # 🔹 Sample (GLOBAL CONTEXT)
        sampled = F.grid_sample(x, new_grid, align_corners=True)

        f_spatial = self.proj(sampled)

        # 🔹 Fuse (Eqn 6)
        out = f_channel + f_spatial

        # 🔹 Residual
        out = self.norm(out + x)

        # 🔹 Boundary prediction (training only)
        boundary_pred = self.boundary_head(out)

        return out, boundary_pred