import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, fsf_blocks):
        """
        fsf_blocks: [fsf3, fsf2, fsf1, fsf0]
        """
        super().__init__()

        self.fsf3, self.fsf2, self.fsf1, self.fsf0 = fsf_blocks

        # ✅ FIXED CHANNELS
        self.dec4 = DecoderBlock(256, 256)
        self.dec3 = DecoderBlock(128, 128)
        self.dec2 = DecoderBlock(64, 64)
        self.dec1 = DecoderBlock(64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x4,input_size=None,return_features=False):
        x = x4

        # Stage 1 (x4 + x3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fsf3(x, x3)
        x = self.dec4(x)

        # Stage 2 (x + x2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fsf2(x, x2)
        x = self.dec3(x)

        # Stage 3 (x + x1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fsf1(x, x1)
        x = self.dec2(x)

        # Stage 4 (x + x0)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.fsf0(x, x0)
        x = self.dec1(x)

        features = x  # save last enriched feature map
        mask = self.final_conv(x)

         # ----- UPSAMPLE TO INPUT SIZE -----
        # ----- UPSAMPLE TO INPUT SIZE -----
        if input_size is None:
            input_size = x0.shape[2:]

        mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        if return_features:
            return mask, features
        else:
            return mask
        