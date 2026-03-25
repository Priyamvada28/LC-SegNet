import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNet34Encoder
from .decoder import Decoder
from .fsf import FSFBlock
from .sie import SIEBlock


class LCSegNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # -------- Encoder --------
        self.encoder = ResNet34Encoder(pretrained=pretrained)

        # -------- SIE Blocks (applied on x1, x2, x3) --------
        self.sie1 = SIEBlock(64)
        self.sie2 = SIEBlock(128)
        self.sie3 = SIEBlock(256)

        # -------- FSF Blocks (IMPORTANT: naming matches decoder) --------
        self.fsf3 = FSFBlock(in_channels=512 + 256, out_channels=256)  # x4 + x3
        self.fsf2 = FSFBlock(in_channels=256 + 128, out_channels=128)  # x + x2
        self.fsf1 = FSFBlock(in_channels=128 + 64, out_channels=64)    # x + x1
        self.fsf0 = FSFBlock(in_channels=64 + 64, out_channels=64)     # x + x0

        # -------- Decoder --------
        self.decoder = Decoder([
            self.fsf3,
            self.fsf2,
            self.fsf1,
            self.fsf0
        ])

        # -------- Boundary Head (only used during training) --------
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Save original input size
        input_size = x.shape[2:]

        # -------- Encoder --------
        x0, x1, x2, x3, x4 = self.encoder(x)

        # -------- Apply SIE (ONLY here) --------
        x1 = self.sie1(x1)
        x2 = self.sie2(x2)
        x3 = self.sie3(x3)


        # -------- Ensure only tensors go to decoder --------
        x0 = x0[0] if isinstance(x0, tuple) else x0
        x1 = x1[0] if isinstance(x1, tuple) else x1
        x2 = x2[0] if isinstance(x2, tuple) else x2
        x3 = x3[0] if isinstance(x3, tuple) else x3
        x4 = x4[0] if isinstance(x4, tuple) else x4


       
        # -------- Decoder --------
        # Returns both seg_mask (1 channel) and features (64 channels)
        input_size = x.shape[2:]  # original HxW
        seg_mask, features = self.decoder(x0, x1, x2, x3, x4, input_size=input_size, return_features=True)

        # print("x0:", x0.shape)
        # print("pred_mask before return:", seg_mask.shape)
      
        # -------- Boundary Head (ONLY during training) --------
        if self.training:
            boundary_map = self.boundary_head(features)
            boundary_map = F.interpolate(boundary_map, size=input_size, mode='bilinear', align_corners=False)
            return seg_mask, boundary_map

        return seg_mask